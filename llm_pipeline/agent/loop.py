# -*- coding: utf-8 -*-
"""取证循环（规划 §3.2 状态机）：取证 → 判读 → 自检 →（不一致）再取证 → 终判/升级。

- 循环只对规则命中的事件展开（预估 25%~35%）；无命中的事件一轮即终判（≈基线 one-shot）。
- 预算：单事件最多 3 轮补充取证、≤8 次模型调用；超限一律 ESCALATE。
- 断点续跑：reviews.jsonl append + "只有决定性结论才算完成"（与 judge_pack 同策略）；
  agent 轨迹另行落盘 trace.jsonl，升级落盘 escalate.jsonl（复核队列只增不减）。
- 置信度由 selfcheck.derive_confidence 推导，模型自报值仅存 trace。

自检（不调 API）：
  python -m llm_pipeline.agent.loop --selftest
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from .. import config
from ..clients.base import get_client
from . import selfcheck
from .tools import AgentTools

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
# agent 默认对齐生产工作点（规划 §2.3 结论 4）：glm-4.6v + v4 提示词 + 3 关键帧
DEFAULT_PROMPT = PROMPTS_DIR / "judge_system_v4.md"

MAX_ROUNDS = 3    # 补充取证轮数上限
MAX_CALLS = 8     # 单事件模型调用上限（含首轮）

ACTION_DESCRIBE = {
    "r1_post_wave": "冲击后最近帧（中下部裁剪放大）+ 事件前后带符号波形",
    "r2_more_frames": "该事件关键帧之外的额外帧（CLAHE 增强）",
    "r3_two_frames": "2 张独立补充帧（无预处理）",
    "r4_more_evidence": "更多帧（CLAHE 增强）",
}


def _build_followup_text(rec, prev_review, fires, evidence_desc):
    ids = "；".join("%s(%s)" % (f.rule, f.reason) for f in fires)
    return (
        "事件 #%d 第 %d 轮。上一轮你的判读：%s（置信 %s；理由：%s）。\n"
        "系统自检触发：%s。\n"
        "本轮已附加新证据（%s）。请结合全部证据重新输出 JSON 判读——"
        "词表与输出规则不变；若证据仍不足，如实给低置信，不要猜。"
        % (rec["event_id"], 1 + rec.get("_round", 1),
           prev_review.get("visual_label"), prev_review.get("confidence"),
           prev_review.get("why", ""), ids, evidence_desc)
    )


def agent_one(rec, tools, client, system_prompt, state_out, zip_lock):
    """单事件取证循环。返回 (review, trace)。任何异常由调用方的 one() 包裹。"""
    from ..judge import build_user_text, blocked_review, normalize_review

    eid = rec["event_id"]
    state = {"rounds": 0, "calls": 0, "actions_done": set(),
             "usable_kf": len(tools.usable_keyframes(rec)),
             "extra_frames": 0, "last_fires": [], "frames_shown": 3}
    trace = {"event_id": eid, "rounds": [], "model_confidences": [],
             "escalations": [], "final": None}

    if rec["blocked"]:
        review = blocked_review(rec)
        review["_agent"] = {"rounds": 0, "calls": 0, "note": "L0 blocked 直通"}
        trace["final"] = "blocked"
        state_out.update(state)
        return review, trace

    # ── 第 0 轮：与基线完全同口径（v4 提示词 + 3 关键帧 + IMU 特征块）
    with zip_lock:
        jpegs = [(k["role"], pack_frame(tools, k)) for k in rec["keyframes"]]
    text = build_user_text(rec)
    review = None
    escalations = []

    while True:
        obj, meta = client.call_json(
            client.build_messages(system_prompt, text, jpegs))
        state["calls"] += 1
        if meta["status"] != "ok":
            # 网络/解析失败：计入预算，预算内重试一次同证据
            trace["rounds"].append({"round": state["rounds"], "status": meta["status"],
                                    "error": meta.get("error", "")})
            if state["calls"] >= MAX_CALLS:
                review = {"event_id": eid, "source": meta["status"],
                          "error": meta.get("error", ""), "_meta": meta}
                trace["final"] = meta["status"]
                break
            continue
        try:
            review = normalize_review(obj, rec)
        except ValueError as e:
            trace["rounds"].append({"round": state["rounds"], "status": "parse_failed",
                                    "error": str(e)})
            if state["calls"] >= MAX_CALLS:
                review = {"event_id": eid, "source": "parse_failed",
                          "error": str(e), "_meta": meta}
                trace["final"] = "parse_failed"
                break
            continue

        trace["model_confidences"].append(review["confidence"])
        trace["rounds"].append({"round": state["rounds"], "status": "ok",
                                "label": review["visual_label"],
                                "conf_model": review["confidence"]})

        # ── 自检
        fires = selfcheck.check(rec, review, state)
        state["last_fires"] = fires
        if not fires or state["rounds"] >= MAX_ROUNDS or state["calls"] >= MAX_CALLS:
            trace["final"] = "converged" if not fires else "budget_exhausted"
            break

        # 升级类动作（R2 二次取证仍不足 / R4 低置信）→ ESCALATE，只增不减，
        # review 保留当前判定（round0），由人工终审
        esc = [f for f in fires if f.action.endswith("_escalate")]
        if esc:
            escalations.append("%s：%s" % (esc[0].rule, esc[0].reason))
            trace["final"] = "escalated"
            trace["escalations"].append(escalations[-1])
            break

        # ── 执行确定性取证动作
        new_jpegs = []
        new_texts = []
        for f in fires:
            action = f.action
            if action in state["actions_done"]:
                continue
            state["actions_done"].add(action)
            if action == "r1_post_wave":
                with zip_lock:
                    data, fm = tools.read_frame(eid, offset_ms=_post_offset(rec),
                                                crop=0.6, zoom=2.0, preprocess="clahe")
                if data:
                    new_jpegs.append(("r1_post", data))
                wf = tools.waveform(eid)
                new_texts.append(wf["text"])
            elif action in ("r2_more_frames", "r4_more_evidence"):
                n = 3 if action == "r2_more_frames" else 4
                with zip_lock:
                    new_jpegs += [(("extra%d" % (i + 1)), d) for i, (_, d) in
                                  enumerate(tools.extra_frames_evidence(eid, n=n,
                                                                        preprocess="clahe"))]
            elif action == "r3_two_frames":
                with zip_lock:
                    new_jpegs += [(("ind%d" % (i + 1)), d) for i, (_, d) in
                                  enumerate(tools.extra_frames_evidence(eid, n=2,
                                                                        preprocess="none"))]
        if not new_jpegs and not new_texts:
            trace["final"] = "no_new_evidence"
            break

        state["rounds"] += 1
        state["extra_frames"] += len(new_jpegs)
        state["frames_shown"] += len(new_jpegs)
        jpegs = jpegs + new_jpegs
        desc = "、".join(ACTION_DESCRIBE.get(f.action, f.action) for f in fires)
        text = _build_followup_text(rec, review, fires,
                                    desc + ("；另有波形文本证据" if new_texts else ""))
        trace["rounds"][-1]["actions"] = [f.action for f in fires]

    # ── 终判：置信度由规则推导（模型自报值只留在 trace）
    if review is None:
        review = {"event_id": eid, "source": "net_failed", "error": "无可用判读"}
    if review.get("source") not in ("net_failed", "parse_failed"):
        # 可用帧计数近似：关键帧可用数 + 补充帧数（补充帧未逐帧自检，
        # 差额如实记录在 trace，不夸大）
        state["usable_kf_final"] = state["usable_kf"] + state["extra_frames"]
        conf, why_conf = selfcheck.derive_confidence(rec, review, state, escalations)
        old_why = review.get("why", "")
        review["confidence"] = conf
        review["why"] = ("%s｜置信推导：%s" % (old_why, why_conf))[:300]
        if escalations:
            review["why"] = ("【已升级人工复核】%s｜%s"
                             % (escalations[-1], review["why"]))[:300]
        review["_agent"] = {"rounds": state["rounds"], "calls": state["calls"],
                            "actions": sorted(state["actions_done"]),
                            "conf_model_last": (trace["model_confidences"] or [None])[-1]}
    trace["final_confidence"] = review.get("confidence")
    return review, trace


def _post_offset(rec):
    """post_proxy 关键帧的偏移（R1 取证用）；缺失取 +400ms。"""
    for k in rec["keyframes"]:
        if k["role"] == "post_proxy":
            return k.get("offset_ms") or 400
    return 400


def pack_frame(tools, kf):
    return tools.pack.read_frame(kf["zip_path"])


def run_agent_pack(model_key, pack, records, limit=None, fresh=False,
                   out_tag="agent", prompt_file=None):
    """对已打开的 Pack 跑 agent 判读。输出目录 out/<model>_<tag>/。"""
    dir_name = model_key + ("_" + out_tag if out_tag else "")
    out_dir = config.OUTPUT_DIR / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = out_dir / "reviews.jsonl"
    trace_path = out_dir / "trace.jsonl"
    escalate_path = out_dir / "escalate.jsonl"

    done = set()
    if reviews_path.exists() and not fresh:
        for l in open(reviews_path, encoding="utf-8").read().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("visual_label") or r.get("source") == "l0_blocked":
                    done.add(r["event_id"])
    elif fresh:
        for p in (reviews_path, trace_path, escalate_path):
            if p.exists():
                p.unlink()

    # 波形工具需要全量样本数组（一次性读入）
    import numpy as np
    samples = pack.read_samples()
    t_arr = np.array([int(float(r["t_ms"])) for r in samples], dtype=np.int64)
    hp = np.array([float(r["hp"]) for r in samples], dtype=float)
    vz = np.array([float(r["vz"]) for r in samples], dtype=float)
    tools = AgentTools(pack, records, t_arr, hp, vz)

    client = get_client(model_key)
    prompt_path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT
    if not prompt_path.is_absolute():
        prompt_path = Path(__file__).parent.parent / prompt_path
    system_prompt = prompt_path.read_text(encoding="utf-8")

    todo = [r for r in records if r["event_id"] not in done]
    if limit:
        todo = todo[:limit]

    stats = {"ok": 0, "parse_failed": 0, "net_failed": 0, "blocked": 0,
             "escalated": 0, "skipped": len(records) - len(todo)}
    lock = threading.Lock()
    zip_lock = threading.Lock()
    t_start = time.time()
    f_rev = open(reviews_path, "a", encoding="utf-8")
    f_tr = open(trace_path, "a", encoding="utf-8")
    f_esc = open(escalate_path, "a", encoding="utf-8")
    n_done = [0]

    def one(rec):
        try:
            return _one(rec)
        except Exception as e:  # noqa: BLE001
            review = {"event_id": rec["event_id"], "source": "net_failed",
                      "error": "unhandled %s: %s" % (type(e).__name__, e)}
            with lock:
                f_rev.write(json.dumps(review, ensure_ascii=False) + "\n")
                f_rev.flush()
                stats["net_failed"] += 1
                n_done[0] += 1
            return "net_failed"

    def _one(rec):
        review, trace = agent_one(rec, tools, client, system_prompt, {}, zip_lock)
        escalated = bool(trace.get("escalations"))
        status = ("blocked" if review.get("source") == "l0_blocked"
                  else "net_failed" if review.get("source") == "net_failed"
                  else "parse_failed" if review.get("source") == "parse_failed"
                  else "ok")
        with lock:
            f_rev.write(json.dumps(review, ensure_ascii=False) + "\n")
            f_tr.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if escalated:
                f_esc.write(json.dumps({
                    "event_id": rec["event_id"],
                    "reason": trace["escalations"],
                    "rounds": review.get("_agent", {}).get("rounds"),
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, ensure_ascii=False) + "\n")
            for f in (f_rev, f_tr, f_esc):
                f.flush()
            stats[status] += 1
            if escalated:
                stats["escalated"] += 1
            n_done[0] += 1
            if n_done[0] % 10 == 0 or n_done[0] == len(todo):
                print("  [agent:%s] %d/%d ok=%d esc=%d fail=%d (%.0fs)"
                      % (model_key, n_done[0], len(todo), stats["ok"],
                         stats["escalated"],
                         stats["parse_failed"] + stats["net_failed"],
                         time.time() - t_start), flush=True)
        return status

    for rec in todo:
        one(rec)
    for f in (f_rev, f_tr, f_esc):
        f.close()
    stats["elapsed_s"] = round(time.time() - t_start, 1)
    stats["output"] = str(out_dir)
    return stats


def selftest():
    """规则与置信度推导的自检（不调 API）。"""
    ok = True

    def feat(dur, crest, post=None):
        return {"duration_ms": dur, "crest": crest, "post_ratio": post,
                "post_ratio_positive": post is not None and post > 0.089}

    rec = {"event_id": 1, "blocked": False, "t0_ms": 0,
           "keyframes": [{"role": "pre", "quality": {"blocked": False, "lap_var": 300}},
                         {"role": "impact", "quality": {"blocked": False, "lap_var": 300}},
                         {"role": "post_proxy", "quality": {"blocked": False, "lap_var": 300}}],
           "imu_features": feat(80, 9.0, post=0.01)}
    state = {"rounds": 0, "calls": 1, "actions_done": set(), "usable_kf": 3,
             "extra_frames": 0, "last_fires": [], "frames_shown": 3}
    # R1（enforce 档）：报坑洼 + 无持续位移 → 必须火；off 档（默认）→ 不火
    selfcheck.R1_MODE = "enforce"
    fires = selfcheck.check(rec, {"visual_label": "坑洼", "confidence": "high"}, state)
    assert any(f.rule == "R1" for f in fires), "R1(enforce) 未触发"
    selfcheck.R1_MODE = "off"
    fires = selfcheck.check(rec, {"visual_label": "坑洼", "confidence": "high"}, state)
    assert not any(f.rule == "R1" for f in fires), "R1(off) 误触发"
    # R1 不因平路触发（enforce 档）
    selfcheck.R1_MODE = "enforce"
    fires = selfcheck.check(rec, {"visual_label": "平路", "confidence": "high"}, state)
    assert not any(f.rule == "R1" for f in fires), "R1 误触发"
    # R1 不因有持续位移的病害触发（真凹陷：post_ratio > 0.089）
    rec_pd = dict(rec, imu_features=feat(120, 6.0, post=0.15))
    fires = selfcheck.check(rec_pd, {"visual_label": "坑洼", "confidence": "high"}, state)
    assert not any(f.rule == "R1" for f in fires), "R1 对有持续位移的病害误触发"
    selfcheck.R1_MODE = "off"
    # R2：可用帧 0 → 火；再次 → 升级动作
    state2 = dict(state, usable_kf=0)
    fires = selfcheck.check(rec, {"visual_label": "平路", "confidence": "high"}, state2)
    assert any(f.action == "r2_more_frames" for f in fires)
    state3 = dict(state2, actions_done={"r2_more_frames"})
    fires = selfcheck.check(rec, {"visual_label": "平路", "confidence": "high"}, state3)
    assert any(f.action == "r2_escalate" for f in fires), "R2 未升级"
    # R3：弱证据 + 报病害 → 火
    rec3 = dict(rec, imu_features=feat(200, 1.5))
    fires = selfcheck.check(rec3, {"visual_label": "纵向裂缝", "confidence": "mid"}, state)
    assert any(f.rule == "R3" for f in fires)
    # R4：模型自报 low → 升级（不改判）
    fires = selfcheck.check(rec, {"visual_label": "平路", "confidence": "low"}, state)
    assert any(f.rule == "R4" and f.action == "r4_escalate" for f in fires), "R4 未升级"

    # 置信度推导
    st_ok = dict(state, last_fires=[], usable_kf_final=3)
    conf, why = selfcheck.derive_confidence(rec, {"visual_label": "坑洼"}, st_ok, [])
    assert conf == "high", why
    st_r1 = dict(state, last_fires=[selfcheck.RuleFire("R1", "x", "r1_post_wave")],
                 usable_kf_final=3)
    conf, _ = selfcheck.derive_confidence(rec, {"visual_label": "坑洼"}, st_r1, [])
    assert conf == "low"
    st_esc = dict(state, last_fires=[], usable_kf_final=0)
    conf, _ = selfcheck.derive_confidence(rec, {"visual_label": "坑洼"}, st_esc,
                                          ["R2：可用帧 ≤1"])
    assert conf == "low"
    print("agent.selftest OK：R1–R4 触发/不触发、R2 升级、置信度推导 4 条路径全部通过")
    return ok


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("用法：python -m llm_pipeline.agent.loop --selftest")
