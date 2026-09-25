# -*- coding: utf-8 -*-
"""L2 主力判读：逐事件 3 帧 + IMU 特征块 → JSON 判读。

- 逐事件 append 到 out/<model_key>/reviews.jsonl，重跑自动跳过已判事件（断点续跑）
- blocked 事件（L0 帧自检全部不可判）不调用模型，直接按终判写入
  （visual_label=遮挡/无法判定，confidence=high，why=帧自检结论）
- --limit N 冒烟模式（取前 N 个未判事件）

用法：
  python -m llm_pipeline.judge --model glm-4.6v [--limit 10] [--fresh]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import config
from .clients.base import get_client

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_system_prompt(prompt_file=None) -> str:
    if prompt_file is None:
        path = PROMPTS_DIR / "judge_system.md"
    else:
        path = Path(prompt_file)
        if not path.is_absolute():
            path = PROMPTS_DIR.parent / prompt_file  # 相对路径以包目录为基准
    return path.read_text(encoding="utf-8")


def build_user_text(rec: dict) -> str:
    kf = {k["role"]: k for k in rec["keyframes"]}
    a = kf.get("pre", {}).get("offset_ms", 0)
    b = kf.get("impact", {}).get("offset_ms", 0)
    c = kf.get("post_proxy", {}).get("offset_ms", 0)
    return (
        "事件 #%d。以下为车轮上方前视摄像头在同一事件的 3 帧画面\n"
        "（第 1 帧：冲击前 %s%dms；第 2 帧：冲击即时 %s%dms；第 3 帧：冲击后最近帧 %s%dms。\n"
        "负值表示该帧摄于冲击发生之前，缺陷本体通常位于画面中下部前方）。\n\n"
        "%s\n\n请按系统提示的词表与规则输出 JSON 判读。"
        % (rec["event_id"], "+" if a >= 0 else "", a, "+" if b >= 0 else "", b,
           "+" if c >= 0 else "", c, rec["imu_block"])
    )


def build_user_text_multi(rec: dict, offsets) -> str:
    """多帧版用户文本（--frames N 模式）。offsets: 按时间升序的帧偏移列表。"""
    desc = "，".join("第%d帧%s%+dms" % (i + 1, "+" if o >= 0 else "", o)
                     for i, o in enumerate(offsets))
    return (
        "事件 #%d。以下为车轮上方前视摄像头在同一事件的 %d 帧连续画面\n"
        "（%s。负值表示该帧摄于冲击发生之前，缺陷本体通常位于画面中下部前方；\n"
        "帧与帧之间路面在画面中逐渐靠近/移出，可互相印证）。\n\n"
        "%s\n\n请按系统提示的词表与规则输出 JSON 判读。"
        % (rec["event_id"], len(offsets), desc, rec["imu_block"])
    )


def blocked_review(rec: dict) -> dict:
    q = "；".join(k["quality"]["reason"] for k in rec["keyframes"] if k["quality"]["blocked"])
    return {
        "event_id": rec["event_id"],
        "visual_label": config.BLOCKED_LABEL,
        "confidence": "high",
        "why": "帧自检全部不可判（%s），按 L0 规则直接终判" % (q or "未知原因"),
        "agree_with_imu": False,
        "arbitrated": False,
        "source": "l0_blocked",
    }


def normalize_review(obj: dict, rec: dict) -> dict:
    """模型输出 → 严格契约字段（roadcheck.vision_review.v0 的 reviews[] 项）。"""
    label = str(obj.get("visual_label", "")).strip()
    if label not in config.LABELS:
        raise ValueError("visual_label 不在十类词表内: %r" % label)
    conf = str(obj.get("confidence", "")).strip().lower()
    if conf not in ("high", "mid", "low"):
        raise ValueError("confidence 非法: %r" % conf)
    return {
        "event_id": rec["event_id"],
        "visual_label": label,
        "confidence": conf,
        "why": str(obj.get("why", ""))[:300],
        "agree_with_imu": bool(obj.get("agree_with_imu")),
        "arbitrated": False,
        "source": "llm",
    }


def preprocess_frame(data: bytes, mode: str) -> bytes:
    """帧预处理（v3.1）：clahe = LAB 亮度通道自适应直方图均衡。

    依据：2026-09-25 人工查验——低对比度接缝/粗糙纹理在原图肉眼难辨，
    CLAHE 后清晰可见（_tmp_x/enhanced/ 有对比样本）。
    标线（白/黄涂料）为高亮特征，CLAHE 后仍保留。
    """
    if mode == "none":
        return data
    if mode == "clahe":
        import cv2
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return data
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else data
    raise ValueError("未知预处理模式: %r" % mode)


def judge_pack(model_key: str, pack, records, limit=None, fresh=False,
               out_tag=None, prompt_file=None, frames_n=None, preprocess="none"):
    """对已打开的 Pack 逐事件判读（持有 zip 句柄以读帧）。

    out_tag: 输出目录后缀（如 "v2" → out/glm-4.6v_v2/），用于隔离不同提示词版本的实验
    prompt_file: 系统提示词文件路径（缺省 prompts/judge_system.md）
    frames_n: 每事件喂给模型的帧数。None=3 关键帧（L0 产出）；N>3 时从包内该事件
              全部帧中均匀抽 N 帧（消除与人工标注「看全部 6 帧」的信息不对称）

    并发：按 config.MODELS[model_key]["concurrency"] 开线程池（默认 1）。
    zipfile.read 非线程安全 → 读帧用 zip_lock 串行化，仅 API 调用并发。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    dir_name = model_key + ("_" + out_tag if out_tag else "")
    out_dir = config.OUTPUT_DIR / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reviews.jsonl"

    done = set()
    if out_path.exists() and not fresh:
        for l in open(out_path, encoding="utf-8").read().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            # 只有决定性结论才算完成：失败行（net_failed/parse_failed）重跑时重试
            if r.get("visual_label") or r.get("source") == "l0_blocked":
                done.add(r["event_id"])
    elif fresh and out_path.exists():
        out_path.unlink()

    client = get_client(model_key)
    system_prompt = load_system_prompt(prompt_file)
    max_workers = max(1, int(config.MODELS[model_key].get("concurrency", 1)))
    frames_by_ev = pack.list_event_frames() if (frames_n and frames_n > 3) else None

    todo = [r for r in records if r["event_id"] not in done]
    if limit:
        todo = todo[:limit]

    stats = {"ok": 0, "parse_failed": 0, "net_failed": 0, "blocked": 0,
             "skipped": len(records) - len(todo)}
    t_start = time.time()
    write_lock = threading.Lock()
    zip_lock = threading.Lock()
    fout = open(out_path, "a", encoding="utf-8")
    done_cnt = [0]

    def one(rec):
        # 单事件异常隔离：任何未预期错误只记失败行，绝不拖垮整场运行
        try:
            return _one(rec)
        except Exception as e:  # noqa: BLE001
            review = {"event_id": rec["event_id"], "source": "net_failed",
                      "error": "unhandled %s: %s" % (type(e).__name__, e)}
            status = "net_failed"
            with write_lock:
                fout.write(json.dumps(review, ensure_ascii=False) + "\n")
                fout.flush()
                stats[status] += 1
                done_cnt[0] += 1
            return status

    def _one(rec):
        if rec["blocked"]:
            review = blocked_review(rec)
            status = "blocked"
        else:
            with zip_lock:  # zipfile.read 非线程安全
                if frames_by_ev is not None:
                    # 多帧模式：从包内该事件全部帧中均匀抽 frames_n 帧
                    fl = frames_by_ev.get(rec["event_id"], [])
                    t0 = rec["t0_ms"]
                    if len(fl) > frames_n:
                        idx = [round(i * (len(fl) - 1) / (frames_n - 1))
                               for i in range(frames_n)]
                        fl = [fl[i] for i in dict.fromkeys(idx)]
                    jpegs = [((t - rec["t0_ms"]),
                              preprocess_frame(pack.read_frame(p), preprocess))
                             for t, p in fl]
                    offsets = [t - t0 for t, _ in fl]
                    text = build_user_text_multi(rec, offsets)
                else:
                    jpegs = [(k["role"],
                              preprocess_frame(pack.read_frame(k["zip_path"]), preprocess))
                             for k in rec["keyframes"]]
                    text = build_user_text(rec)
            messages = client.build_messages(system_prompt, text, jpegs)
            obj, meta = client.call_json(messages)
            if meta["status"] == "ok":
                try:
                    review = normalize_review(obj, rec)
                    review["_meta"] = {"status": "ok", "attempts": meta["attempts"]}
                    status = "ok"
                except ValueError as e:
                    review = {"event_id": rec["event_id"], "source": "parse_failed",
                              "error": str(e), "_meta": meta}
                    status = "parse_failed"
            else:
                review = {"event_id": rec["event_id"], "source": meta["status"],
                          "error": meta.get("error", ""), "_meta": meta}
                status = ("parse_failed" if meta["status"] == "parse_failed"
                          else "net_failed")
        with write_lock:
            fout.write(json.dumps(review, ensure_ascii=False) + "\n")
            fout.flush()
            stats[status] += 1
            done_cnt[0] += 1
            if done_cnt[0] % 10 == 0 or done_cnt[0] == len(todo):
                print("  [%s] %d/%d  ok=%d parse_failed=%d net_failed=%d (%.0fs)"
                      % (model_key, done_cnt[0], len(todo), stats["ok"],
                         stats["parse_failed"], stats["net_failed"],
                         time.time() - t_start), flush=True)
        return status

    try:
        if max_workers == 1:
            for rec in todo:
                one(rec)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(one, todo))
    finally:
        fout.close()
    stats["elapsed_s"] = round(time.time() - t_start, 1)
    stats["output"] = str(out_path)
    return stats


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="RoadCheck L2 判读（M-L2）")
    ap.add_argument("--model", required=True, choices=list(config.MODELS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="清空该模型已有判读重跑")
    ap.add_argument("--tag", default=None, help="输出目录后缀，如 v2 → out/<model>_v2/")
    ap.add_argument("--prompt", default=None, help="系统提示词文件路径（缺省 judge_system.md）")
    ap.add_argument("--frames", type=int, default=None,
                    help="每事件喂给模型的帧数（>3 时从包内全部帧均匀抽样；缺省 3 关键帧）")
    ap.add_argument("--preprocess", default="none", choices=["none", "clahe"],
                    help="帧预处理：clahe = 自适应直方图均衡（低对比度接缝/纹理增强）")
    args = ap.parse_args()
    from . import l0_extract, unpack
    features_path = config.OUTPUT_DIR / "l0" / "features.jsonl"
    if not features_path.exists():
        print("先跑 run_ml1 生成 %s" % features_path)
        sys.exit(1)
    records = [json.loads(l) for l in
               open(features_path, encoding="utf-8").read().splitlines() if l.strip()]
    with unpack.Pack(config.DEFAULT_PACK) as pack:
        stats = judge_pack(args.model, pack, records, limit=args.limit,
                           fresh=args.fresh, out_tag=args.tag, prompt_file=args.prompt,
                           frames_n=args.frames, preprocess=args.preprocess)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
