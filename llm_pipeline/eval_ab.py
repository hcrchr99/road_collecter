# -*- coding: utf-8 -*-
"""A/B 对照评测（规划 §6）：基线 / 对照A(多帧) / 实验B(agent) + autoType 基线。

口径（与 M-L2 实验总结逐项对齐，跑完不许改）：
- 主指标 = 缺陷召回率（黄金集 24 条真值病害：粗糙 14 / 坑洼 7 / 纵缝 2 / 横缝 1）
- 复核队列 = 判成病害（二分类）的事件数；捕获 = tp；单位复核成本 = 复核条数 ÷ 捕获
  （v4 基线：17 捕获 + 41 假病害 = 58 条复核，58/17 = 3.4）
- 判定标准（规划 §6.3，写死）：单位复核成本 ≤3.4 且 召回显著 >70.8% => 智能体化有效；
  召回升但成本同步升 => 仅沿前沿滑动，不得作为结论。
- agent 附加队列（单列，不混入主口径）：escalate 升级 + 推导置信 low 的事件。

用法：python -m llm_pipeline.eval_ab   （输出 reports/RoadCheck_智能体判读_AB对照_<date>.md）
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from . import config, evaluate

CONFIGS = [
    ("glm-4.6v", "基线：v4 提示词 + 3 关键帧（one-shot，生产工作点）", "glm-4.6v_v4"),
    ("glm-4.6v", "对照 A：v4 + 6 帧（one-shot，能力扩展对照）", "glm-4.6v_v4f6"),
    ("glm-4.6v", "实验 B'': agent 循环终版（R1 关闭；R4=升级不改判）", "glm-4.6v_agent_b3"),
    # 参考组（迭代过程，不进判定）
    ("glm-4.6v", "参考：实验 B'（R1 关闭，R4=质询版，★负结果：R4 摧毁召回）", "glm-4.6v_agent_b2"),
    ("glm-4.6v", "参考：实验 B 第一轮（R1=post_ratio 版，★负结果：召回崩塌）", "glm-4.6v_agent_d2"),
    ("glm-4.6v", "参考：v1 保守端提示词（前沿下界）", "glm-4.6v"),
    ("deepseek-v4.1f", "参考：DeepSeek V4.1F + v4（跨厂商）", "deepseek-v4.1f_v4"),
]

# 规划 §6.3 判定线（写死）
BASELINE_RECALL = 70.8
BASELINE_COST = 3.4


def review_cost(reviews, gold):
    """复核口径：复核条数 = 判成病害的事件数；捕获 = 其中真值为病害的。"""
    n_review = captured = 0
    for eid, g in gold.items():
        r = reviews.get(eid)
        if not r or not r.get("visual_label"):
            continue
        if evaluate.binary_of(r["visual_label"]) == "病害":
            n_review += 1
            if evaluate.binary_of(g["truth_label"]) == "病害":
                captured += 1
    cost = round(n_review / captured, 2) if captured else None
    return n_review, captured, cost


def agent_extras(tag):
    """agent 附加队列与循环统计（trace/escalate）。"""
    out_dir = config.OUTPUT_DIR / tag
    traces = {}
    tp = out_dir / "trace.jsonl"
    if tp.exists():
        for l in open(tp, encoding="utf-8").read().splitlines():
            if l.strip():
                t = json.loads(l)
                traces[t["event_id"]] = t
    n_esc = 0
    ep = out_dir / "escalate.jsonl"
    if ep.exists():
        n_esc = len([l for l in open(ep, encoding="utf-8").read().splitlines() if l.strip()])
    loops = sum(1 for t in traces.values() if len(t.get("rounds", [])) > 1)
    calls = sum(len(t.get("rounds", [])) for t in traces.values())
    return {"traces": len(traces), "loops": loops, "calls": calls,
            "calls_per_event": round(calls / len(traces), 2) if traces else None,
            "escalated": n_esc}


def changed_labels(tag, reviews):
    """agent 改判链统计：开环事件中 round0 -> 终判 的标签变化。"""
    tp = config.OUTPUT_DIR / tag / "trace.jsonl"
    if not tp.exists():
        return []
    out = []
    for l in open(tp, encoding="utf-8").read().splitlines():
        if not l.strip():
            continue
        t = json.loads(l)
        oks = [r for r in t.get("rounds", []) if r.get("status") == "ok"]
        if len(oks) >= 2:
            out.append({"event_id": t["event_id"],
                        "chain": [(r.get("label"), r.get("conf_model")) for r in oks],
                        "final_conf": t.get("final_confidence")})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gold = evaluate.load_gold()
    truth = evaluate.truth_distribution(gold)
    n_defect_truth = sum(1 for g in gold.values()
                         if evaluate.binary_of(g["truth_label"]) == "病害")
    rows = []
    for _model, desc, tag in CONFIGS:
        reviews = evaluate.load_reviews(tag)
        if reviews is None:
            rows.append({"tag": tag, "desc": desc, "missing": True})
            continue
        m = evaluate.metrics_for(reviews, gold)
        n_rev, captured, cost = review_cost(reviews, gold)
        row = {"tag": tag, "desc": desc, "m": m, "n_review": n_rev,
               "captured": captured, "cost": cost}
        if "agent" in tag:
            row["extras"] = agent_extras(tag)
            row["changed"] = changed_labels(tag, reviews)
        rows.append(row)

    lines = [
        "# 智能体判读 A/B 对照报告（D3 初稿）",
        "",
        "> 生成 %s ｜ 评测集：黄金集 130 条（真值病害 %d 条：%s）" % (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            n_defect_truth,
            "、".join("%s%d" % (k, v) for k, v in sorted(truth.items())
                      if evaluate.binary_of(k) == "病害")),
        "> 口径与《LLM 判读管线实验总结》逐项对齐；复核队列 = 判成病害的事件数。",
        "> **判定标准（规划 §6.3，跑前写死）：单位复核成本 ≤%.1f 且 缺陷召回显著 >%.1f%% => 智能体化有效；"
        "召回升但成本同步升 => 仅沿前沿滑动，不得作为结论。**" % (BASELINE_COST, BASELINE_RECALL),
        "",
        "## 主对照表",
        "",
        "| 配置 | 复核条数 | 捕获 | **缺陷召回%** | 单位复核成本 | 假病害 | 漏报 | 二分类% | 幻觉% | 细类一致% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append("| %s（%s） | — | — | 运行中/缺失 | — | — | — | — | — | — |" % (r["tag"], r["desc"]))
            continue
        m = r["m"]
        lines.append("| %s | %d | %d | **%s** | %s | %d | %d | %s | %s | %s |" % (
            r["tag"], r["n_review"], r["captured"], m["defect_recall"],
            r["cost"], m["false_defect"], m["missed_defect"],
            m["binary_agreement"], m["hallucination"], m["agreement"]))
    # autoType 基线行
    lines.append("")
    lines.append("算法 autoType 同口径基线：一致率 %s、幻觉 %s（对同一黄金集）。"
                 % (rows[0]["m"]["baseline_agreement"], rows[0]["m"]["baseline_hallucination"]))

    # agent 附加分析
    agent_row = next((r for r in rows if "agent" in r.get("tag", "")), None)
    if agent_row and not agent_row.get("missing") and "extras" in agent_row:
        ex = agent_row["extras"]
        lines += [
            "",
            "## 实验 B：取证循环行为（trace 口径）",
            "",
            "- 轨迹 %d 条；开环 %d（%.0f%%）；模型调用 %d 次（均值 %s/事件，预算 ≤8）"
            % (ex["traces"], ex["loops"], 100.0 * ex["loops"] / max(ex["traces"], 1),
               ex["calls"], ex["calls_per_event"]),
            "- 升级人工复核（ESCALATE，只增不减）：%d 条" % ex["escalated"],
            "",
            "### 改判链（round0 → 终判，仅开环事件）",
            "",
            "| 事件 | 改判链（标签(模型置信)） | 终判推导置信 |",
            "|---|---|---|",
        ]
        for c in agent_row.get("changed", []):
            chain = " → ".join("%s(%s)" % (lb, cf) for lb, cf in c["chain"])
            lines.append("| #%d | %s | %s |" % (c["event_id"], chain, c["final_conf"]))

    # 判定
    lines += ["", "## 判定（按 §6.3，不事后修改）", ""]
    base = next((r for r in rows if r["tag"] == "glm-4.6v_v4" and not r.get("missing")), None)
    ag = next((r for r in rows if "agent" in r.get("tag", "") and not r.get("missing")), None)
    if base and ag:
        br = float(base["m"]["defect_recall"])
        ar = float(ag["m"]["defect_recall"])
        ac = ag["cost"]
        lines.append("- 基线召回 %.1f%% / 成本 %s；实验 B 召回 %.1f%% / 成本 %s"
                     % (br, base["cost"], ar, ac))
        if ar > br and ac is not None and ac <= BASELINE_COST:
            lines.append("- **判定：召回提升（%.1f→%.1f）且单位复核成本 %.2f ≤ %.1f，智能体化有效。**"
                         % (br, ar, ac, BASELINE_COST))
        elif ar > br:
            lines.append("- **判定：召回提升但成本同步上升（%.2f > %.1f）——按 §6.3 视为仅沿前沿滑动，"
                         "不得作为「智能体化有效」的结论。**" % (ac, BASELINE_COST))
        else:
            lines.append("- **判定：召回未提升（%.1f → %.1f），本期智能体化未达判定标准；"
                         "失败案例归因见 trace.jsonl。**" % (br, ar))
    else:
        lines.append("- 运行未齐，判定待全部配置完成后执行。")

    out = config.REPORTS_DIR / ("RoadCheck_智能体判读_AB对照_%s.md"
                                % datetime.now().strftime("%Y%m%d"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("报告 -> %s" % out)
    for r in rows:
        if r.get("missing"):
            print("  [缺失] %s" % r["tag"])
        else:
            print("  %-22s 召回 %s%% 复核 %d 成本 %s" % (r["tag"], r["m"]["defect_recall"],
                                                        r["n_review"], r["cost"]))
    return out


if __name__ == "__main__":
    main()
