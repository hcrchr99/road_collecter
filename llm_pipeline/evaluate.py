# -*- coding: utf-8 -*-
"""评测框架（开发文档 §7.2）：四指标 + 对照表 + 混淆矩阵。

指标定义：
  判读一致率：真值≠遮挡/无法判定 且 判读成功 的样本中，visual_label==truth_label 的比例
              —— 门槛 ≥75%，否则只允许做「分歧提示」
  幻觉率    ：真值∈{平路(含误报), 遮挡/无法判定} 却被判成具体病害的比例
              —— 门槛 ≤10%（最危险的错误方向）；24 条误报是主考题
  JSON 合规率：一次解析成功（含 1 次重试后成功）的比例 —— 门槛 ≥95%
  不可判率  ：判「遮挡/无法判定」的比例 —— 仅报告，无门槛（诚实优于逞强）

对照组基线：auto_label（算法 autoType）按同一口径算一遍 ——「算法说 vs AI 看」。
"""
import json
import sys
from collections import Counter
from pathlib import Path

from . import config

# 具体病害类（幻觉率的判离集合）：标线与平路同为「非病害」振动来源，判成标线不算幻觉
SPECIFIC_LABELS = [l for l in config.LABELS
                   if l not in (config.BLOCKED_LABEL, "平路", "标线")]

# ---- 二分类口径（v3.1 评测结论：细粒度 11 类是难点，二分类才接近产品需求）----
# 2026-09-25 用户裁定修正：减速带、井盖是**交通设施/附属设施**，不是病害——
# 完好时压过产生振动但不构成路面退化。病害 = 路面本身的损坏/退化。
# ⚠️ 已知盲区：破损/下陷井盖是真实危害，但当前 11 类词表无法区分「完好井盖」
# 与「破损井盖」（后者往往看起来像坑洼）——留待词表 v1.3 或 severity 字段解决。
DEFECT_SET = {"坑洼", "纵向裂缝", "横向裂缝", "龟裂", "粗糙路面"}
NON_DEFECT_SET = {"设计接缝", "标线", "井盖", "减速带", "平路", config.BLOCKED_LABEL}


def binary_of(label: str) -> str:
    return "病害" if label in DEFECT_SET else "非病害"

# autoType 复合词表 → 候选真值集合（labels/schema.md §2.3 映射）。
# 未定 = 算法未做判断：不给分（计入分母恒为 miss），否则基线虚高。
IMU_LABEL_MAP = {
    "井盖/减速带": {"井盖", "减速带"},
    "疑似浅坑": {"坑洼", "平路"},
    "未定": set(),
}


def imu_label_candidates(imu_label: str):
    """算法标签 → 可与真值匹配的候选集合。普通标签即其自身。"""
    if imu_label in IMU_LABEL_MAP:
        return IMU_LABEL_MAP[imu_label]
    return {imu_label}


def load_gold(gold_path=None):
    """加载黄金集，并叠加人工仲裁修正层（若存在）。

    修正层格式同标注 CSV，按 event_id 覆盖 truth_label / is_false_positive，
    并在 note 里保留仲裁依据。原始标注文件永不修改（schema §4）。
    """
    gold_path = Path(gold_path or config.DEFAULT_GOLD)
    gold = {}
    import csv
    with open(gold_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gold[int(r["event_id"])] = {
                "truth_label": r["truth_label"].strip(),
                "is_false_positive": r["is_false_positive"] == "1",
                "confidence": r["confidence"].strip(),
                "imu_label": r["imu_label"].strip(),
                "adjudicated": False,
            }
    adj_path = Path(getattr(config, "DEFAULT_GOLD_ADJUDICATED", ""))
    if adj_path and adj_path.exists():
        with open(adj_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                eid = int(r["event_id"])
                if eid in gold:
                    if r.get("truth_label", "").strip():
                        gold[eid]["truth_label"] = r["truth_label"].strip()
                    if r.get("is_false_positive", "") != "":
                        gold[eid]["is_false_positive"] = r["is_false_positive"] == "1"
                    gold[eid]["adjudicated"] = True
                    gold[eid]["adjudicated_note"] = r.get("note", "")
    return gold


def load_reviews(model_key):
    p = config.OUTPUT_DIR / model_key / "reviews.jsonl"
    if not p.exists():
        return None
    reviews = {}
    for l in open(p, encoding="utf-8").read().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        reviews[r["event_id"]] = r
    return reviews


def metrics_for(reviews: dict, gold: dict):
    """reviews: {event_id: review}；返回指标 dict + 明细。"""
    n_total = len(gold)
    judged, parse_failed, net_failed, blocked = [], 0, 0, 0
    for eid, g in gold.items():
        r = reviews.get(eid)
        if r is None:
            continue
        src = r.get("source") or r.get("_meta", {}).get("status") or "ok"
        if src == "parse_failed" or "error" in r and "visual_label" not in r:
            parse_failed += 1
        elif src == "net_failed":
            net_failed += 1
        elif r.get("source") == "l0_blocked":
            blocked += 1
            judged.append((eid, g, config.BLOCKED_LABEL))
        else:
            judged.append((eid, g, r["visual_label"]))

    n_calls = len([r for r in reviews.values()])  # 实际发起判读的条目
    # --- 判读一致率（真值≠遮挡，排除 parse_failed/net_failed 行）
    agree_n = agree_hit = 0
    # --- 幻觉率（真值=平路 或 遮挡，却判成具体病害）
    hall_n = hall_hit = 0
    # --- 不可判率
    unable = 0
    conf_matrix = Counter()
    per_class = Counter()

    for eid, g, label in judged:
        truth = g["truth_label"]
        if label == config.BLOCKED_LABEL:
            unable += 1
        if truth != config.BLOCKED_LABEL:
            agree_n += 1
            if label == truth:
                agree_hit += 1
        if truth in ("平路", config.BLOCKED_LABEL):
            hall_n += 1
            if label in SPECIFIC_LABELS:
                hall_hit += 1
        conf_matrix[(truth, label)] += 1
        if label == truth:
            per_class[truth] += 1

    # --- autoType 基线（同一口径；复合标签按 §2.3 映射，未定不给分）
    # 分母恒为全黄金集口径（幻觉考题 = 真值∈{平路,遮挡} 全集），与被判子集无关
    base_n = base_hit = base_hall = base_hall_n = 0
    for eid, g in gold.items():
        al_set = imu_label_candidates(g["imu_label"])
        truth = g["truth_label"]
        if truth != config.BLOCKED_LABEL:
            base_n += 1
            if truth in al_set:
                base_hit += 1
        if truth in ("平路", config.BLOCKED_LABEL):
            base_hall_n += 1
            base_hall += 1 if (al_set & set(SPECIFIC_LABELS)
                               and "平路" not in al_set) else 0

    pct = lambda a, b: round(100.0 * a / b, 1) if b else None  # noqa: E731

    # --- 二分类口径（病害 vs 非病害）：产品只需要这一层，细类是加分项
    bin_n = bin_hit = 0
    missed_defect = false_defect = 0   # 漏报病害 / 把非病害判成病害
    tp_defect = 0                      # 病害真阳性
    for eid, g, label in judged:
        bt, bp = binary_of(g["truth_label"]), binary_of(label)
        bin_n += 1
        if bt == bp:
            bin_hit += 1
        if bt == "病害":
            if bp == "病害":
                tp_defect += 1
            else:
                missed_defect += 1
        elif bp == "病害":
            false_defect += 1
        if bt == bp:
            pass
    # 召回优先（项目不对称性原则：漏报不可逆，误报可由人工过滤/RHI 偏保守消化）
    defect_recall = pct(tp_defect, tp_defect + missed_defect)
    defect_precision = pct(tp_defect, tp_defect + false_defect)

    return {
        "n_gold": n_total,
        "n_judged": len(judged),
        "json_compliance": pct(n_total - parse_failed - net_failed, n_total),
        "parse_failed": parse_failed,
        "net_failed": net_failed,
        "l0_blocked": blocked,
        "agreement": pct(agree_hit, agree_n),
        "agree_detail": "%d/%d" % (agree_hit, agree_n),
        "hallucination": pct(hall_hit, hall_n),
        "hall_detail": "%d/%d" % (hall_hit, hall_n),
        "unable_rate": pct(unable, len(judged)),
        "binary_agreement": pct(bin_hit, bin_n),
        "binary_detail": "%d/%d" % (bin_hit, bin_n),
        "missed_defect": missed_defect,      # 漏报病害（二分类口径）
        "false_defect": false_defect,        # 把非病害判成病害（二分类口径）
        "defect_recall": defect_recall,      # ★ 召回优先：主指标
        "defect_precision": defect_precision,
        "baseline_agreement": pct(base_hit, base_n),
        "baseline_hallucination": pct(base_hall, base_hall_n),
        "confusion": conf_matrix,
        "per_class_hit": dict(per_class),
    }


def truth_distribution(gold):
    return Counter(g["truth_label"] for g in gold.values())


def compare(models, gold_path=None):
    gold = load_gold(gold_path)
    rows = []
    for m in models:
        reviews = load_reviews(m)
        if reviews is None:
            rows.append({"model": m, "missing": True})
            continue
        mt = metrics_for(reviews, gold)
        rows.append({"model": m, "missing": False, **{
            k: mt[k] for k in ("json_compliance", "agreement", "agree_detail",
                               "hallucination", "hall_detail", "unable_rate",
                               "binary_agreement", "binary_detail",
                               "defect_recall", "defect_precision",
                               "missed_defect", "false_defect",
                               "baseline_agreement", "baseline_hallucination",
                               "parse_failed", "net_failed", "l0_blocked")}})
    return gold, rows


def render_report(models, gold_path=None, out_path=None, notes=""):
    gold, rows = compare(models, gold_path)
    td = truth_distribution(gold)
    lines = [
        "# M-L2 对照实验 · 判读评测",
        "",
        "## 评测集",
        "- 黄金集：%s（%d 条全量人工标注，tang，2026-09-25）" % (config.DEFAULT_GOLD.name, len(gold)),
        "- 真值分布：%s" % dict(td.most_common()),
        "- 其中幻觉考题（真值=平路/遮挡）：%d 条" % (td["平路"] + td[config.BLOCKED_LABEL]),
        "",
        "## 对照表（主指标：缺陷召回率——召回优先原则，漏报不可逆、误报可人工过滤；"
        "细类一致率≥75% 为自设参考线）",
        "",
        "| 配置 | JSON合规% | 细类一致% | 幻觉% | 不可判% | 二分类% | **缺陷召回%** | 缺陷精确% | 漏报 | 假病害 | 细类(命中/分母) | parse/net_failed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["missing"]:
            lines.append("| %s | | | | | | | | | | | |" % r["model"])
        else:
            lines.append("| %s | %s | %s | %s | %s | %s | **%s** | %s | %s | %s | %s | %s/%s |" % (
                r["model"], r["json_compliance"], r["agreement"], r["hallucination"],
                r["unable_rate"], r["binary_agreement"],
                r["defect_recall"], r["defect_precision"],
                r["missed_defect"], r["false_defect"], r["agree_detail"],
                r["parse_failed"], r["net_failed"]))
    lines += [
        "",
        "基线（算法 autoType，同口径）：一致率见各模型行的 baseline 列计算，",
        "参考值：autoType 一致率 %s%%、幻觉率 %s%%（对同一黄金集）" % (
            rows[1]["baseline_agreement"] if len(rows) > 1 and not rows[1]["missing"] else "—",
            rows[1]["baseline_hallucination"] if len(rows) > 1 and not rows[1]["missing"] else "—"),
        "",
        "## 混淆矩阵与门槛判定",
        "",
        notes,
    ]

    # 每个模型的混淆矩阵
    g = load_gold(gold_path)
    for m in models:
        reviews = load_reviews(m)
        if reviews is None:
            continue
        mt = metrics_for(reviews, g)
        labels = sorted(set(t for t, _ in mt["confusion"]) |
                        set(p for _, p in mt["confusion"]),
                        key=lambda x: (x not in config.LABELS, x))
        lines += ["", "### %s 混淆矩阵（行=真值，列=判读）" % m, "",
                  "| | " + " | ".join(labels) + " |",
                  "|---" * (len(labels) + 1) + "|"]
        for t in labels:
            lines.append("| **%s** | " % t + " | ".join(
                str(mt["confusion"].get((t, p), 0)) for p in labels) + " |")
        gates = [
            ("一致率≥75%", mt["agreement"] is not None and mt["agreement"] >= 75),
            ("幻觉率≤10%", mt["hallucination"] is not None and mt["hallucination"] <= 10),
            ("JSON合规≥95%", mt["json_compliance"] is not None and mt["json_compliance"] >= 95),
        ]
        lines += ["", "门槛判定：%s" % "；".join(
            "%s %s" % (name, "✅" if ok else "❌") for name, ok in gates)]

    out_path = Path(out_path) if out_path else config.OUTPUT_DIR / "ml2_eval.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser(description="RoadCheck M-L2 评测")
    ap.add_argument("--models", nargs="+", default=list(config.MODELS.keys()))
    args = ap.parse_args()
    path, rows = render_report(args.models)
    print("评测报告：%s" % path)
    for r in rows:
        if r["missing"]:
            print("  %-14s （未跑）" % r["model"])
        else:
            print("  %-14s 一致率=%s%% 幻觉率=%s%% JSON合规=%s%% 不可判=%s%%"
                  % (r["model"], r["agreement"], r["hallucination"],
                     r["json_compliance"], r["unable_rate"]))


if __name__ == "__main__":
    main()
