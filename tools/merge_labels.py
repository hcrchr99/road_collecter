#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注合并与一致性检查

这是多 agent 协作的防冲突工具。它只做「读多个文件 → 产出新文件」，
从不修改任何输入，所以任意多个 agent 同时跑它都是安全的。

三种模式：

    # 1) 合并：把多份标注汇总成一份（只读输入，输出新文件）
    python tools/merge_labels.py --mode merge \\
           --labels raw/a.labels.csv raw/b.labels.csv \\
           --out raw/labels.all.csv

    # 2) 校验：检查标注文件是否符合 schema
    python tools/merge_labels.py --mode validate --labels raw/a.labels.csv

    # 3) 一致性：两份独立标注的对比 + Cohen's Kappa
    python tools/merge_labels.py --mode kappa \\
           --labels raw/a.labels.part-alice.csv raw/a.labels.part-bob.csv

依赖：无（标准库即可）
"""

import argparse
import csv
import os
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- schema v1.1 定义（与 labels/schema.md 保持一致）----

TRUTH_LABELS = {
    "坑洼", "减速带", "井盖", "设计接缝", "粗糙路面",
    "纵向裂缝", "横向裂缝", "龟裂",
    "平路", "遮挡/无法判定",
}

IMU_EVENT_FIELDS = [
    "event_id", "t_start_ms", "t_end_ms", "peak", "imu_label",
    "truth_label", "is_false_positive", "confidence", "note",
    "labeled_by", "labeled_at",
]

FRAME_FIELDS = [
    "frame_id", "t_ms", "source", "vision_label", "truth_label",
    "severity", "occlusion", "confidence", "note",
    "labeled_by", "labeled_at",
]

CONFIDENCE = {"high", "mid", "low"}
SEVERITY = {"light", "medium", "severe"}


def detect_kind(fieldnames):
    """按字段集合判断这是事件标注还是帧标注。"""
    f = set(fieldnames or [])
    if {"event_id", "imu_label"} & f:
        return "event"
    if {"frame_id", "vision_label"} & f:
        return "frame"
    return "unknown"


def load_csv(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def atomic_write_csv(rows, fieldnames, path):
    """原子写：先写临时文件再替换。避免别的 agent 读到半截文件。"""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "." + os.path.basename(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


# ---------------------------------------------------------------- validate

def cmd_validate(args):
    total_issues = 0
    for path in args.labels:
        fields, rows = load_csv(path)
        name = os.path.basename(path)
        if fields is None:
            print("✗ %s —— 文件不存在" % name)
            total_issues += 1
            continue

        kind = detect_kind(fields)
        issues = []

        if kind == "unknown":
            issues.append("无法识别的字段结构（既不像事件标注也不像帧标注）")
            print("✗ %s" % name)
            for i in issues:
                print("    %s" % i)
            total_issues += 1
            continue

        expected = IMU_EVENT_FIELDS if kind == "event" else FRAME_FIELDS
        missing = [f for f in expected if f not in (fields or [])]
        if missing:
            issues.append("缺少字段：%s" % ", ".join(missing))

        key = "event_id" if kind == "event" else "frame_id"
        seen = set()
        for i, r in enumerate(rows, start=2):
            rid = r.get(key, "")
            if not rid:
                issues.append("第 %d 行：%s 为空" % (i, key))
            elif rid in seen:
                issues.append("第 %d 行：%s=%s 重复" % (i, key, rid))
            else:
                seen.add(rid)

            tl = (r.get("truth_label") or "").strip()
            if not tl:
                issues.append("第 %d 行：truth_label 为空" % i)
            elif tl not in TRUTH_LABELS:
                issues.append("第 %d 行：truth_label「%s」不在取值域内" % (i, tl))

            cf = (r.get("confidence") or "").strip()
            if cf and cf not in CONFIDENCE:
                issues.append("第 %d 行：confidence「%s」应为 high/mid/low" % (i, cf))

            if kind == "event":
                fp = (r.get("is_false_positive") or "").strip()
                if fp not in ("0", "1"):
                    issues.append("第 %d 行：is_false_positive 应为 0 或 1" % i)
            else:
                sv = (r.get("severity") or "").strip()
                if sv and sv not in SEVERITY:
                    issues.append("第 %d 行：severity「%s」应为 light/medium/severe" % (i, sv))
                oc = (r.get("occlusion") or "").strip()
                if oc and oc not in ("0", "1"):
                    issues.append("第 %d 行：occlusion 应为 0 或 1" % i)

        if issues:
            print("⚠ %s（%s，%d 行）" % (name, kind, len(rows)))
            for i in issues[:20]:
                print("    %s" % i)
            if len(issues) > 20:
                print("    ... 还有 %d 条" % (len(issues) - 20))
            total_issues += len(issues)
        else:
            print("✓ %s（%s，%d 行）" % (name, kind, len(rows)))

    print("")
    if total_issues:
        print("共发现 %d 个问题。请修正后再合并。" % total_issues)
        return 1
    print("全部通过。")
    return 0


# ---------------------------------------------------------------- merge

def cmd_merge(args):
    all_rows = []
    kind = None
    sources = []

    for path in args.labels:
        fields, rows = load_csv(path)
        if fields is None:
            print("✗ 文件不存在：%s" % path)
            return 1
        k = detect_kind(fields)
        if k == "unknown":
            print("✗ 无法识别字段结构：%s" % path)
            return 1
        if kind is None:
            kind = k
        elif kind != k:
            print("✗ 类型不一致：%s 是 %s，但前面的是 %s" % (path, k, kind))
            print("   事件标注与帧标注不能合并到一个文件，请分开处理。")
            return 1
        sources.append((os.path.basename(path), len(rows)))
        for r in rows:
            r["_source_file"] = os.path.basename(path)
            all_rows.append(r)

    key = "event_id" if kind == "event" else "frame_id"

    # 冲突检测：同一 key 出现在多个来源，且 truth_label 不同
    by_key = {}
    for r in all_rows:
        k = r.get(key, "")
        by_key.setdefault(k, []).append(r)

    conflicts = []
    for k, rs in by_key.items():
        labels = {(x.get("truth_label") or "").strip() for x in rs}
        if len(labels) > 1:
            conflicts.append((k, sorted(labels),
                              sorted({x["_source_file"] for x in rs})))

    out_fields = (IMU_EVENT_FIELDS if kind == "event" else FRAME_FIELDS) + ["_source_file"]
    all_rows.sort(key=lambda r: (str(r.get(key, ""))))

    atomic_write_csv(all_rows, out_fields, args.out)

    print("合并完成 → %s" % args.out)
    print("  类型：%s" % ("IMU 事件" if kind == "event" else "视觉帧"))
    print("  来源：")
    for n, c in sources:
        print("    %-46s %d 行" % (n, c))
    print("  合计 %d 行" % len(all_rows))

    if conflicts:
        print("")
        print("  ⚠ 检测到 %d 处真值冲突（同一 %s 被标成了不同结果）：" % (len(conflicts), key))
        for k, labels, files in conflicts[:15]:
            print("    %s → %s  来自 %s" % (k, " / ".join(labels), ", ".join(files)))
        if len(conflicts) > 15:
            print("    ... 还有 %d 处" % (len(conflicts) - 15))
        print("")
        print("  处理方式：逐条讨论后把结论写进单独的 adjudicated 文件，")
        print("           不要直接改合并结果。参见 labels/schema.md 第 4 节。")
    else:
        print("  无真值冲突。")
    return 0


# ---------------------------------------------------------------- kappa

def cohen_kappa(pairs):
    """pairs: [(a, b), ...]。返回 (kappa, po, pe, n)。"""
    n = len(pairs)
    if n == 0:
        return None, None, None, 0
    cats = sorted({x for p in pairs for x in p})
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    return kappa, po, pe, n


def cmd_kappa(args):
    if len(args.labels) != 2:
        print("kappa 模式需要恰好两个标注文件（两份独立标注）")
        return 1

    (f1, r1), (f2, r2) = (load_csv(args.labels[0]), load_csv(args.labels[1]))
    if f1 is None or f2 is None:
        print("文件不存在")
        return 1

    k1, k2 = detect_kind(f1), detect_kind(f2)
    if k1 != k2 or k1 == "unknown":
        print("两个文件的标注类型不一致（%s vs %s），无法比对" % (k1, k2))
        return 1

    key = "event_id" if k1 == "event" else "frame_id"
    m1 = {r[key]: (r.get("truth_label") or "").strip() for r in r1 if r.get(key)}
    m2 = {r[key]: (r.get("truth_label") or "").strip() for r in r2 if r.get(key)}

    common = sorted(set(m1) & set(m2), key=lambda x: str(x))
    only1 = sorted(set(m1) - set(m2), key=lambda x: str(x))
    only2 = sorted(set(m2) - set(m1), key=lambda x: str(x))

    print("一致性比对")
    print("  A：%s（%d 条）" % (os.path.basename(args.labels[0]), len(m1)))
    print("  B：%s（%d 条）" % (os.path.basename(args.labels[1]), len(m2)))
    print("  共同条目：%d" % len(common))
    if only1:
        print("  仅 A 有：%d 条  %s" % (len(only1), ", ".join(map(str, only1[:8]))))
    if only2:
        print("  仅 B 有：%d 条  %s" % (len(only2), ", ".join(map(str, only2[:8]))))
    print("")

    if not common:
        print("没有可比对的共同条目。")
        return 1

    pairs = [(m1[k], m2[k]) for k in common]
    kappa, po, pe, n = cohen_kappa(pairs)

    print("  一致率（observed agreement）：%.3f" % po)
    print("  偶然一致率（expected）：    %.3f" % pe)
    print("  Cohen's Kappa：             %.3f" % kappa)
    print("")
    if kappa >= 0.75:
        print("  判定：一致性良好。标注标准清晰，真值可信，可直接用于评估。")
    elif kappa >= 0.60:
        print("  判定：一致性中等。建议复核分歧集中的类别，明确标准后再扩大标注量。")
    else:
        print("  判定：一致性偏低。优先检查 schema 是否定义不清，而不是催标注者。")

    diffs = [(k, m1[k], m2[k]) for k in common if m1[k] != m2[k]]
    if diffs:
        print("")
        print("  分歧清单（%d 处）：" % len(diffs))
        print("    %-12s %-16s %-16s" % (key, "A 判定", "B 判定"))
        print("    " + "-" * 46)
        for k, a, b in diffs[:25]:
            print("    %-12s %-16s %-16s" % (k, a, b))
        if len(diffs) > 25:
            print("    ... 还有 %d 处" % (len(diffs) - 25))
        print("")
        print("  下一步：逐条讨论 → 结论写入 <pack_id>.labels.adjudicated.csv")
        print("         并在 note 里写明仲裁依据。")

        # 分歧集中在哪些类别
        pair_counter = Counter((a, b) for _, a, b in diffs)
        print("")
        print("  分歧最集中的组合：")
        for (a, b), c in pair_counter.most_common(5):
            print("    %s ↔ %s：%d 次" % (a, b, c))
        print("")
        print("  → 这几组是 schema 最需要补充判定依据的地方。")
    return 0


# ---------------------------------------------------------------- guard

def cmd_guard():
    """提交前防泄漏检查：确保没有采集包或标注数据被误加入 Git 索引。

    这是规范 9.1 节的自动化版本。建议配成 pre-commit hook 或提交前手动跑。
    """
    import re
    import subprocess

    def git(*a):
        r = subprocess.run(["git"] + list(a), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "")

    rc, out = git("rev-parse", "--is-inside-work-tree")
    if rc != 0 or "true" not in out:
        print("当前目录不是 Git 仓库，跳过检查")
        return 0

    rc, files = git("ls-files")
    tracked = [f for f in files.split("\n") if f.strip()]

    problems = []

    # 1) 采集包与压缩包（zip 一律不该入库，除非在显式放行的目录）
    allow_dirs = ("examples/",)
    for f in tracked:
        if f.lower().endswith(".zip") and not f.startswith(allow_dirs):
            problems.append(("采集包/压缩包入库", f))

    # 2) 标注数据：只允许 schema 与 MANIFEST.csv
    allowed_labels = {"labels/schema.md", "labels/MANIFEST.csv",
                      "labels/README.md"}
    for f in tracked:
        if f.startswith("labels/") and f not in allowed_labels:
            problems.append(("标注数据入库", f))
        if f.startswith("golden/"):
            problems.append(("黄金集入库", f))

    # 3) 运行产物
    for f in tracked:
        if re.match(r"^(out|vis|fused|out_selftest|out_fused_selftest)/", f):
            problems.append(("运行产物入库", f))

    # 4) 密钥泄漏：扫描已入库的文本文件内容
    secret_pat = re.compile(
        r"(?i)(sk-[A-Za-z0-9]{16,}|AKID[A-Za-z0-9]{16,}|"
        r"(?:api[_-]?key|secret|token|passwd|password)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,})")
    text_ext = (".py", ".js", ".json", ".md", ".txt", ".html", ".yml", ".yaml",
                ".toml", ".ini", ".cfg", ".env", ".xml", ".csv")
    for f in tracked:
        if not f.lower().endswith(text_ext):
            continue
        if not os.path.exists(f):
            continue
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for i, ln in enumerate(fh, 1):
                    if len(ln) > 500:
                        continue
                    if secret_pat.search(ln):
                        problems.append(("疑似密钥（%s:%d）" % (f, i), f))
                        break
        except OSError:
            continue

    print("提交前防泄漏检查（%d 个已入库文件）" % len(tracked))
    print("")
    if not problems:
        print("通过：未发现采集包、标注数据、运行产物或疑似密钥。")
        print("")
        print("可以安全提交。")
        return 0

    print("发现 %d 项问题：" % len(problems))
    print("")
    seen = set()
    for kind, f in problems:
        key = (kind.split("（")[0], f)
        if key in seen:
            continue
        seen.add(key)
        print("  [%s] %s" % (kind, f))
    print("")
    print("处理建议：")
    print("  · 采集包与标注数据 → 移出仓库，走私有共享盘（见规范第 5 节）")
    print("  · 运行产物 → 加入 .gitignore，它们可由脚本重造")
    print("  · 密钥 → 立即吊销并改用本地 config.py（不入库）")
    print("")
    print("若确认某个文件确实该入库，请把它加入本脚本的 allow 列表，")
    print("不要直接删掉检查。")
    return 1


def main():
    ap = argparse.ArgumentParser(description="标注合并、校验与提交前防泄漏检查")
    ap.add_argument("--mode", choices=["validate", "merge", "kappa", "guard"], required=True)
    ap.add_argument("--labels", nargs="*", help="标注文件（guard 模式不需要）")
    ap.add_argument("--out", help="merge 模式的输出路径")
    args = ap.parse_args()

    if args.mode == "guard":
        return cmd_guard()
    if not args.labels:
        ap.error("除 guard 外，其他模式都需要 --labels")
    if args.mode == "merge":
        if not args.out:
            ap.error("merge 模式必须指定 --out")
        return cmd_merge(args)
    if args.mode == "kappa":
        return cmd_kappa(args)
    return cmd_validate(args)


if __name__ == "__main__":
    sys.exit(main())
