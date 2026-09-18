#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集包登记工具 —— 团队数据共享的枢纽

解决的问题：
    采集包因为含 GPS 轨迹不进 Git，于是「团队现在有哪些数据、谁采的、
    标到哪一步」就没有地方记录。本工具用一份不含任何轨迹坐标的
    MANIFEST.csv 把这个信息补上，而 MANIFEST.csv 本身可以安全入库。

用法：
    # 登记一份新采集包（自动读 zip 计算 sha256 与基本统计）
    python tools/register_pack.py ../roadcheck_20260918_2127.zip \\
           --collected-by alice --mode bike --storage tdrive

    # 更新已有记录的状态
    python tools/register_pack.py --update roadcheck_20260918_2127 \\
           --label-status done --labeled-by bob

    # 列出待标注的包
    python tools/register_pack.py --list --status todo

依赖：无（标准库即可）
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone

# Windows 上重定向 stdout 到文件时默认走 GBK，中文会乱码。统一为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, os.pardir, "labels", "MANIFEST.csv")

FIELDS = [
    "pack_id", "collected_by", "collected_at", "mode",
    "duration_s", "samples", "fs_hz", "distance_m",
    "event_count", "sweep_count", "sha256",
    "storage", "label_status", "labeled_by", "notes",
]

VALID_LABEL_STATUS = ("todo", "wip", "done", "frozen")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_manifest(rows, path):
    """原子写：先写临时文件再替换，避免并发读到半截文件。"""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".MANIFEST.csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, path)          # 原子操作


def probe_zip(path):
    """从采集包里读出可公开的聚合统计，不碰任何坐标。

    注意：只读 samples.csv 的 t_ms 列与 events/sweeps 的行数，
    不解析 lat/lon，因此结果不含任何位置信息，可以安全写进入库的 MANIFEST.csv。
    """
    info = {"samples": "", "duration_s": "", "fs_hz": "",
            "event_count": "", "sweep_count": "", "distance_m": ""}
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if "samples.csv" in names:
                rows = z.read("samples.csv").decode("utf-8", "replace").splitlines()
                n = max(0, len(rows) - 1)
                info["samples"] = n
                if n > 1:
                    hdr = rows[0].split(",")
                    if "t_ms" in hdr:
                        ci = hdr.index("t_ms")
                        ts = []
                        for r in rows[1:]:
                            try:
                                ts.append(float(r.split(",")[ci]))
                            except (ValueError, IndexError):
                                continue
                        if len(ts) > 1:
                            ts.sort()
                            info["duration_s"] = round((ts[-1] - ts[0]) / 1000.0, 1)
                            # 用中位数间隔估采样率：对丢包与时间抖动稳健。
                            # 与 replay_detect.py 的算法保持一致，避免两处数字对不上。
                            gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
                            if gaps:
                                gaps.sort()
                                med = gaps[len(gaps) // 2]
                                if med > 0:
                                    info["fs_hz"] = round(1000.0 / med, 1)
            if "events.csv" in names:
                info["event_count"] = max(
                    0, len(z.read("events.csv").decode("utf-8", "replace").splitlines()) - 1)
            if "sweeps.csv" in names:
                info["sweep_count"] = max(
                    0, len(z.read("sweeps.csv").decode("utf-8", "replace").splitlines()) - 1)
    except (zipfile.BadZipFile, KeyError):
        pass
    return info


def cmd_register(args):
    path = args.zip
    if not os.path.exists(path):
        print("找不到采集包：%s" % path)
        return 1
    pack_id = args.pack_id or os.path.splitext(os.path.basename(path))[0]

    rows = read_manifest(args.manifest)
    if any(r["pack_id"] == pack_id for r in rows):
        print("已存在同名记录：%s" % pack_id)
        print("如需修改请加 --update %s" % pack_id)
        return 1

    info = probe_zip(path)
    row = {
        "pack_id": pack_id,
        "collected_by": args.collected_by or "",
        "collected_at": args.collected_at or datetime.now(timezone.utc)
                        .astimezone().replace(microsecond=0).isoformat(),
        "mode": args.mode or "",
        "duration_s": info["duration_s"],
        "samples": info["samples"],
        "fs_hz": info["fs_hz"],
        "distance_m": info["distance_m"],
        "event_count": info["event_count"],
        "sweep_count": info["sweep_count"],
        "sha256": sha256_of(path)[:16],
        "storage": args.storage or "local",
        "label_status": "todo",
        "labeled_by": "",
        "notes": args.notes or "",
    }
    rows.append(row)
    write_manifest(rows, args.manifest)

    print("已登记：%s" % pack_id)
    print("  采集人 %s ｜ 模式 %s ｜ 时长 %ss ｜ 样本 %s ｜ 采样率 %sHz"
          % (row["collected_by"], row["mode"], row["duration_s"],
             row["samples"], row["fs_hz"]))
    print("  事件 %s 个 ｜ 扫描帧 %s 张 ｜ sha256 %s"
          % (row["event_count"], row["sweep_count"], row["sha256"]))
    print("  标注状态：todo")
    return 0


def cmd_update(args):
    rows = read_manifest(args.manifest)
    hit = None
    for r in rows:
        if r["pack_id"] == args.update:
            hit = r
            break
    if hit is None:
        print("找不到记录：%s" % args.update)
        return 1

    changed = []
    if args.label_status:
        if args.label_status not in VALID_LABEL_STATUS:
            print("label-status 只能是 %s" % " / ".join(VALID_LABEL_STATUS))
            return 1
        hit["label_status"] = args.label_status
        changed.append("label_status=%s" % args.label_status)
    if args.labeled_by:
        hit["labeled_by"] = args.labeled_by
        changed.append("labeled_by=%s" % args.labeled_by)
    if args.storage:
        hit["storage"] = args.storage
        changed.append("storage=%s" % args.storage)
    if args.notes is not None:
        hit["notes"] = args.notes
        changed.append("notes=%s" % args.notes)

    if not changed:
        print("没有指定要更新的字段")
        return 1

    write_manifest(rows, args.manifest)
    print("已更新 %s：%s" % (args.update, "，".join(changed)))
    return 0


def cmd_list(args):
    rows = read_manifest(args.manifest)
    if args.status:
        rows = [r for r in rows if r["label_status"] == args.status]
    if not rows:
        print("没有符合条件的记录")
        return 0

    print("%-32s %-8s %-8s %-6s %-6s %-6s %s"
          % ("pack_id", "采集人", "模式", "时长", "事件", "状态", "标注人"))
    print("-" * 92)
    for r in rows:
        print("%-32s %-8s %-8s %-6s %-6s %-6s %s"
              % (r["pack_id"][:32], r.get("collected_by", ""), r.get("mode", ""),
                 r.get("duration_s", ""), r.get("event_count", ""),
                 r.get("label_status", ""), r.get("labeled_by", "")))
    print("")
    print("共 %d 份" % len(rows))
    return 0


def main():
    ap = argparse.ArgumentParser(description="采集包登记与状态跟踪")
    ap.add_argument("zip", nargs="?", help="采集包路径（登记新包时用）")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="MANIFEST.csv 路径（默认 labels/MANIFEST.csv）")
    ap.add_argument("--pack-id", help="自定义 pack_id（默认取文件名）")
    ap.add_argument("--collected-by", help="采集人")
    ap.add_argument("--collected-at", help="采集时间 ISO8601（默认当前时间）")
    ap.add_argument("--mode", choices=["bike", "vehicle", "handheld"], help="采集模式")
    ap.add_argument("--storage", help="存储位置，如 tdrive / nas / local:alice")
    ap.add_argument("--notes", default=None, help="备注")

    ap.add_argument("--update", metavar="PACK_ID", help="更新已有记录")
    ap.add_argument("--label-status", choices=list(VALID_LABEL_STATUS), help="标注状态")
    ap.add_argument("--labeled-by", help="标注人")

    ap.add_argument("--list", action="store_true", help="列出记录")
    ap.add_argument("--status", help="配合 --list 过滤状态")

    args = ap.parse_args()

    if args.list:
        return cmd_list(args)
    if args.update:
        return cmd_update(args)
    if args.zip:
        return cmd_register(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
