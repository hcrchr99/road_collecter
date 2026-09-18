# -*- coding: utf-8 -*-
"""检查采集包内容"""
import json
import os
import zipfile

import numpy as np
import pandas as pd

SRC = r"C:\Users\Admin\Downloads\roadcheck_20260918_2127.zip"

z = zipfile.ZipFile(SRC)
print("ZIP: %s  (%.1f KB)" % (os.path.basename(SRC), os.path.getsize(SRC) / 1024))
print("-" * 78)
for i in z.infolist():
    print("  %-34s %9d 字节" % (i.filename, i.file_size))

print("-" * 78)
names = z.namelist()

if "meta.json" in names:
    meta = json.loads(z.read("meta.json").decode("utf-8"))
    print("meta.json:")
    for k in ("version", "sample_count", "event_count", "sweep_count",
              "duration_s", "sample_rate_hz", "camera_enabled", "labelled_events"):
        if k in meta:
            print("   %-18s %s" % (k, meta[k]))
    if "method" in meta:
        print("   method:")
        for k, v in meta["method"].items():
            print("      %-20s %s" % (k, v))

if "samples.csv" in names:
    s = pd.read_csv(z.open("samples.csv"))
    print("-" * 78)
    print("samples.csv: %d 行 × %d 列" % (len(s), len(s.columns)))
    print("列: %s" % ", ".join(s.columns))
    print("-" * 78)
    t = s["t_ms"].to_numpy(dtype=float)
    dt = np.diff(t)
    print("时长            %.1f s" % (t[-1] / 1000.0))
    print("采样间隔中位数  %.2f ms  -> %.1f Hz" % (np.median(dt), 1000.0 / np.median(dt)))
    print("采样间隔最大    %.2f ms" % dt.max())
    print("时间戳单调      %s" % bool(np.all(dt > 0)))
    print("-" * 78)
    for c in ("ax", "ay", "az", "hp", "vz"):
        if c in s.columns:
            v = s[c].dropna().to_numpy(dtype=float)
            print("%-4s min %9.3f  max %9.3f  mean %7.3f  std %7.3f"
                  % (c, v.min(), v.max(), v.mean(), v.std()))
    if "speed_ms" in s.columns:
        sp = s["speed_ms"].dropna().to_numpy(dtype=float)
        print("-" * 78)
        print("速度(m/s) min %.2f  中位 %.2f  max %.2f" % (sp.min(), np.median(sp), sp.max()))
        print("  对应 km/h: min %.1f  中位 %.1f  max %.1f"
              % (sp.min() * 3.6, np.median(sp) * 3.6, sp.max() * 3.6))
        print("  >= 8 km/h 的样本占比: %.1f%%" % (100.0 * np.mean(sp >= 8 / 3.6)))
        print("  > 0      的样本占比: %.1f%%" % (100.0 * np.mean(sp > 0)))
    if "lat" in s.columns:
        la = s["lat"].dropna()
        lo = s["lon"].dropna()
        print("-" * 78)
        print("定位点数 %d；lat %s ~ %s；lon %s ~ %s"
              % (len(la), ("%.6f" % la.min()) if len(la) else "-",
                 ("%.6f" % la.max()) if len(la) else "-",
                 ("%.6f" % lo.min()) if len(lo) else "-",
                 ("%.6f" % lo.max()) if len(lo) else "-"))
        if len(la):
            kx = 111320 * np.cos(np.radians(la.mean()))
            dx = (lo.max() - lo.min()) * kx
            dy = (la.max() - la.min()) * 110540
            print("轨迹包围盒约 %.0f m × %.0f m" % (dx, dy))

if "events.csv" in names:
    e = pd.read_csv(z.open("events.csv"))
    print("-" * 78)
    print("events.csv: %d 条" % len(e))
    if len(e):
        print(e.to_string(index=False, max_colwidth=28))

if "sweeps.csv" in names:
    sw = pd.read_csv(z.open("sweeps.csv"))
    print("-" * 78)
    print("sweeps.csv: %d 条" % len(sw))
    if len(sw):
        print("  时间范围 %.1f ~ %.1f s"
              % (sw["t_ms"].min() / 1000.0, sw["t_ms"].max() / 1000.0))
        if "speed_kmh" in sw.columns:
            print("  速度中位 %.1f km/h" % sw["speed_kmh"].median())

print("-" * 78)
nf = [n for n in names if n.startswith("frames/")]
print("画面文件 %d 个" % len(nf))
for n in nf[:6]:
    print("   %s" % n)
if len(nf) > 6:
    print("   ...")
