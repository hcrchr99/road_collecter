#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成一个与采集端导出格式完全一致的演示 ZIP，用于端到端验证整条链路：

    make_demo_zip.py  ->  demo_pack.zip
      -> analyze.py --input samples.csv          (IMU 事件检测)
      -> vision.py  --zip demo_pack.zip          (视觉判定 + 扫描帧发现)
      -> fusion.py  --imu --vision               (双模态融合)

演示场景刻意覆盖了全部融合分支：
  · 坑洼 且视觉也看到坑洼        -> R7 双模态互证（最高置信）
  · 井盖/减速带 但视觉见平路     -> R5/R6 结构性凸起
  · 扫描帧发现裂缝，附近无 IMU 事件 -> R1 视觉独有（IMU 原理上抓不到）
  · 扫描帧看到的坑洼与某事件重合  -> R15 空间去重（避免重复计数）
"""

import io
import json
import os
import zipfile

import cv2
import numpy as np
import pandas as pd

from analyze import synth
from vision import synth_road

DUR = 120.0
SPACING = 12.0
SPEED = 6.0
SWEEP_S = 2.0


def main():
    raw, truth = synth(seed=5, dur=DUR, spacing=SPACING, speed=SPEED)
    n = len(raw)
    fs = 100.0

    lat = raw["lat"].to_numpy()
    lon = raw["lon"].to_numpy()

    def at(tt):
        i = int(min(max(tt * fs, 0), n - 1))
        return float(lat[i]), float(lon[i])

    # ---------------- samples.csv（采集端列格式）
    rng = np.random.default_rng(3)
    samples = pd.DataFrame({
        "t_ms": raw["t"].round(0).astype(int),
        "ax": raw["ax"].round(4), "ay": raw["ay"].round(4), "az": raw["az"].round(4),
        "gx": (0.5 * rng.standard_normal(n)).round(4),
        "gy": (0.5 * rng.standard_normal(n)).round(4),
        "gz": (0.5 * rng.standard_normal(n)).round(4),
        "lat": raw["lat"].round(7), "lon": raw["lon"].round(7),
        "speed_ms": raw["speed"].round(3),
        "hp": np.abs(raw["az"] - 9.81).round(4),
        "vz": (raw["az"] - 9.81).round(4),
    })

    # ---------------- events.csv + 事件帧
    ev_rows, ev_frames = [], []
    for i, (tt, kind) in enumerate(truth):
        la, lo = at(tt)
        ev_rows.append({
            "id": i, "t_ms": int(tt * 1000), "duration_ms": int(300 if kind != "减速带" else 480),
            "peak": round(float(rng.uniform(9, 12) if kind != "减速带" else rng.uniform(5, 7)), 2),
            "rms": round(float(rng.uniform(1.5, 3.0)), 2),
            "lat": la, "lon": lo, "speed_kmh": SPEED * 3.6,
            "auto_label": kind, "final_label": kind, "frames": 4,
        })
        # 坑洼事件：画面里真的有坑洼；井盖/减速带：画面是完好路面
        img_kind = "坑洼" if kind == "坑洼" else "平路"
        for k, off in enumerate([-600, -200, 100, 300]):
            img = synth_road(img_kind, seed=(i * 10 + k) * 7 + 1)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            ev_frames.append(("frames/ev%d_t%d.jpg" % (i, int(tt * 1000) + off), buf.tobytes()))

    # ---------------- sweeps.csv + 扫描帧
    # 注意：扫描帧只在 SWEEP_S 的整数倍时刻产生（2,4,6,...），
    # 所以用例时间必须是偶数秒，否则条件永远匹配不上（这个坑踩过一次）。
    ev_times = [t for t, _ in truth]
    crack_at = [10.0, 22.0, 46.0, 70.0]   # 距最近事件 5 s ≈ 30 m，超出 20 m 去重半径
    dedup_t = 4.0                          # 距事件 #0 (t=3s, 6 m/s) 仅 6 m，应被去重
    assert all(abs(c % SWEEP_S) < 1e-9 for c in crack_at), "裂缝用例时间必须落在扫描节拍上"
    assert abs(dedup_t % SWEEP_S) < 1e-9, "去重用例时间必须落在扫描节拍上"

    sw_rows, sw_frames = [], []
    t = SWEEP_S
    while t < DUR - 1:
        la, lo = at(t)
        if abs(t - dedup_t) < 1e-6:
            kind, tag = "坑洼", "dedup_case"
        elif any(abs(t - c) < 1e-6 for c in crack_at):
            kind, tag = "裂缝", "vision_only"
        else:
            kind, tag = "平路", "normal"
        sw_rows.append({"t_ms": int(t * 1000), "lat": la, "lon": lo,
                        "speed_kmh": SPEED * 3.6,
                        "frame": "frames/sweep_t%d.jpg" % int(t * 1000)})
        img = synth_road(kind, seed=int(t * 13) + 101)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        sw_frames.append(("frames/sweep_t%d.jpg" % int(t * 1000), buf.tobytes()))
        t += SWEEP_S

    meta = {
        "version": "0.2", "demo": True,
        "sample_count": len(samples), "event_count": len(ev_rows),
        "sweep_count": len(sw_rows), "sweep_interval_s": SWEEP_S,
        "duration_s": DUR, "sample_rate_hz": fs,
        "scene": {
            "crack_sweeps_s": crack_at,
            "dedup_sweep_s": dedup_t,
            "note": "裂缝扫描帧位于相邻事件中点，距最近事件约 36 m；去重用例距事件 #0 约 6 m",
        },
    }

    out = "demo_pack.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("samples.csv", samples.to_csv(index=False))
        z.writestr("events.csv", pd.DataFrame(ev_rows).to_csv(index=False))
        z.writestr("sweeps.csv", pd.DataFrame(sw_rows).to_csv(index=False))
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for name, data in ev_frames + sw_frames:
            z.writestr(name, data)

    print("已生成 %s" % out)
    print("  样本 %d 条、IMU 事件 %d 个（事件帧 %d 张）、扫描帧 %d 张（每 %.0fs 一张）"
          % (len(samples), len(ev_rows), len(ev_frames), len(sw_frames), SWEEP_S))
    print("  场景：裂缝扫描帧于 %s（视觉独有）；坑洼扫描帧于 %.0fs（应被空间去重）"
          % (", ".join("%.0fs" % c for c in crack_at), dedup_t))
    print("  事件类型：%s" % dict(pd.Series([k for _, k in truth]).value_counts()))

    # 顺便把 samples.csv 单独落盘，供 analyze.py 直接读取
    samples.to_csv("demo_samples.csv", index=False)


if __name__ == "__main__":
    main()
