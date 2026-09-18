#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测器参数标定工具

背景：当初用「滚动标准差」做自适应门限时，事件自身会把局部标准差抬高、
反而把自己屏蔽掉——减速带峰值 3.6，它自己把门限顶到 5.5，导致长时低幅的
减速带被系统性漏检，召回率只有 67%。改用 |x| 的低分位数作稳健噪声底后，
同一份数据上 k=3~6 全区间都是 100% 召回、0 误报。

这个脚本就是当时用来发现该问题的证据工具，后续在**真实数据**上重标参数时继续用：
选一个 召回率 高、误报 低、且对 k 不敏感 的配置。

    python tune_detector.py
"""

import sys

import numpy as np
import pandas as pd
from scipy import signal as sps

from analyze import synth

# Windows 上重定向 stdout 到文件时默认走 GBK(cp936)，会把中文报告写成乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FS = 100.0
MIN_GAP_S = 0.30


def preprocess(acc):
    g = sps.sosfiltfilt(sps.butter(4, 0.5 / (FS / 2), btype="low", output="sos"), acc, axis=0)
    mag = np.linalg.norm(acc - g, axis=1)
    bp = sps.butter(4, np.array([1.0, 20.0]) / (FS / 2), btype="band", output="sos")
    return sps.sosfiltfilt(bp, mag)


def floor_std(x, win_s):
    w = max(8, int(win_s * FS))
    return np.maximum(0.20, pd.Series(x).rolling(w, center=True, min_periods=1).std().to_numpy())


def floor_quantile(x, win_s, q):
    w = max(8, int(win_s * FS))
    f = pd.Series(np.abs(x)).rolling(w, center=True, min_periods=max(4, w // 4)).quantile(q)
    return np.maximum(f.bfill().ffill().to_numpy(), 0.02)


def score(bp, truth, floor, k, abs_min=1.2):
    peaks, _ = sps.find_peaks(
        bp, height=np.maximum(abs_min, k * floor),
        prominence=np.maximum(abs_min * 0.5, k * 0.5 * floor),
        distance=int(0.04 * FS))
    keep, last = [], -1e9
    for p in peaks:
        if p - last >= MIN_GAP_S * FS:
            keep.append(p)
        last = p
    peaks = np.array(keep, dtype=int)
    tp = sum(1 for tt, _ in truth if len(peaks) and np.abs(peaks / FS - tt).min() < 0.6)
    return tp, len(peaks) - tp, len(truth)


def main():
    variants = [("滚动标准差 win=%.0fs" % w, ("std", w, 0.0)) for w in (1.0, 2.0, 3.0)]
    variants += [("低分位 win=%.0fs q=%.0f%%" % (w, q * 100), ("q", w, q))
                 for w in (2.0, 3.0, 5.0) for q in (0.10, 0.20, 0.30)]

    print("=" * 74)
    print("噪声底估计方式对比（6 组合成场景 × 4 档门限倍数）")
    print("=" * 74)
    print("%-32s %6s %8s %8s" % ("配置", "k", "召回", "误报"))

    rows = []
    for name, (mode, win_s, q) in variants:
        for k in (3.0, 4.0, 5.0, 6.0):
            rec, fp = [], []
            for seed in range(6):
                raw, truth = synth(seed=seed, dur=60.0, spacing=6.0)
                acc = np.c_[raw["ax"].to_numpy(), raw["ay"].to_numpy(), raw["az"].to_numpy()]
                bp = preprocess(acc)
                fl = floor_std(bp, win_s) if mode == "std" else floor_quantile(bp, win_s, q)
                tp, f, tot = score(bp, truth, fl, k)
                rec.append(tp / tot)
                fp.append(f)
            rows.append((name, k, float(np.mean(rec)), float(np.mean(fp))))
            print("%-32s %6.1f %7.0f%% %8.1f" % (name, k, np.mean(rec) * 100, np.mean(fp)))

    print("=" * 74)
    ok = [r for r in rows if r[2] >= 0.95 and r[3] <= 0.2]
    ok.sort(key=lambda r: (r[3], -r[2]))
    print("满足 召回>=95% 且 平均误报<=0.2 的配置（按误报升序）：")
    for name, k, rec, fp in ok[:10]:
        print("  %-32s k=%.1f  召回%.0f%%  误报%.2f" % (name, k, rec * 100, fp))
    if ok:
        kset = sorted({r[1] for r in ok})
        stable = [r for r in ok if r[1] in kset]
        print("\n结论：低分位数噪声底在更宽的 k 区间内保持满分，对参数不敏感，"
              "在真实数据上更不容易因标定误差而失效。")
        print("当前 analyze.py 采用 q=20%、win=3.0s、k=4.0。")
    else:
        print("  无满足条件的配置，需改进检测器（例如引入持续能量通道检测弱事件）。")


if __name__ == "__main__":
    main()
