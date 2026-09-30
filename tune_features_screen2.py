# -*- coding: utf-8 -*-
"""第二轮筛选：绕过 hp 管线，直接用原始 6 轴 + 路段级上下文。

用户假设：检测器只用事件「时刻」，事件区间附近的**原始波形**（未被 hp=|mag|−EMA
压缩掉方向信息）可能还有判别信息。测三类：
  A. 路段上下文：事件周围 ±5s / ±15s 的 hp RMS（路段纹理），事件能量对比度
  B. 原始 6 轴：ax/ay/az/gx/gy/gz 在事件窗的峰峰值比（相对背景）——
     横向晃动/旋转冲击是「轮子掉进坑」的候选签名，hp 管线把它丢了
  C. 垂直低频：az 的低通残差（车轮跌落的准静态倾斜，~1Hz 量级，hp 的高通保不住）
"""
import json
import sys

sys.path.insert(0, ".")

import numpy as np

from llm_pipeline import config, evaluate, unpack

RECS = {r["event_id"]: r for r in
        (json.loads(l) for l in
         open(config.OUTPUT_DIR / "l0" / "features.jsonl", encoding="utf-8") if l.strip())}


def p2p(x):
    return float(x.max() - x.min()) if len(x) else 0.0


def main():
    gold = evaluate.load_gold()
    with unpack.Pack(config.DEFAULT_PACK) as pack:
        samples = pack.read_samples()
        cols = {}
        for k in ("t_ms", "ax", "ay", "az", "gx", "gy", "gz", "hp", "vz"):
            cols[k] = np.array([float(r[k]) for r in samples], dtype=float)
    t = cols["t_ms"]

    def sl(t0, t1):
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="right"))
        return i0, i1

    rows = {}
    for eid, rec in RECS.items():
        if rec["blocked"]:
            continue
        t0 = rec["imu_features"]["t0_ms"]
        t1 = t0 + rec["imu_features"]["duration_ms"]
        i0, i1 = sl(t0 - 150, t1 + 400)
        if i1 - i0 < 8:
            continue
        # 背景窗（±5s，扣掉事件窗）
        b0, b1 = sl(t0 - 5000, t1 + 5000)
        ev = slice(i0, i1)
        bg_mask = np.ones(b1 - b0, dtype=bool)
        bg_mask[max(0, i0 - b0):max(0, i1 - b0)] = False
        bg = slice(b0, b0 + (b1 - b0))
        f = {}
        # A. 上下文能量与对比度
        bg_hp = cols["hp"][bg][bg_mask]
        ctx5 = float(np.sqrt(np.mean(bg_hp ** 2))) if bg_mask.sum() else 0.0
        b015, b115 = sl(t0 - 15000, t1 + 15000)
        bg15_mask = np.ones(b115 - b015, dtype=bool)
        bg15_mask[max(0, i0 - b015):max(0, i1 - b015)] = False
        hp15 = cols["hp"][b015:b115][bg15_mask]
        ctx15 = float(np.sqrt(np.mean(hp15 ** 2))) if hp15.size else 0.0
        ev_rms = float(np.sqrt(np.mean(cols["hp"][ev] ** 2)))
        f["ctx_rms_5s"] = ctx5
        f["ctx_rms_15s"] = ctx15
        f["contrast_5s"] = ev_rms / max(ctx5, 1e-9)
        f["contrast_15s"] = ev_rms / max(ctx15, 1e-9)
        # B. 原始 6 轴：事件窗峰峰值 / 背景峰峰值（每轴自身归一，免校准）
        for ax_name in ("ax", "ay", "az", "gx", "gy", "gz"):
            sig_ev = cols[ax_name][ev]
            sig_bg = cols[ax_name][bg][bg_mask]
            ratio = p2p(sig_ev) / max(p2p(sig_bg), 1e-9)
            f["p2p_" + ax_name] = ratio
        # 横向最大（ax/ay 中较大的比值）+ 陀螺最大
        f["p2p_horiz_max"] = max(f["p2p_ax"], f["p2p_ay"])
        f["p2p_gyro_max"] = max(f["p2p_gx"], f["p2p_gy"], f["p2p_gz"])
        # C. az 低频残差（重力方向倾斜：az 的 0.5s 滑动均值在事件前后的漂移）
        i0b, i1b = sl(t0 - 1000, t1 + 1200)
        az_low = cols["az"][i0b:i1b]
        if len(az_low) > 20:
            k = max(len(az_low) // 20, 1)
            drift = float(np.max(np.abs(np.convolve(az_low, np.ones(k) / k, mode="valid")
                                        - np.median(az_low))))
        else:
            drift = 0.0
        f["az_drift"] = drift
        rows[eid] = f

    truth = {eid: evaluate.binary_of(g["truth_label"]) for eid, g in gold.items()}
    pos = [e for e in rows if truth.get(e) == "病害"]
    neg = [e for e in rows if truth.get(e) == "非病害"]
    print("样本：病害 %d / 非病害 %d" % (len(pos), len(neg)))
    print()

    def auc(pv, nv):
        vals = sorted([(v, 1) for v in pv] + [(v, 0) for v in nv],
                      key=lambda x: x[0])
        n = len(vals)
        rs = 0.0
        i = 0
        while i < n:
            j = i
            while j < n and vals[j][0] == vals[i][0]:
                j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                if vals[k][1] == 1:
                    rs += avg
            i = j
        np_, nn = len(pv), len(nv)
        return (rs - np_ * (np_ + 1) / 2.0) / (np_ * nn)

    print("%-16s %8s %9s %9s   判定" % ("特征", "AUC", "病害中位", "非病中位"))
    for k in rows[min(rows)]:
        pv = [rows[e][k] for e in pos]
        nv = [rows[e][k] for e in neg]
        a = auc(pv, nv)
        a2 = max(a, 1 - a)
        verdict = "★有信号" if a2 >= 0.65 else "弱" if a2 >= 0.58 else "无"
        print("%-16s %8.3f %9.2f %9.2f   %s" % (
            k, a, float(np.median(pv)), float(np.median(nv)), verdict))


if __name__ == "__main__":
    main()
