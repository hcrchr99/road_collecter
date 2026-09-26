# -*- coding: utf-8 -*-
"""离线特征筛选（一次性分析脚本）：原始 IMU 波形里还有没有可喂给模型的判别信息。

对黄金集每个事件从 samples.csv 原始数据提 10 个候选特征，
算「真值病害(24) vs 非病害」的 AUC。AUC≈0.5 = 无判别力。
只筛特征不做模型改动——结果决定是否值得改 build_feature_block。
"""
import sys
sys.path.insert(0, ".")

import json

import numpy as np

from llm_pipeline import config, evaluate, unpack

RECS = {r["event_id"]: r for r in
        (json.loads(l) for l in
         open(config.OUTPUT_DIR / "l0" / "features.jsonl", encoding="utf-8")
         if l.strip())}


def features_for_event(t, hp, vz, spd, t0, t1, feat):
    """10 个候选特征（原始波形派生，未喂过模型）。"""
    i0 = int(np.searchsorted(t, t0 - 150, side="left"))
    i1 = int(np.searchsorted(t, t1 + 900, side="right"))
    win_t = t[i0:i1]
    win_hp = hp[i0:i1]
    win_vz = vz[i0:i1]
    if len(win_hp) < 10:
        return None
    peak_i = int(np.argmax(np.abs(win_hp)))
    peak = abs(win_hp[peak_i])
    t_peak = win_t[peak_i]
    # 1) 上升时间：起点 -> 峰值
    rise_ms = float(t_peak - win_t[0])
    # 2) 衰减时间：峰值 -> 持续低于 30% 峰值
    below = np.abs(win_hp[peak_i:]) < 0.3 * peak
    decay_ms = None
    for k in range(len(below) - 3):
        if below[k] and below[k + 1] and below[k + 2]:
            decay_ms = float(win_t[peak_i + k] - t_peak)
            break
    # 3) 振铃数：峰值后高于 30% 峰值的局部极大个数
    after = np.abs(win_hp[peak_i:])
    loc_max = sum(1 for k in range(1, len(after) - 1)
                  if after[k] > 0.3 * peak and after[k] >= after[k - 1]
                  and after[k] >= after[k + 1])
    # 4) 过零率（高频代理，次/100ms）
    zc = int(np.sum(np.diff(np.sign(win_hp)) != 0))
    dur_s = max((win_t[-1] - win_t[0]) / 1000.0, 1e-6)
    zc_rate = zc / dur_s * 0.1
    # 5/6) 频谱：60Hz 采样 → Nyquist 30Hz，分 0-10 / 10-30 Hz 两带
    seg = win_hp - win_hp.mean()
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / 60.0)
    e_low = float(np.sum(spec[freqs < 10] ** 2))
    e_high = float(np.sum(spec[(freqs >= 10) & (freqs <= 30)] ** 2))
    hf_ratio = e_high / max(e_low + e_high, 1e-9)
    centroid = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-9))
    # 7) 峰度（冲击尖锐度）
    sd = float(np.std(seg)) or 1e-9
    kurt = float(np.mean((seg / sd) ** 4))
    # 8) 空间长度 = 车速 × 时长（米）
    dur_ms = float(t1 - t0)
    spatial_m = float(spd) * dur_ms / 1000.0
    # 9) vz 冲量（带符号积分 / |hp| 峰）
    vz_imp = float(np.trapezoid(win_vz, win_t) / 1000.0) / max(peak, 1e-9)
    # 10) 双峰性（前后轴/两次触地）：峰后 200ms 内是否有 >60% 峰值的第二峰
    second_peak = 0.0
    for k in range(peak_i + 1, len(after)):
        if win_t[peak_i + k] - t_peak > 250:
            break
        if after[k] > 0.6 * peak:
            second_peak = float(win_t[peak_i + k] - t_peak)
            break
    return {
        "rise_ms": rise_ms, "decay_ms": decay_ms, "ringdown": float(loc_max),
        "zc_rate": zc_rate, "hf_ratio": hf_ratio, "centroid": centroid,
        "kurtosis": kurt, "spatial_m": spatial_m, "vz_impulse": vz_imp,
        "second_peak_ms": second_peak,
    }


def auc(pos, neg):
    """秩和 AUC。"""
    if not pos or not neg:
        return None
    vals = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    vals.sort(key=lambda x: x[0])
    ranks = {}
    n = len(vals)
    i = 0
    rank_sum_pos = 0.0
    # 平均秩处理并列
    while i < n:
        j = i
        while j < n and vals[j][0] == vals[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if vals[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    gold = evaluate.load_gold()
    with unpack.Pack(config.DEFAULT_PACK) as pack:
        samples = pack.read_samples()
        t = np.array([int(float(r["t_ms"])) for r in samples], dtype=np.int64)
        hp = np.array([float(r["hp"]) for r in samples], dtype=float)
        vz = np.array([float(r["vz"]) for r in samples], dtype=float)
        spd = np.array([float(r.get("speed_ms") or 0) for r in samples], dtype=float)
        rows = {}
        for eid, rec in RECS.items():
            if rec["blocked"]:
                continue
            f = rec["imu_features"]
            t0 = f["t0_ms"]
            i_at = int(np.searchsorted(t, t0))
            spd_at = float(spd[i_at]) if i_at < len(spd) else 0.0
            r = features_for_event(t, hp, vz, spd_at, t0,
                                   t0 + f["duration_ms"], f)
            if r:
                rows[eid] = r
    truth = {eid: evaluate.binary_of(g["truth_label"]) for eid, g in gold.items()}
    pos = [eid for eid in rows if truth.get(eid) == "病害"]
    neg = [eid for eid in rows if truth.get(eid) == "非病害"]
    print("样本：病害 %d / 非病害 %d（blocked 除外）" % (len(pos), len(neg)))
    print()
    print("%-16s %8s %8s %8s %8s   说明" % ("特征", "AUC", "病害中位", "非病中位", "1-|AUC|"))
    for k in ["rise_ms", "decay_ms", "ringdown", "zc_rate", "hf_ratio", "centroid",
              "kurtosis", "spatial_m", "vz_impulse", "second_peak_ms"]:
        pv = [rows[e][k] for e in pos if rows[e][k] is not None]
        nv = [rows[e][k] for e in neg if rows[e][k] is not None]
        a = auc(pv, nv)
        if a is None:
            continue
        a2 = max(a, 1 - a)   # 方向未知，取优
        print("%-16s %8.3f %8.2f %8.2f %8.3f   %s" % (
            k, a, float(np.median(pv)) if pv else 0, float(np.median(nv)) if nv else 0,
            1 - abs(a2 - 0.5) * 2,
            "★有信号" if a2 >= 0.65 else "弱" if a2 >= 0.58 else "无"))


if __name__ == "__main__":
    main()
