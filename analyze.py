#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路面体检 / RoadCheck - 分析基线 v0.1.1

链路：读取 CSV -> 重采样 100Hz -> 重力分离 -> 带通滤波
      -> (A) 峰值检测出事件（无标签推理）  (B) 滑窗特征提取（有标签训练）
      -> 分类 -> 空间聚类 -> 导出事件表 / GeoJSON / 健康度指数

自检（无需任何数据，生成合成振动信号验证整条链路）：
    python analyze.py --selftest

分析自采数据：
    python analyze.py --input roadcheck_raw_20260918_1930.csv --outdir out

在公开数据集上训练分类器（需有逐行标签列）：
    python analyze.py --input dataset.csv --label-col "class - groundtruth" --train --outdir out

依赖：numpy scipy pandas scikit-learn
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import signal as sps

# ---------------------------------------------------------------- 配置
FS_TARGET = 100.0          # 统一重采样频率（文献常用；采集端约 60 Hz）
BANDPASS = (1.0, 20.0)     # 路面激励主频段 Hz
WIN_SEC = 0.64             # 特征窗口长度（减速带典型持续时间量级）
HOP_SEC = 0.32             # 训练集滑窗步长
PROM_ABS = 1.2             # 峰值最小绝对门限 m/s^2（防止静止时的微小噪声被当成事件）
NOISE_K = 4.0              # 相对稳健噪声底的门限倍数（真实数据上主要调这个）
NOISE_WIN_S = 3.0          # 稳健噪声底的估计窗口长度
NOISE_Q = 0.20             # 用 |x| 的 20% 分位数估噪声底（事件占空比远低于此，故不受事件污染）
REFRACTORY_S = 0.04        # 峰间距下限秒（find_peaks 的 distance 参数）
MIN_EVENT_GAP_S = 0.30     # 两个独立事件的最小间隔
CLUSTER_EPS_M = 15.0       # 空间聚类半径（米）
MAX_EVENTS_PER_KM = 60.0   # 事件密度上限，超过视为手抖/挂载松动，整段丢弃
RHI_K = 15.0               # 健康度指数标定常数：缺陷密度每增加 K，RHI 衰减到 1/e

# 经纬度换算常数（统一用「米/度」，避免与千米尺度混用）
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0

FEATURE_COLS = [
    "peak", "rms", "kurtosis", "skew", "duration", "impulse",
    "zero_cross", "energy_low", "energy_high", "pos_neg_ratio", "speed_ms",
]

CLASS_ORDER = ["平路", "坑洼", "减速带", "粗糙路面", "井盖", "其他"]


# ---------------------------------------------------------------- 数据读取
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    rename = {
        "t_ms": "t", "timestamp": "t", "time": "t", "timestep": "t",
        "accx": "ax", "accy": "ay", "accz": "az",
        "accelerometer x": "ax", "accelerometer y": "ay", "accelerometer z": "az",
        "gyroscope x": "gx", "gyroscope y": "gy", "gyroscope z": "gz",
        "latitude": "lat", "longitude": "lon",
        "speed_ms": "speed", "speed": "speed",
        "class - groundtruth": "label_raw", "label": "label_raw", "type": "type",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "t" not in df.columns:
        df["t"] = np.arange(len(df)) / FS_TARGET * 1000.0
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    if len(df) < 16:
        raise ValueError("有效数据点不足 16 个")

    if "ax" not in df.columns:
        for cand in ("mag", "magnitude", "acceleration", "hp"):
            if cand in df.columns:
                df["ax"] = pd.to_numeric(df[cand], errors="coerce")
                df["ay"] = 0.0
                df["az"] = 0.0
                break
        else:
            raise ValueError("CSV 中找不到加速度列（ax/ay/az 或 mag/acceleration/hp）")

    for c in ("ax", "ay", "az"):
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    for c in ("lat", "lon", "speed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def infer_source_fs(df):
    """用时间戳中位数间隔估计原始采样率，比 len/duration 稳健（可容忍丢帧）"""
    t = df["t"].to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return float(FS_TARGET)
    med = float(np.median(dt)) / 1000.0
    return 1.0 / med if med > 0 else float(FS_TARGET)


# ---------------------------------------------------------------- 重采样
def resample(df, fs=FS_TARGET):
    t = df["t"].to_numpy(dtype=float)
    if t[-1] <= t[0]:
        raise ValueError("时间戳无效（末值不大于首值）")
    t_s = (t - t[0]) / 1000.0
    n = int(np.floor(t_s[-1] * fs)) + 1
    tt = np.arange(n) / fs

    out = {"t": tt * 1000.0 + t[0]}
    for c in ("ax", "ay", "az"):
        out[c] = np.interp(tt, t_s, df[c].to_numpy(dtype=float))
    for c in ("lat", "lon", "speed"):
        if c in df.columns:
            out[c] = _step_hold(tt, t_s, df[c].to_numpy(dtype=float))
        else:
            out[c] = np.full(n, np.nan)
    d = pd.DataFrame(out)
    return d, float(fs)


def _step_hold(tt, t_s, v):
    """坐标与速度用阶梯保持（GPS 更新率远低于 IMU，线性插值会造出虚假轨迹）"""
    idx = np.clip(np.searchsorted(t_s, tt, side="right") - 1, 0, len(v) - 1)
    return v[idx]


# ---------------------------------------------------------------- 预处理
def preprocess(d, fs):
    """重力分离 + 动态加速度幅值 + 重力方向投影（带符号的垂直分量）"""
    sos = sps.butter(4, 0.5 / (fs / 2), btype="low", output="sos")
    acc = np.c_[d["ax"], d["ay"], d["az"]]
    g = sps.sosfiltfilt(sos, acc, axis=0)
    lin = acc - g
    d = d.copy()

    d["lin_mag"] = np.linalg.norm(lin, axis=1)

    gn = np.linalg.norm(g, axis=1)
    gn[gn < 1e-6] = 1e-6
    d["vz"] = np.sum(lin * g, axis=1) / gn      # 垂直分量，保留正负号

    bp = sps.butter(4, np.array(BANDPASS) / (fs / 2), btype="band", output="sos")
    d["bp_mag"] = sps.sosfiltfilt(bp, d["lin_mag"].to_numpy(dtype=float))
    d["bp_vz"] = sps.sosfiltfilt(bp, d["vz"].to_numpy(dtype=float))
    return d


# ---------------------------------------------------------------- 特征
def window_features(w, fs, speed_ms=np.nan):
    """w 为带符号的垂向带通信号窗口"""
    w = np.asarray(w, dtype=float)
    if len(w) < 8:
        return None
    w = w - np.mean(w)
    peak = float(np.max(np.abs(w)))
    if peak < 1e-9:
        return None
    rms = float(np.sqrt(np.mean(w ** 2)))
    sd = float(np.std(w)) or 1e-9
    z = (w - np.mean(w)) / sd

    active = np.abs(w) > 0.3 * peak
    duration = float(active.sum()) / fs
    impulse = float(np.sum(np.abs(w[active])) / fs) if active.any() else 0.0

    zc = int(np.sum(np.diff(np.signbit(w))))

    nper = int(min(len(w), 128))
    f, pxx = sps.welch(w, fs=fs, nperseg=nper)
    tot = float(np.sum(pxx)) or 1e-9
    band = (f >= 1.0) & (f < 8.0)
    hi = (f >= 8.0) & (f <= 20.0)

    pos = float(np.sum(w[w > 0]))
    neg = float(-np.sum(w[w < 0]))
    pn = pos / neg if neg > 1e-6 else 5.0

    return {
        "peak": round(peak, 3),
        "rms": round(rms, 3),
        "kurtosis": round(float(np.mean(z ** 4)), 3),
        "skew": round(float(np.mean(z ** 3)), 3),
        "duration": round(duration, 3),
        "impulse": round(impulse, 3),
        "zero_cross": zc,
        "energy_low": round(float(np.sum(pxx[band]) / tot), 4),
        "energy_high": round(float(np.sum(pxx[hi]) / tot), 4),
        "pos_neg_ratio": round(pn, 3),
    }


def extract_windows(d, fs, labels=None, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    """滑窗提取特征。labels 为逐行标签 Series 时同步输出窗口标签（多数投票）。"""
    x = d["bp_vz"].to_numpy(dtype=float)
    speed = d["speed"].to_numpy(dtype=float) if "speed" in d.columns else np.full(len(d), np.nan)
    lab = labels.to_numpy() if labels is not None else None

    nwin = int(win_sec * fs)
    hop = max(1, int(hop_sec * fs))
    rows, ys, idxs = [], [], []

    for s in range(0, len(x) - nwin + 1, hop):
        e = s + nwin
        sp = np.nanmean(speed[s:e])
        f = window_features(x[s:e], fs, sp)
        if f is None:
            continue
        if lab is not None:
            vals = [v for v in lab[s:e] if isinstance(v, str) and v.strip()]
            if not vals:
                continue
            ys.append(Counter(vals).most_common(1)[0][0])
        f["speed_ms"] = float(sp) if not np.isnan(sp) else np.nan
        f["t_ms"] = float(d["t"].iloc[s])
        f["lat"] = float(d["lat"].iloc[s]) if "lat" in d.columns else np.nan
        f["lon"] = float(d["lon"].iloc[s]) if "lon" in d.columns else np.nan
        rows.append(f)
        idxs.append(s)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS), (np.array([]) if lab is not None else None)

    X = pd.DataFrame(rows)
    return X, (np.array(ys) if lab is not None else None)


# ---------------------------------------------------------------- 事件检测
def estimate_noise_floor(x, fs, win_s=NOISE_WIN_S, q=NOISE_Q):
    """稳健噪声底估计。

    关键：不能用滚动标准差。事件本身会把局部标准差顶高，反而把自己屏蔽掉
    （实测：减速带峰值 3.6 会把滚动 std 从 0.2 抬到 1.38，门限随之升到 5.5，
    结果长时低幅的减速带被系统性漏检，召回率只有 67%）。
    改用 |x| 的 20% 分位数：单次事件占窗口的时间比例远低于 20%，
    因此该分位数几乎只反映背景噪声，不受事件影响。
    """
    w = max(8, int(win_s * fs))
    f = pd.Series(np.abs(x)).rolling(w, center=True,
                                     min_periods=max(4, w // 4)).quantile(q)
    f = f.bfill().ffill()
    return np.maximum(f.to_numpy(dtype=float), 0.02)


def detect_events(d, fs, min_gap_s=MIN_EVENT_GAP_S):
    x = d["bp_mag"].to_numpy(dtype=float)
    n = len(x)
    if n < int(2 * fs) + 8:
        return pd.DataFrame()

    floor = estimate_noise_floor(x, fs)

    peaks, _ = sps.find_peaks(
        x,
        height=np.maximum(PROM_ABS, NOISE_K * floor),
        prominence=np.maximum(PROM_ABS * 0.5, NOISE_K * 0.5 * floor),
        distance=max(1, int(REFRACTORY_S * fs)),
    )
    if len(peaks) == 0:
        return pd.DataFrame()

    # 合并过近的峰（自行车过减速带常连续激励出多个峰）
    keep, last = [], -1e9
    for p in peaks:
        if p - last >= min_gap_s * fs:
            keep.append(p)
        last = p
    peaks = np.array(keep)

    half = int(WIN_SEC * fs / 2)
    vz = d["bp_vz"].to_numpy(dtype=float)
    speed = d["speed"].to_numpy(dtype=float) if "speed" in d.columns else np.full(n, np.nan)
    lat = d["lat"].to_numpy(dtype=float) if "lat" in d.columns else np.full(n, np.nan)
    lon = d["lon"].to_numpy(dtype=float) if "lon" in d.columns else np.full(n, np.nan)

    events = []
    for p in peaks:
        a, b = max(0, p - half), min(n, p + half + 1)
        f = window_features(vz[a:b], fs)
        if f is None:
            continue
        sp = speed[p]
        f["t_ms"] = float(d["t"].iloc[p])
        f["t_s"] = f["t_ms"] / 1000.0
        f["lat"] = float(lat[p]) if not np.isnan(lat[p]) else None
        f["lon"] = float(lon[p]) if not np.isnan(lon[p]) else None
        f["speed_ms"] = float(sp) if not np.isnan(sp) else np.nan
        f["mag_peak"] = round(float(np.max(x[a:b])), 3)
        events.append(f)

    if not events:
        return pd.DataFrame()
    ev = pd.DataFrame(events)

    # 事件密度门控：挂载松动或全程手抖会产生密集伪事件
    dur_km = _track_km(ev)
    if dur_km and len(ev) / dur_km > MAX_EVENTS_PER_KM:
        print("[warn] 事件密度 %.0f 个/km 超过上限 %.0f，疑似挂载松动或设备抖动，"
              "结果仅作参考" % (len(ev) / dur_km, MAX_EVENTS_PER_KM))
    return ev


def _track_km(ev):
    pts = ev.dropna(subset=["lat", "lon"])
    if len(pts) < 2:
        return 0.0
    lat = pts["lat"].to_numpy(dtype=float)
    lon = pts["lon"].to_numpy(dtype=float)
    dx = np.diff(lon) * M_PER_DEG_LON_EQ * np.cos(np.radians(np.mean(lat)))
    dy = np.diff(lat) * M_PER_DEG_LAT
    return float(np.sum(np.hypot(dx, dy))) / 1000.0


# ---------------------------------------------------------------- 规则分类 (v1)
def rule_classify(f):
    """无标签时的启发式分类（冷启动兜底）。

    能力边界要说清楚，方案里别夸大：
    - 规则只能给**已检出的事件**贴类型标签，无法回答「这段路是不是正常平路」。
      而真实场景里 90% 以上的数据是平路，区分"正常振动"与"真实缺陷"才是难点，
      这一层只能由模型承担（见 extract_windows 的窗口级多分类）。
    - 下面的阈值是按 synth() 合成数据的特征均值定的，在合成数据上命中率很高，
      但这恰恰说明它过拟合了合成假设。真实数据上必须用 collect + 手动打标的数据重新标定。
    """
    if f["duration"] > 0.62:
        return "粗糙路面"          # 整窗持续激励
    if f["duration"] >= 0.28:
        return "减速带"            # 宽而平滑的单瓣凸起
    if f["skew"] > 1.0:
        return "井盖"              # 短促圆钝的单瓣，偏度大
    return "坑洼"                  # 短促双极脉冲，近似零偏


# ---------------------------------------------------------------- 聚类与指数
def cluster_events(ev, eps_m=CLUSTER_EPS_M):
    if ev.empty:
        return ev.assign(cluster=-1, confirmations=0)
    ev = ev.copy()
    ev["cluster"] = -1
    valid = ev.dropna(subset=["lat", "lon"])
    if valid.empty:
        ev["confirmations"] = 0
        return ev

    lat0 = float(valid["lat"].mean())
    # 注意单位：必须用「米/度」，否则 eps=15 会被当成 15 公里，把所有事件并成一个点
    xy = np.c_[valid["lon"].to_numpy(dtype=float) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0)),
               valid["lat"].to_numpy(dtype=float) * M_PER_DEG_LAT]

    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps_m, min_samples=1).fit_predict(xy)
    except Exception:
        labels = np.arange(len(xy))

    ev.loc[valid.index, "cluster"] = labels
    cnt = Counter(labels)
    ev["confirmations"] = [int(cnt.get(c, 0)) if c >= 0 else 0 for c in ev["cluster"]]
    return ev


SEVERITY = {"坑洼": 3.0, "减速带": 1.2, "井盖": 1.5, "粗糙路面": 2.0, "其他": 0.8, "平路": 0.0}
SEG_BASE = {"坑洼": 35, "减速带": 70, "井盖": 65, "粗糙路面": 55, "其他": 75, "平路": 100}


def health_index(ev, track_km=None):
    """道路健康度指数 RHI：100 = 完好，0 = 极差。

    采用「加权缺陷密度」而非缺陷绝对数量，否则长路段会被无理由地判成 0 分：
        density = Σ(严重度 × 复现次数权重) / 里程(km)
        RHI     = 100 × exp(-density / K)

    说明：本指数是面向众包感知的自研相对指标，不能替代《公路技术状况评定标准》
    (JTG 5210) 的 PCI/IRI 等需要专业设备标定的官方指标。方案中务必如实说明。
    """
    if ev.empty:
        return 100.0
    km = track_km if track_km is not None else _track_km(ev)
    km = max(km, 0.2)                       # 里程下限，避免短程样本把密度放大到失真
    types = ev["final_type"].fillna("其他")
    sev = types.map(lambda t: SEVERITY.get(t, 0.8)).to_numpy(dtype=float)
    conf = ev["confirmations"].clip(lower=1).to_numpy(dtype=float) if "confirmations" in ev \
        else np.ones(len(ev))
    density = float(np.sum(sev * (1.0 + np.log1p(conf - 1)))) / km
    return round(100.0 * float(np.exp(-density / RHI_K)), 1)


def to_geojson(ev):
    """仅输出聚合后的异常点，不含任何原始轨迹坐标序列，符合个人信息保护要求"""
    if ev.empty:
        return {"type": "FeatureCollection", "features": []}

    pts = [p for p in ev.to_dict("records")
           if p.get("lat") is not None and p.get("lon") is not None]
    if not pts:
        return {"type": "FeatureCollection", "features": []}

    groups = {}
    for p in pts:
        groups.setdefault(int(p.get("cluster", -1)), []).append(p)

    feats = []
    for cid, rows in groups.items():
        t = _vote(rows)
        conf = len(rows)
        feats.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(np.mean([r["lon"] for r in rows])), 7),
                                round(float(np.mean([r["lat"] for r in rows])), 7)],
            },
            "properties": {
                "cluster": cid,
                "type": t,
                "confirmations": conf,
                "max_peak": round(max(float(r["peak"]) for r in rows), 2),
                "mean_duration": round(float(np.mean([r["duration"] for r in rows])), 3),
                "severity_score": SEG_BASE.get(t, 70),
                "maintenance_priority": _priority(t, conf),
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def _vote(rows):
    return Counter(r.get("final_type") or r.get("type") or "其他" for r in rows).most_common(1)[0][0]


def _priority(t, conf):
    base = {"坑洼": "高", "粗糙路面": "中", "井盖": "中", "减速带": "低", "其他": "低"}
    p = base.get(t, "低")
    if conf >= 3 and p == "中":
        p = "高"
    return p


# ---------------------------------------------------------------- 训练
def train(X, y, outdir, label_names=None):
    if y is None or len(y) < 20 or len(set(y)) < 2:
        print("[train] 样本不足或类别单一（需要 >=20 个窗口且 >=2 类），跳过训练")
        return None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder

    Xv = X[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    le = LabelEncoder()
    yv = le.fit_transform(y)
    counts = np.bincount(yv)

    Xtr, Xte, ytr, yte = train_test_split(
        Xv, yv, test_size=0.3, random_state=42, stratify=yv)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    clf.fit(Xtr, ytr)

    print("[train] 留出法测试集报告：")
    print(classification_report(yte, clf.predict(Xte), labels=np.unique(yv),
                                target_names=le.inverse_transform(np.unique(yv)),
                                zero_division=0))

    k = int(min(5, counts.min()))
    if k >= 2:
        cv = cross_val_score(clf, Xv, yv, cv=StratifiedKFold(k, shuffle=True, random_state=42),
                             scoring="f1_macro")
        print("[train] %d 折交叉验证 F1-macro = %.3f (±%.3f)" % (k, cv.mean(), cv.std()))
    else:
        print("[train] 最小类别样本仅 %d 个，跳过交叉验证（这本身就是方案里要讲的问题：样本不平衡）"
              % counts.min())

    os.makedirs(outdir, exist_ok=True)
    imp = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
    imp_df = pd.DataFrame(imp, columns=["feature", "importance"])
    imp_df.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False, encoding="utf-8-sig")
    print("[train] Top5 特征：%s" % ", ".join("%s=%.3f" % (a, b) for a, b in imp[:5]))
    print("[train] 完整特征重要性 -> %s/feature_importance.csv（方案文档可直接引用）" % outdir)
    return clf, le


# ---------------------------------------------------------------- 合成数据
def synth(seed=7, dur=240.0, spacing=7.0, fs=FS_TARGET, speed=6.0):
    """生成带真值标签的合成骑行数据（用于自检与参数标定）

    物理设定：路面激励以**垂直**方向为主，且手机通过支架固连在车上，
    因此振动必须注入与重力同向的轴（az），而不是水平轴。
    若注入水平轴，重力投影通道 vz 里将不含信号，特征全成噪声——
    这是本项目最容易踩的物理陷阱。
    """
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs

    vib = 0.30 * rng.standard_normal(n) + 0.22 * np.sin(2 * np.pi * 12.0 * t)
    labels = np.array(["平路"] * n, dtype=object)
    truth = []
    kinds = ["坑洼", "减速带", "井盖"]
    for i in range(int((dur - 4.0) / spacing)):
        tt = 3.0 + i * spacing
        k = int(tt * fs)
        kind = kinds[i % 3]
        if kind == "坑洼":
            # 凹陷：先向下后回弹的双极尖脉冲，近似零偏、峭度高
            L = int(0.10 * fs)
            vib[k:k + L] += 20.0 * np.hanning(L) * np.sin(np.linspace(0, 2 * np.pi, L))
        elif kind == "减速带":
            # 凸起：宽而平滑的单瓣，持续时间长、峭度低
            L = int(0.42 * fs)
            vib[k:k + L] += 9.0 * np.hanning(L)
        else:
            # 井盖：短促圆钝的单瓣，偏度大
            L = int(0.14 * fs)
            vib[k:k + L] += 11.0 * np.hanning(L)
        labels[max(0, k - int(0.32 * fs)):k + int(0.32 * fs)] = kind
        truth.append((tt, kind))

    # 以 6 m/s 匀速前进（约 1.4 km），轨迹真实可算
    lat = 31.2304 + np.cumsum(np.full(n, speed * np.cos(0.3) / M_PER_DEG_LAT / fs))
    lon = 121.4737 + np.cumsum(np.full(n, speed * np.sin(0.3) /
                                      (M_PER_DEG_LON_EQ * np.cos(np.radians(31.23))) / fs))
    raw = pd.DataFrame({
        "t": t * 1000,
        "ax": 0.02 * rng.standard_normal(n),
        "ay": 0.02 * rng.standard_normal(n),
        "az": 9.81 + vib,
        "lat": lat, "lon": lon, "speed": np.full(n, speed),
        "label_raw": labels,
    })
    return raw, truth


def selftest():
    print("=" * 66)
    print("自检：合成骑行数据 -> 预处理 -> 检测 -> 特征 -> 训练 -> 聚类 -> 导出")
    print("=" * 66)
    raw, truth = synth(seed=7)
    fs = FS_TARGET

    fs_src = infer_source_fs(raw)
    assert abs(fs_src - fs) < 0.5, "源采样率估计偏差过大：%.3f" % fs_src
    print("[1/7] 源采样率估计 %.1f Hz  (生成 %.0f Hz)；注入事件 %d 个" % (fs_src, fs, len(truth)))

    d, fs_used = resample(raw, fs)
    d = preprocess(d, fs_used)
    print("[2/7] 重采样 %d 点 @ %.0fHz；重力分离后 lin_mag 均值 %.3f m/s^2"
          % (len(d), fs_used, d["lin_mag"].mean()))

    ev = detect_events(d, fs_used)
    matched = {}
    for idx, r in ev.iterrows():
        for tt, kind in truth:
            if abs(r["t_s"] - tt) < 0.6:
                matched[idx] = kind
                break
    recall = len(matched) / len(truth)
    fp = len(ev) - len(matched)
    print("[3/7] 检出 %d 个 / 注入 %d 个，召回率 %.0f%%，误报 %d 个"
          % (len(ev), len(truth), recall * 100, fp))
    assert recall >= 0.95, "召回率 %.0f%% 过低，噪声底被事件污染（自遮蔽）" % (recall * 100)
    assert fp <= 1, "误报 %d 个，门限过松（调大 NOISE_K）" % fp

    # 输出各类事件的特征均值，规则阈值就照这张表定
    ev["truth"] = [matched.get(i, "未匹配") for i in ev.index]
    show = ["peak", "rms", "kurtosis", "skew", "duration", "zero_cross", "pos_neg_ratio"]
    print("[4/7] 各真值类别的特征均值（规则阈值应据此设定）：")
    print("      %-10s %8s %8s %9s %8s %9s %7s %8s"
          % ("类别", *[s[:8] for s in show]))
    for k in ("坑洼", "减速带", "井盖"):
        sub = ev[ev["truth"] == k]
        if sub.empty:
            continue
        print("      %-10s %8.2f %8.2f %9.2f %8.2f %9.3f %7.1f %8.2f"
              % (k, *[sub[s].mean() for s in show]))

    ev["final_type"] = [rule_classify(r) for _, r in ev[FEATURE_COLS].iterrows()]
    hit = sum(1 for i in ev.index if ev.at[i, "truth"] != "未匹配"
              and ev.at[i, "final_type"] == ev.at[i, "truth"])
    n_ev = sum(1 for i in ev.index if ev.at[i, "truth"] != "未匹配")
    print("[5/7] 规则分类器命中 %d/%d = %.0f%%  分布 %s"
          % (hit, n_ev, hit / max(n_ev, 1) * 100, dict(ev["final_type"].value_counts())))

    X, y = extract_windows(d, fs_used, labels=raw["label_raw"].astype(str))
    assert y is not None and len(X) == len(y) > 30, "窗口特征提取异常"
    print("[6/7] 窗口特征 %d 个 × %d 维，标签分布 %s"
          % (len(X), len(FEATURE_COLS), dict(Counter(y))))
    print("      规则分类器在窗口级任务上的准确率 %.3f（它根本没有「平路」这个概念，"
          "因此这一层只能靠模型）"
          % float(np.mean([rule_classify(r) == y[i]
                           for i, (_, r) in enumerate(X[FEATURE_COLS].iterrows())])))
    train(X, y, "out_selftest")

    ev = cluster_events(ev)
    gj = to_geojson(ev)
    assert gj["type"] == "FeatureCollection" and gj["features"], "GeoJSON 输出为空"
    assert len(gj["features"]) == len(ev.dropna(subset=["lat", "lon"])), \
        "聚类把不同位置的独立事件并到了一起（检查经纬度换算单位）"
    rhi = health_index(ev)
    assert 0.0 <= rhi < 100.0, "RHI 未反映异常（%s），检查健康度计算" % rhi
    print("[7/7] 空间聚类 %d 个独立异常点 / 事件 %d 个；轨迹 %.3f km；RHI = %s"
          % (len(gj["features"]), len(ev), _track_km(ev), rhi))
    print("      养护优先级 %s" % dict(Counter(f["properties"]["maintenance_priority"]
                                               for f in gj["features"])))
    print("-" * 66)
    print("自检通过。链路完整可跑，把 --input 换成真实数据即可。")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="路面体检 分析基线 v0.1.1")
    ap.add_argument("--input", help="采集端导出的 CSV，或带标签的公开数据集 CSV")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--label-col", default=None, help="公开数据集中逐行标签的列名")
    ap.add_argument("--train", action="store_true", help="存在标签时训练并评估分类器")
    ap.add_argument("--dump-windows", action="store_true", help="额外导出窗口特征表供外部建模")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.input:
        ap.print_help()
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    raw = load_csv(args.input)
    if args.label_col and args.label_col in raw.columns:
        raw = raw.rename(columns={args.label_col: "label_raw"})

    fs_src = infer_source_fs(raw)
    d, fs = resample(raw, FS_TARGET)
    print("输入 %d 点 (约 %.1f Hz) -> 重采样 %d 点 @ %.0f Hz" % (len(raw), fs_src, len(d), fs))
    if fs_src < 40:
        print("[warn] 源采样率偏低（<40Hz），20Hz 以上的路面激励细节会丢失，"
              "建议在方案中说明并做频谱有效带宽讨论")

    d = preprocess(d, fs)

    labels = raw["label_raw"] if "label_raw" in raw.columns else None
    if args.train and labels is not None:
        X, y = extract_windows(d, fs, labels=labels.astype(str))
        if args.dump_windows and len(X):
            X.assign(label=y).to_csv(os.path.join(args.outdir, "windows.csv"),
                                     index=False, encoding="utf-8-sig")
        train(X, y, args.outdir)

    ev = detect_events(d, fs)
    print("检出候选事件 %d 个" % len(ev))
    if ev.empty:
        print("未检出事件。请确认设备已固定、确有位移，且速度门控未滤除全部数据。")
        return

    ev["final_type"] = [rule_classify(r) for _, r in ev[FEATURE_COLS].iterrows()]
    ev = cluster_events(ev)
    ev.insert(0, "id", range(len(ev)))   # 便于与视觉/融合侧对齐
    ev.to_csv(os.path.join(args.outdir, "events.csv"), index=False, encoding="utf-8-sig")

    gj = to_geojson(ev)
    with open(os.path.join(args.outdir, "events.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)

    rhi = health_index(ev)
    report = {
        "input": args.input,
        "samples": int(len(d)),
        "source_fs_hz": round(fs_src, 1),
        "track_km": round(_track_km(ev), 3),
        "events": int(len(ev)),
        "clusters": int(ev["cluster"].nunique()),
        "type_distribution": {str(k): int(v) for k, v in ev["final_type"].value_counts().items()},
        "road_health_index": rhi,
        "priority_distribution": dict(Counter(f["properties"]["maintenance_priority"]
                                              for f in gj["features"])),
        "method": {
            "resample_hz": FS_TARGET, "bandpass_hz": list(BANDPASS),
            "window_s": WIN_SEC,             "peak_rule": "height=max(%.1f, %.1f * quantile(|x|, %.0f%%) over %.0fs centered window)"
                         % (PROM_ABS, NOISE_K, NOISE_Q * 100, NOISE_WIN_S),
            "cluster_eps_m": CLUSTER_EPS_M,
        },
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("道路健康度指数 RHI = %s  |  养护优先级 %s" % (rhi, report["priority_distribution"]))
    print("输出：%s/{events.csv, events.geojson, summary.json}" % args.outdir)


if __name__ == "__main__":
    main()
