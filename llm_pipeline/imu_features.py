# -*- coding: utf-8 -*-
"""IMU 派生特征块（开发文档 §4）。

核心决策：不喂裸振幅数值（跨车振幅不可比，同路段两车幅度比中位仅 0.33），
喂「派生特征 + 证据强度分级」，LLM 不得重新解读数值。

信号口径（采集端 index.html 导出，samples.csv 已预计算）：
  hp = mag − ema(mag, 0.5s)   去直流幅值（检测信号域，events.peak 与之同域）
  vz = 线性加速度在重力方向的投影（带符号；|vzMax|>|vzMin| → 凸起，否则凹陷）

新实现的特征（全仓此前无代码）：
  crest        = events.peak ÷ 噪声本底（|hp| 的 20% 分位，±3s 邻域，下钳 0.02，
                 与 analyze.estimate_noise_floor 同思路）
  post_ratio   = 事件结束后 0.3~0.9s 内 hp 均值 ÷ 事件峰值
                 （跨车检验 §6：>0.089 时可复现率 33%→73%；阈值沿用待更多数据验证）
  evidence_tier: 强/中/弱 —— ★ TODO-calibration：分档阈值暂无实测标定数据，
                 当前占位值（8 / 4）仅为工程占位，不得当作标定结论引用。
"""
import numpy as np

# TODO-calibration: 证据强度分档阈值未标定（占位值，勿当结论）
TIER_STRONG, TIER_MEDIUM = 8.0, 4.0
# 沿用跨车检验 §6 的阈值（0.089），待更多重复路段数据验证
POST_RATIO_THRESHOLD = 0.089


def _slice(t_arr, t0, t1):
    i0 = int(np.searchsorted(t_arr, t0, side="left"))
    i1 = int(np.searchsorted(t_arr, t1, side="right"))
    return i0, i1


def compute_event_features(ev, t_arr, hp, vz):
    """ev: events.csv 一行（dict，字符串值）；返回特征 dict（数值化）。"""
    t0 = int(float(ev["t_ms"]))
    dur = int(float(ev.get("duration_ms") or 0))
    t1 = t0 + dur
    peak = float(ev["peak"])

    # --- 事件窗口内 vz 极值（凸起/凹陷判定，与采集端 autoType 同口径）
    i0, i1 = _slice(t_arr, t0, t1)
    vz_win = vz[i0:i1] if i1 > i0 else np.array([])
    if vz_win.size:
        vz_min, vz_max = float(vz_win.min()), float(vz_win.max())
        bump = "凸起" if abs(vz_max) > abs(vz_min) else "凹陷"
        vz_pp = round(vz_max - vz_min, 2)
    else:
        bump, vz_min, vz_max, vz_pp = None, None, None, None

    # --- crest（峰值/本底倍数）
    i0n, i1n = _slice(t_arr, t0 - 3000, t1 + 3000)
    hp_local = np.abs(hp[i0n:i1n]) if i1n > i0n else np.array([])
    if hp_local.size:
        floor = max(float(np.quantile(hp_local, 0.20)), 0.02)
        crest = peak / floor
    else:
        floor, crest = None, None

    # --- 事件后持续位移比值（0.3~0.9s 窗，跨车检验 §6 定义）
    i0p, i1p = _slice(t_arr, t1 + 300, t1 + 900)
    post_win = hp[i0p:i1p] if i1p > i0p else np.array([])
    if post_win.size and peak > 1e-9:
        post_ratio = round(float(np.mean(post_win)) / peak, 3)
    else:
        post_ratio = None

    return {
        "t0_ms": t0,
        "duration_ms": dur,
        "peak": peak,
        "noise_floor": None if floor is None else round(floor, 4),
        "crest": None if crest is None else round(crest, 2),
        "vz_min": None if vz_min is None else round(vz_min, 2),
        "vz_max": None if vz_max is None else round(vz_max, 2),
        "bump": bump,                # 凸起 | 凹陷
        "vz_pp": vz_pp,              # 上下极值差
        "post_ratio": post_ratio,
        "post_ratio_positive": (post_ratio is not None
                                and post_ratio > POST_RATIO_THRESHOLD),
        "speed_kmh": float(ev.get("speed_kmh") or 0.0),
        "low_speed": ev.get("low_speed") == "1",
        "auto_label": ev.get("auto_label") or "",
    }


def evidence_tier(crest) -> str:
    """证据强度分档。★ TODO-calibration：阈值占位（TIER_STRONG/TIER_MEDIUM）。"""
    if crest is None:
        return "弱"
    if crest >= TIER_STRONG:
        return "强"
    if crest >= TIER_MEDIUM:
        return "中"
    return "弱"


def build_feature_block(ev, feat) -> str:
    """生成喂给 L2 的 IMU 特征文本块（模板对齐开发文档 §4.2）。"""
    lines = ["IMU 证据（由确定性代码预计算，模型不得重新解读数值）："]
    lines.append("- 证据强度：%s" % evidence_tier(feat["crest"]))
    lines.append("- 持续时长：%d ms" % feat["duration_ms"])
    if feat["bump"]:
        lines.append("- 垂直分量：%s（上下极值差 %.1f）" % (feat["bump"], feat["vz_pp"]))
    if feat["post_ratio"] is not None:
        lines.append("- 事件后持续位移：比值 %.3f%s"
                     % (feat["post_ratio"],
                        "（>0.089，含持续位移成分）" if feat["post_ratio_positive"] else ""))
    else:
        lines.append("- 事件后持续位移：无有效数据")
    lines.append("- 记录时车速：%.1f km/h；低速标记：%s"
                 % (feat["speed_kmh"], "是" if feat["low_speed"] else "否"))
    lines.append("- 算法自动标签：%s（供对照，不供采信）" % (feat["auto_label"] or "无"))
    return "\n".join(lines)
