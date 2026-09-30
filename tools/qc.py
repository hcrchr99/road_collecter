#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoadCheck 采集包体检工具（T1 qc_pack，规划 §4.2）。

对任意采集包产出 qc.json，一条命令：
    python tools/qc.py --zip road_check/raw/roadcheck_xxx.zip
    python tools/qc.py --all          # 扫描 workspace/road_check{,/raw}/*.zip

全部检查为**确定性**计算，不许模型参与。检查项与判读阈值来自
`roadcheck-package-diagnostics` 诊断经验与《RoadCheck_智能体开发规划》§4.2，
每条 check 携带 status(pass/warn/fail/info) 与依据 note。

核心设计（继承既有实验结论）：
- detect() 复现按 meta.version 分叉（≤0.6 / 0.7 / ≥0.8 三套公式）。
  仓库 replay_detect.py 停在 v0.7 口径，对 v0.8 包会算错，本工具自带第三份实现。
- 零假设检验：替代信号（平稳 / 包络）各 N 次喂给同一检测器，
  噪声事件数 >= 真实事件数 => 事件数不含病害信息。
- 视觉判读阈值换算成"落在本包特征分布的分位"：≈0% => 判据恒真 => 分类器退化为常量。

输出：reports/qc/<pack_id>_qc.json（schema: roadcheck.qc.v0）
"""
import argparse
import json
import math
import sys
import time
from bisect import bisect_left, insort
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "llm_pipeline"))

from llm_pipeline import unpack  # noqa: E402

# ------------------------------------------------------------------ 常量
# 判读阈值全部来自已有诊断经验（规划 §4.2 表 + diagnostics skill），不许拍脑袋。
THR_SAMPLE_RATE = 50.0        # Hz，低于此标黄（正常约 60，实测有 47）
THR_SUSPECT_RATIO = 0.5       # frame_quality.suspect_ratio >= 此值 => 视觉通道不可用
THR_CREST_MEDIAN = 2.0        # crest 中位 < 2 => 一半"事件"没超过本底波动
THR_CREST_TAIL = 4.0          # crest >= 4 占比：真实病害事件的尾部特征
THR_CV_PERIODIC = 0.25        # 间隔 CV 低于此且 min(iv) 贴不应期 => 人造周期嫌疑
THR_DARK_BRIGHTNESS = 50.0    # 平均亮度低于此 => 暗帧
THR_BLUR_LAPVAR = 50.0        # lap_var 低于此 => 失焦/运动模糊（480x360 实测标定）
THR_GAP_MS = 800.0            # 采样中断阈值（与采集端 sampling_gaps 判据一致）

# detect() 三套口径（严格对齐 index.html，实测可复现到事件数与首事件 0 ms）
MIN_SPEED_MS = 8.0 / 3.6
BLIND_ZONE_M = 1.2
REFRACTORY_MIN_MS, REFRACTORY_MAX_MS, REFRACTORY_FALLBACK_MS = 250.0, 1500.0, 800.0
MAX_EVENT_MS = 3000.0
RELEASE_RATIO = 0.5
TAU_SMOOTH = 0.010

# 旧公式（≤v0.7）：max(1.2, 4 × 最近3s |smooth| 的 20% 分位) —— 已证明 ≡ 1.013σ
OLD_NOISE_WIN_MS, OLD_NOISE_Q, OLD_NOISE_K, OLD_NOISE_FLOOR = 3000.0, 0.20, 4.0, 0.02
OLD_REFRACTORY_MS = 1500.0
# 新公式（≥v0.8）：max(1.2, 4 × 最近1s |smooth| 的 RMS)，本底下限 0.02
NEW_NOISE_WIN_MS, NEW_NOISE_K, NEW_NOISE_FLOOR = 1000.0, 4.0, 0.02

# GSD 解析式假设（写明在输出里，供人工核对）
GSD_H_MM, GSD_HFOV_DEG = 1000.0, 68.0
CRACK_VISIBLE_GSD = 2.5       # mm/px，5mm 裂缝占 2px 的要求


def ver_tuple(v):
    try:
        parts = [int(x) for x in str(v).split(".")[:2]]
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts)
    except Exception:
        return (0, 0)


# ------------------------------------------------------------------ 信号基础
def ema_smooth(t_ms, hp):
    """smooth = ema(hp, tau=10ms)，dt = min((t-prev)/1000, 0.2)，首样本 1/60。"""
    n = len(hp)
    sm = np.empty(n)
    s = 0.0
    prev = None
    for i in range(n):
        dt = min((t_ms[i] - prev) / 1000.0, 0.2) if prev is not None else 1.0 / 60.0
        prev = t_ms[i]
        a = math.exp(-dt / TAU_SMOOTH)
        s = s * a + hp[i] * (1.0 - a)
        sm[i] = s
    return sm


def rolling_rms(t_ms, sm, win_ms):
    """最近 win_ms 的 |smooth| RMS（时间窗，含当前样本）。前缀平方和向量化。"""
    a = np.abs(sm)
    ps = np.concatenate(([0.0], np.cumsum(a * a)))
    lo = np.searchsorted(t_ms, t_ms - win_ms, side="left")
    cnt = np.arange(len(t_ms)) - lo + 1
    rms2 = (ps[np.arange(len(t_ms)) + 1] - ps[lo]) / np.maximum(cnt, 1)
    return np.sqrt(np.maximum(rms2, 0.0))


def rolling_quantile_thr(t_ms, sm, win_ms, q, k, floor):
    """旧公式：门限 = max(1.2, k × max(最近 win_ms |smooth| 的 q 分位, floor))。

    有序窗口 + 双端队列，np.percentile 线性插值口径。
    """
    a = np.abs(sm)
    n = len(a)
    thr = np.empty(n)
    buf = deque()
    sorted_vals = []
    for i in range(n):
        v = a[i]
        ti = t_ms[i]
        insort(sorted_vals, v)
        buf.append((ti, v))
        while buf and ti - buf[0][0] > win_ms:
            _, v0 = buf.popleft()
            j = bisect_left(sorted_vals, v0)
            if j < len(sorted_vals):
                del sorted_vals[j]
        w = len(sorted_vals)
        if w == 0:
            qv = 0.0
        else:
            pos = q * (w - 1)
            lo = int(pos)
            hi = min(lo + 1, w - 1)
            frac = pos - lo
            qv = sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
        thr[i] = max(1.2, k * max(qv, floor))
    return thr


def threshold_series(t_ms, sm, spd, ver):
    """按版本分叉的门限序列。返回 (thr, noise_kind)。"""
    if ver >= (0, 8):
        rms = rolling_rms(t_ms, sm, NEW_NOISE_WIN_MS)
        thr = np.maximum(1.2, NEW_NOISE_K * np.maximum(rms, NEW_NOISE_FLOOR))
        return thr, "rms_1s_K4(>=v0.8)"
    if ver >= (0, 7):
        # v0.7：门限仍是旧分位公式，但不应期已改里程盲区
        thr = rolling_quantile_thr(t_ms, sm, OLD_NOISE_WIN_MS, OLD_NOISE_Q,
                                   OLD_NOISE_K, OLD_NOISE_FLOOR)
        return thr, "quantile20_3s_K4(v0.7)"
    thr = rolling_quantile_thr(t_ms, sm, OLD_NOISE_WIN_MS, OLD_NOISE_Q,
                               OLD_NOISE_K, OLD_NOISE_FLOOR)
    return thr, "quantile20_3s_K4(<=v0.6)"


def refractory_of(spd_i, ver):
    if ver < (0, 7):
        return OLD_REFRACTORY_MS
    if spd_i > 0.5:
        return min(REFRACTORY_MAX_MS,
                   max(REFRACTORY_MIN_MS, BLIND_ZONE_M / spd_i * 1000.0))
    return REFRACTORY_FALLBACK_MS


# ------------------------------------------------------------------ detect() 复现
def detect_replay(t_ms, sm, thr, spd, ver, mode):
    """严格复刻 index.html 事件机。返回 (events, missed)。

    events: [(t0, dur, peak, i0)]
      - 结束条件是「任一帧低于释放线 thr*0.5 且时长>60ms」或 3000ms —— 不是"连续60ms"
      - ≤v0.6：moving=(mode=='test') or spd>=8/3.6，不 moving 则事件被静默丢弃
      - ≥v0.7：速度不参与开启判定，只影响不应期（盲区换算）
    missed: {原因: [样本索引]}，原因 ∈ speed_gate / refractory / in_event
    """
    ab = np.abs(sm)
    n = len(t_ms)
    events = []
    missed = {"speed_gate": [], "refractory": [], "in_event": []}
    in_ev = False
    last_end = 0.0
    cur_t0 = cur_peak = cur_i0 = None
    old = ver < (0, 7)
    for i in range(n):
        a = ab[i]
        ti = t_ms[i]
        th = thr[i]
        if in_ev:
            if a > th:
                cur_peak = max(cur_peak, a)
                missed["in_event"].append(i)
            dur = ti - cur_t0
            # ★ 结束条件对每个样本都要判（包括低于门限的样本）：
            #   「任一帧低于释放线 thr*0.5 且时长>60ms」或 3000ms
            if (a < th * RELEASE_RATIO and dur > 60.0) or dur > MAX_EVENT_MS:
                events.append((cur_t0, dur, cur_peak, cur_i0))
                in_ev = False
                last_end = ti
            continue
        if a <= th:
            continue
        if old:
            # test 模式门控关闭。实测 meta 值为 'handheld_test'（不只 'test'），
            # 精确等值比较会漏判（tang_1700 包复现 0 事件的根因）
            moving = ("test" in mode) or (spd[i] >= MIN_SPEED_MS)
            if not moving:
                missed["speed_gate"].append(i)
                continue
        refr = refractory_of(spd[i], ver)
        if (ti - last_end) <= refr:
            missed["refractory"].append(i)
            continue
        in_ev = True
        cur_t0, cur_peak, cur_i0 = ti, a, i
    if in_ev:
        events.append((cur_t0, t_ms[-1] - cur_t0, cur_peak, cur_i0))
    return events, missed


def merge_segments(idxs, t_ms, max_gap_ms=500.0):
    """把被吞样本索引按时间合并成段（间隔>max_gap 断开），返回 [(t0,t1,ms)]。"""
    if not idxs:
        return []
    segs = []
    s = e = t_ms[idxs[0]]
    for i in idxs[1:]:
        ti = t_ms[i]
        if ti - e <= max_gap_ms:
            e = ti
        else:
            segs.append((s, e, e - s))
            s = e = ti
    segs.append((s, e, e - s))
    return segs


# ------------------------------------------------------------------ 各组检查
def add_check(checks, group, item, value, threshold, status, note="", subject=None):
    checks.append({"group": group, "item": item, "value": value,
                   "threshold": threshold, "status": status, "note": note,
                   "subject": subject})


def meta_checks(meta, samples, t_ms, checks):
    ver = ver_tuple(meta.get("version"))
    sr = meta.get("sample_rate_hz")
    if sr is None:
        add_check(checks, "meta", "sample_rate_hz", None, ">=50", "fail", "meta 缺字段")
    else:
        st = "pass" if float(sr) >= THR_SAMPLE_RATE else "warn"
        add_check(checks, "meta", "sample_rate_hz", sr, ">=%.0f" % THR_SAMPLE_RATE, st,
                  "正常约 60 Hz；实测有 47 Hz 的包，采样不足会压低事件幅度")
    n_rows = len(samples)
    ok = (meta.get("sample_count") == n_rows)
    add_check(checks, "meta", "sample_count == samples.csv 行数",
              "%s vs %s" % (meta.get("sample_count"), n_rows), "相等",
              "pass" if ok else "fail", "对不上说明包不完整或被改过")
    dur_meta = meta.get("duration_s")
    dur_real = (t_ms[-1] - t_ms[0]) / 1000.0 if len(t_ms) else 0.0
    if dur_meta:
        rel = abs(dur_real - float(dur_meta)) / max(float(dur_meta), 1)
        add_check(checks, "meta", "duration_s 与样本时间轴",
                  "%.1fs vs %.1fs" % (float(dur_meta), dur_real), "相对差<2%",
                  "pass" if rel < 0.02 else "warn")
    col = meta.get("collection", {}) or {}
    gb = col.get("gate_blocked_segments")
    if gb:
        add_check(checks, "meta", "collection.gate_blocked_segments", gb, "==0", "warn",
                  "被速度门控拦下的段数 >0，说明有静默漏报（时间戳不可重建）")
    else:
        add_check(checks, "meta", "collection.gate_blocked_segments", gb or 0, "==0", "pass")
    su = col.get("speed_usable")
    add_check(checks, "meta", "collection.speed_usable", su, "true",
              "pass" if su else "fail",
              "false 时全趟速度门控相关结论作废，不应期换算也不可信" if not su else "")
    fq = meta.get("frame_quality", {}) or {}
    sratio = fq.get("suspect_ratio")
    if sratio is None:
        add_check(checks, "meta", "frame_quality.suspect_ratio", None, "<0.5", "info",
                  "meta 无该字段（旧版）")
    else:
        st = "pass" if float(sratio) < THR_SUSPECT_RATIO else "fail"
        add_check(checks, "meta", "frame_quality.suspect_ratio", sratio,
                  "<%.1f" % THR_SUSPECT_RATIO, st,
                  "自检只测平坦/偏色，测不了暗与失焦—— vision 组另有补充检查")
    gaps = meta.get("sampling_gaps")
    if gaps is None:
        add_check(checks, "meta", "sampling_gaps", None, "—", "info",
                  "v0.6+ 才有该字段；用样本时间轴自行复核（见 sampling 组）")
    else:
        st = "pass" if gaps.get("usable") else "fail"
        add_check(checks, "meta", "sampling_gaps.usable", gaps.get("usable"), "true", st,
                  "count=%s max=%sms" % (gaps.get("count"), gaps.get("max_ms")))
    ts = meta.get("threshold_stats") or {}
    if ts:
        add_check(checks, "meta", "threshold_stats.floor_active_ratio",
                  ts.get("floor_active_ratio"), "<0.5", "info",
                  "接近 1 = 门限全程退化为固定 1.2（自适应项没起作用）")
    mc = col.get("mode_changes")
    if mc:
        modes = sorted({m.get("mode") for m in mc})
        cons = col.get("mode_consistent")
        add_check(checks, "meta", "collection.mode_consistent", cons, "true",
                  "pass" if cons else "warn",
                  "mode_changes 中出现过的模式：%s" % modes)
    else:
        add_check(checks, "meta", "collection.mode(终值)", col.get("mode"), "—", "info",
                  "旧版只有导出那一刻的模式；若中途切过模式则无法事后重建（已知坑）")
    return ver


def sampling_checks(t_ms, checks):
    if len(t_ms) < 2:
        return
    dt = np.diff(t_ms)
    n0 = int((dt == 0).sum())
    nneg = int((dt < 0).sum())
    nbig = int((dt > THR_GAP_MS).sum())
    add_check(checks, "sampling", "重复时间戳(dt==0)", n0, "—", "info",
              "导出排序过的包 dt==0 对 EMA 无实质影响；逆序才是数据坏了")
    add_check(checks, "sampling", "逆序样本(dt<0)", nneg, "==0",
              "pass" if nneg == 0 else "fail")
    st = "pass" if nbig == 0 else "warn"
    add_check(checks, "sampling", "采样中断(>%dms)" % THR_GAP_MS, nbig, "==0", st,
              "iOS 手指按屏会让 devicemotion 停摆（已知坑），中断期信号不可信")


def reproduction_checks(meta, pack_events, events, thr, t_ms, sm, ver, checks, repro):
    n_meta = int(meta.get("event_count") or 0)
    n_rep = len(events)
    n_pack = len(pack_events)
    repro["meta_event_count"] = n_meta
    repro["replayed_event_count"] = n_rep
    repro["pack_events_csv_rows"] = n_pack

    pack_starts = [int(float(e["t_ms"])) for e in pack_events]
    rep_starts = [int(round(e[0])) for e in events]
    tol = 3
    pi = ri = 0
    matched = 0
    missing_in_rep = []
    while pi < len(pack_starts) and ri < len(rep_starts):
        if abs(pack_starts[pi] - rep_starts[ri]) <= tol:
            matched += 1
            pi += 1
            ri += 1
        elif pack_starts[pi] < rep_starts[ri]:
            missing_in_rep.append(pi)
            pi += 1
        else:
            ri += 1
    missing_in_rep.extend(range(pi, len(pack_starts)))
    repro["matched_starts"] = matched
    repro["pack_events_missing_in_replay"] = [pack_starts[i] for i in missing_in_rep]

    if n_pack and n_rep and n_pack == n_rep:
        pr = [pack_events[i].get("peak") for i in range(min(n_pack, n_rep))]
        ratios = []
        for i in range(min(len(pr), len(events))):
            try:
                pv = float(pr[i])
                if pv > 0:
                    ratios.append(float(events[i][2]) / pv)
            except (TypeError, ValueError):
                pass
        if ratios:
            repro["peak_ratio_median"] = round(float(np.median(ratios)), 4)

    missing_count = n_pack - n_rep
    if missing_count == 0 and n_meta == n_pack:
        add_check(checks, "reproduction", "事件数复现", n_rep, "=meta %d" % n_meta,
                  "pass", "逐事件起始时刻 %d/%d 精确匹配(±%dms)" % (matched, n_pack, tol))
        repro["status"] = "pass"
    elif missing_count == 1 and missing_in_rep == [0]:
        # 已知问题：录制起始处的不应期初值，实际部署的采集端与仓库实现不一致
        pk = pack_events[0]
        peak = float(pk.get("peak") or 0)
        th0 = float(thr[bisect_left(t_ms, pack_starts[0])])
        ratio = peak / th0 if th0 > 0 else None
        repro["first_event_peak_over_threshold"] = round(ratio, 3) if ratio else None
        far = ratio is not None and ratio > 1.15
        add_check(checks, "reproduction", "事件数复现", n_rep, "=meta %d" % n_meta, "warn",
                  "仅缺首个事件(t=%dms, 峰值/门限=%.2f)。%s 判读：起始状态差异"
                  "(采集端录制起始处不应期初值与仓库实现不一致，已知问题)，非 hp 舍入。"
                  % (pack_starts[0], ratio or 0,
                     "信号远离门限" if far else "信号紧贴门限(可能叠加 hp 4位小数舍入)"))
        repro["status"] = "warn_first_event"
    elif ver < (0, 8) and n_pack and matched >= 0.8 * n_pack \
            and missing_count <= max(2, int(0.02 * n_pack)):
        # 旧版分位门限的插值口径与采集端 JS 存在边界差，起始时刻错位会引起
        # 不应期级联漂移；旧包不进判读（旧公式已判废），如实标注即可。
        add_check(checks, "reproduction", "事件数复现", n_rep,
                  "=meta %d / csv %d" % (n_meta, n_pack), "warn",
                  "旧版包：回放 %d/%d，起始匹配 %d（±%dms），错位 %d 处——"
                  "分位插值口径与采集端 JS 实现存在边界差（不应期级联漂移）。"
                  "旧公式已判废，本包不进判读，仅存档。"
                  % (n_rep, n_pack, matched, tol, missing_count))
        repro["status"] = "warn_legacy"
    else:
        extra = ""
        if ver < (0, 6):
            extra = ("；v0.3 采集端口径未完全考证（无 mode_changes，"
                     "speed_usable 见 meta 组），差异如实保留，不影响判废结论")
        add_check(checks, "reproduction", "事件数复现", n_rep,
                  "=meta %d / csv %d" % (n_meta, n_pack), "fail",
                  "先怀疑复现脚本；缺事件起始：%s（最多列 5 个）%s"
                  % ([pack_starts[i] for i in missing_in_rep[:5]], extra))
        repro["status"] = "fail"


def periodicity_checks(events, missed, t_ms, ver, checks, out):
    starts = [e[0] for e in events]
    if len(starts) < 3:
        add_check(checks, "periodicity", "相邻间隔", len(starts), ">=3 事件", "info",
                  "事件太少，无法做周期性体检")
        return
    iv = np.diff(starts)
    cv = float(np.std(iv) / np.mean(iv)) if np.mean(iv) > 0 else None
    min_iv = float(iv.min())
    out["min_interval_ms"] = round(min_iv, 1)
    out["cv"] = round(cv, 3) if cv is not None else None
    out["events"] = len(events)
    if ver < (0, 7):
        refr = OLD_REFRACTORY_MS
    else:
        refr = REFRACTORY_MIN_MS  # 动态不应期下界
    periodic = cv is not None and cv < THR_CV_PERIODIC and min_iv < refr * 1.4
    add_check(checks, "periodicity", "min(iv) 与不应期关系", round(min_iv, 1),
              "<=不应期下界则有人造周期嫌疑", "warn" if periodic else "pass",
              "不应期下界 %dms（≥v0.8 为动态钳制下限；≤v0.6 固定 1500ms）" % int(refr))
    add_check(checks, "periodicity", "间隔 CV", out["cv"], ">=%.2f" % THR_CV_PERIODIC,
              "warn" if (cv is not None and cv < THR_CV_PERIODIC) else "pass",
              "CV 低 = 事件节奏接近规则节拍，疑似不应期充当节奏发生器")
    frac = float(np.mean((iv >= 1400) & (iv <= 2100))) if len(iv) else 0.0
    out["frac_1400_2100"] = round(frac, 3)
    if ver < (0, 8):
        add_check(checks, "periodicity", "旧门限公式", "quantile20_3s",
                  "已废弃", "fail",
                  "max(1.2, 4×20%分位) ≡ 1.013σ（与σ无关），事件连续触发靠不应期压制"
                  "——事件数不含病害信息（meta.detection_criteria.threshold_old_bug）")


def crest_checks(events, t_ms, sm, checks, out):
    if not events:
        add_check(checks, "crest", "crest factor", None, "—", "info", "无事件")
        return
    rms1 = rolling_rms(t_ms, sm, 1000.0)
    crests = []
    for (t0, dur, peak, i0) in events:
        r = rms1[i0]
        if r > 1e-9:
            crests.append(peak / r)
    if not crests:
        return
    med = float(np.median(crests))
    tail = float(np.mean([c >= THR_CREST_TAIL for c in crests]))
    out["crest_median"] = round(med, 2)
    out["crest_ge4_ratio"] = round(tail, 4)
    st = "pass" if med >= THR_CREST_MEDIAN else "fail"
    add_check(checks, "crest", "crest 中位", out["crest_median"],
              ">=%.1f" % THR_CREST_MEDIAN, st,
              "crest 中位 <2 说明一半以上\"事件\"连本底波动都没超过（均匀粗糙路随机起伏 crest 本就 2~3）")
    add_check(checks, "crest", "crest>=%.0f 占比" % THR_CREST_TAIL, out["crest_ge4_ratio"],
              "参考值", "info",
              "尾部占比是真实病害事件的判别特征（零假设检验中纯噪声 ≈0）")


def missed_checks(missed, t_ms, ver, checks, out):
    for k in ("speed_gate", "refractory", "in_event"):
        idxs = missed.get(k, [])
        segs = merge_segments(idxs, t_ms)
        total_ms = sum(s[2] for s in segs)
        longest = sorted(segs, key=lambda s: -s[2])[:3]
        out[k] = {
            "samples": len(idxs),
            "segments": len(segs),
            "total_ms": round(total_ms, 1),
            "longest": [{"t0_ms": int(s[0]), "t1_ms": int(s[1]), "ms": round(s[2], 1)}
                        for s in longest],
        }
    if ver < (0, 7):
        add_check(checks, "missed", "速度门控静默丢弃样本",
                  out["speed_gate"]["samples"], "—", "warn" if out["speed_gate"]["samples"] else "pass",
                  "≤v0.6 低速时事件被直接删除且无时间戳，事后无法重建（v0.7 起已修复）")
    r = out["refractory"]
    add_check(checks, "missed", "不应期吞掉样本", r["samples"], "—", "info",
              "二道防线的正常代价；关键看最长连续段是否覆盖可疑病害时段")
    ie = out["in_event"]
    add_check(checks, "missed", "事件开启期间吞掉样本", ie["samples"], "—", "info",
              "单事件上限 3000ms 内的正常合并")


def surrogate_checks(t_ms, sm, spd, thr, sm_real_events_count, ver, mode, n_each, checks, out):
    """零假设检验：替代信号喂给同一检测器。噪声事件数 >= 真实 => 事件数不含病害信息。"""
    rng = np.random.default_rng(20260926)
    rms1 = rolling_rms(t_ms, sm, 1000.0)
    sigma0 = float(np.median(rms1))
    # 与真实 smooth 同谱的噪声：高斯白噪过同一 EMA
    n = len(t_ms)
    results = {"stationary": [], "envelope": []}
    crest_tails = {"stationary": [], "envelope": []}
    min_ivs = {"stationary": [], "envelope": []}
    for kind in ("stationary", "envelope"):
        for _ in range(n_each):
            w = rng.normal(0.0, 1.0, n)
            s2 = np.empty(n)
            s = 0.0
            prev = None
            for i in range(n):
                dt = min((t_ms[i] - prev) / 1000.0, 0.2) if prev is not None else 1.0 / 60.0
                prev = t_ms[i]
                a = math.exp(-dt / TAU_SMOOTH)
                s = s * a + w[i] * (1.0 - a)
                s2[i] = s
            sd = float(np.std(s2)) or 1.0
            if kind == "stationary":
                sig = s2 / sd * sigma0
            else:
                env = rms1 / (float(np.mean(rms1)) or 1.0)
                sig = env * (s2 / sd) * sigma0
            thr2, _ = threshold_series(t_ms, sig, spd, ver)
            evs, _ = detect_replay(t_ms, sig, thr2, spd, ver, mode)
            results[kind].append(len(evs))
            if evs:
                c = [e[2] / (rms1[e[3]] + 1e-9) for e in evs]
                crest_tails[kind].append(float(np.mean([x >= THR_CREST_TAIL for x in c])))
                iv = np.diff([e[0] for e in evs])
                if len(iv):
                    min_ivs[kind].append(float(iv.min()))
    mean_stat = float(np.mean(results["stationary"])) if results["stationary"] else 0.0
    mean_env = float(np.mean(results["envelope"])) if results["envelope"] else 0.0
    out["n_each"] = n_each
    out["real_events"] = sm_real_events_count
    out["stationary_events_mean"] = round(mean_stat, 1)
    out["stationary_events_sd"] = round(float(np.std(results["stationary"])), 1)
    out["envelope_events_mean"] = round(mean_env, 1)
    out["envelope_events_sd"] = round(float(np.std(results["envelope"])), 1)
    out["noise_crest_ge4_mean"] = {
        "stationary": round(float(np.mean(crest_tails["stationary"])), 4) if crest_tails["stationary"] else 0.0,
        "envelope": round(float(np.mean(crest_tails["envelope"])), 4) if crest_tails["envelope"] else 0.0,
    }
    out["noise_min_interval_ms"] = {
        "stationary": round(float(np.mean(min_ivs["stationary"])), 1) if min_ivs["stationary"] else None,
        "envelope": round(float(np.mean(min_ivs["envelope"])), 1) if min_ivs["envelope"] else None,
    }
    noise_mean = (mean_stat + mean_env) / 2.0
    # 保守口径：真实事件数必须同时高于两种替代信号（取 max）。
    # 用均值会被较宽松的那一种拉低（实测 2125 包：平稳替代 173 > 真实 152，
    # 但均值口径误判"含信息"——与旧公式 ≡1σ 的已知结论矛盾）
    noise_max = max(mean_stat, mean_env)
    informative = sm_real_events_count > noise_max
    out["informative"] = bool(informative)
    add_check(checks, "null_hypothesis", "噪声事件数 vs 真实事件数",
              "%.1f(平稳)/%.1f(包络) vs %d" % (mean_stat, mean_env, sm_real_events_count),
              "真实 > 两种替代", "pass" if informative else "fail",
              "任一替代信号的事件数 >= 真实 => 事件数不含病害信息，不能当\"病害个数\"报")


# ------------------------------------------------------------------ 视觉与物理
def vision_audit(pack, cap, checks, out):
    """帧质量审计（亮度 / 清晰度 / 暗光趋势）。

    ★ 经典 CV 已弃用（2026-09-26 团队决定），不再审计其判据分位与分类分布；
    本组只回答"帧本身可不可用"——这是 LLM 判读通道的输入质量，
    帧自检（meta.frame_quality）测不了暗与失焦，必须自己算。
    """
    sweeps = pack.list_sweep_frames()
    out["sweep_frames_total"] = len(sweeps)
    if not sweeps:
        add_check(checks, "vision", "扫描帧", 0, "—", "info", "包内无扫描帧，视觉通道无数据")
        return
    if len(sweeps) > cap:
        idx = np.linspace(0, len(sweeps) - 1, cap).astype(int)
        sweeps = [sweeps[i] for i in idx]
    import cv2
    rows = []
    for t_ms, zp in sweeps:
        data = pack.read_frame(zp)
        arrb = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arrb, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 全图拉普拉斯方差（清晰度）。在缩小的归一化图上算会失真，
        # 这里直接用原始灰度：分辨率恒定（480x360），跨包可比。
        lap = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
        rows.append({"t_ms": t_ms, "brightness": float(gray.mean()), "lap_var": lap})
    if not rows:
        add_check(checks, "vision", "扫描帧解码", 0, "—", "fail", "全部解码失败")
        return

    bri = np.array([r["brightness"] for r in rows])
    lap = np.array([r["lap_var"] for r in rows])
    tt = np.array([r["t_ms"] for r in rows], dtype=float) / 1000.0
    out["lap_var_median"] = round(float(np.median(lap)), 1)
    out["blur_ratio"] = round(float(np.mean(lap < THR_BLUR_LAPVAR)), 4)
    out["dark_ratio"] = round(float(np.mean(bri < THR_DARK_BRIGHTNESS)), 4)
    out["brightness_median"] = round(float(np.median(bri)), 1)
    if len(tt) > 10 and np.std(bri) > 0:
        r = float(np.corrcoef(tt, bri)[0, 1])
        out["brightness_time_r"] = round(r, 3)
        st = "warn" if r < -0.3 else "pass"
        add_check(checks, "vision", "亮度-时间相关 r", out["brightness_time_r"], ">=-0.3", st,
                  "r 明显为负 = 光照持续恶化（暮光入夜），后半程帧不可再生地变差",
                  subject="vision_frames")
    add_check(checks, "vision", "模糊帧占比(lap_var<%.0f)" % THR_BLUR_LAPVAR,
              out["blur_ratio"], "<0.3", "warn" if out["blur_ratio"] >= 0.3 else "pass",
              "失焦/运动模糊（帧自检测不了暗与糊，必须自己算）",
              subject="vision_frames")
    add_check(checks, "vision", "暗帧占比(亮度<%.0f)" % THR_DARK_BRIGHTNESS,
              out["dark_ratio"], "<0.2", "warn" if out["dark_ratio"] >= 0.2 else "pass",
              subject="vision_frames")


def physics_checks(meta, checks, out):
    exp = meta.get("exported_at")
    dur = float(meta.get("duration_s") or 0)
    win = None
    if exp:
        try:
            t_end = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
            t_local = t_end + timedelta(hours=8)   # 项目在国内，UTC+8
            t_start = t_local - timedelta(seconds=dur)
            win = {"local_start": t_start.strftime("%Y-%m-%d %H:%M"),
                   "local_end": t_local.strftime("%Y-%m-%d %H:%M")}
            out["collection_window_local"] = win
            hh = t_start.hour + t_start.minute / 60.0
            evening = hh >= 16.5
            add_check(checks, "physics", "采集时段(UTC+8)",
                      "%s ~ %s" % (win["local_start"], win["local_end"]),
                      "避开暮光", "warn" if evening else "pass",
                      "9 月下旬华东日落约 17:50；傍晚采集需对照亮度趋势（vision.brightness_time_r）"
                      "判断是否整趟暮光→入夜。exported_at 为 UTC，+8 换算")
        except Exception as e:  # noqa: BLE001
            add_check(checks, "physics", "exported_at 解析", exp, "—", "info", str(e))
    w = int((meta.get("frame_size") or [480, 360])[0])
    f_px = w / (2.0 * math.tan(math.radians(GSD_HFOV_DEG) / 2.0))
    gsds = {}
    for x in (0.5, 1.0, 1.5, 2.0):
        xm = x * 1000.0
        gsds[str(x)] = round((GSD_H_MM ** 2 + xm ** 2) / (GSD_H_MM * f_px), 1)
    out["gsd_mm_per_px"] = {"assumptions": "h=%dmm, HFOV=%d°, 存储宽=%dpx"
                            % (int(GSD_H_MM), GSD_HFOV_DEG, w), "at_x_m": gsds}
    add_check(checks, "physics", "GSD@x(近场~2m)", gsds, "裂缝需<=%.1f" % CRACK_VISIBLE_GSD,
              "info",
              "解析式 GSD(x)=(h²+x²)/(h·f_px)。5mm 裂缝要占 2px 需 GSD<=%.1f mm/px，"
              "当前参数下任何 x 都达不到 => 裂缝细类在本挂载下物理不可见，"
              "分辨率上限是采集端自己缩的（FRAME_W/JPEG q），属改常量可改善项" % CRACK_VISIBLE_GSD)


# ------------------------------------------------------------------ 主流程
def verdict_of(checks, repro, null_out, meta):
    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    ver = ver_tuple(meta.get("version"))
    repro_ok = repro.get("status") in ("pass", "warn_first_event", "warn_legacy")
    informative = bool(null_out.get("informative"))
    old_thr = ver < (0, 8)
    reasons = []
    if not repro_ok:
        reasons.append("detect() 复现失败（先怀疑复现口径）")
    if old_thr:
        reasons.append("旧门限公式（≡1σ，已证明事件数不含病害信息）")
    if null_out and not informative:
        reasons.append("零假设检验未通过（噪声事件数 >= 真实）")
    frames_fail = [c for c in fails if c.get("subject") == "vision_frames"]
    if frames_fail:
        reasons.append("帧质量不可用：%s" % frames_fail[0]["item"])
    can = (repro_ok and (informative or not null_out)
           and not frames_fail and not old_thr)
    if can:
        summary = "可判" + ("（%d 项 warn 需知悉）" % len(warns) if warns else "")
    else:
        summary = "不该判：" + "；".join(reasons)
    return {"can_judge": bool(can), "channels": {
        "imu": "usable" if (repro_ok and (informative or not null_out)) else "unusable",
        "vision_frames": "unusable" if frames_fail else "degraded" if
        [c for c in warns if c.get("subject") == "vision_frames"] else "usable",
    }, "fail_count": len(fails), "warn_count": len(warns), "summary": summary}


def run_pack(zip_path, out_dir, n_each=20, sweep_cap=300, run_surrogates=True):
    t_start = time.time()
    pack = unpack.Pack(zip_path)
    meta = pack.read_meta()
    events_csv = pack.read_events()
    samples = pack.read_samples()
    t_ms = np.array([int(float(r["t_ms"])) for r in samples], dtype=np.float64)
    hp = np.array([float(r["hp"]) for r in samples], dtype=float)
    spd = np.array([float(r.get("speed_ms") or 0) for r in samples], dtype=float)
    sm = ema_smooth(t_ms, hp)

    checks = []
    ver = meta_checks(meta, samples, t_ms, checks)
    sampling_checks(t_ms, checks)

    thr, noise_kind = threshold_series(t_ms, sm, spd, ver)
    events, missed = detect_replay(t_ms, sm, thr, spd, ver,
                                   (meta.get("collection") or {}).get("mode", ""))

    repro = {"noise_kind": noise_kind}
    reproduction_checks(meta, events_csv, events, thr, t_ms, sm, ver, checks, repro)
    periodicity_checks(events, missed, t_ms, ver, checks, per := {})
    crest_checks(events, t_ms, sm, checks, cr := {})
    missed_checks(missed, t_ms, ver, checks, ms := {})
    ms.update(cr)

    # 超门限样本占比（门限健康度：v0.8 实测 0.62%，旧公式 14.4%）
    over = float(np.mean(np.abs(sm) > thr))
    ms["over_threshold_ratio"] = round(over, 5)
    add_check(checks, "threshold", "超门限样本占比", round(over, 5), "参考 <2%",
              "warn" if over > 0.02 else "pass",
              "v0.8 实测 0.62%%；旧公式实测 14.4%%（1σ 门限打在 23~28%% 样本上的直接表现）")

    null_out = {}
    if run_surrogates:
        surrogate_checks(t_ms, sm, spd, thr, len(events), ver,
                         (meta.get("collection") or {}).get("mode", ""),
                         n_each, checks, null_out)
    else:
        add_check(checks, "null_hypothesis", "替代信号检验", "跳过", "—", "info",
                  "--no-surrogates 跳过")

    vis_out = {}
    vision_audit(pack, sweep_cap, checks, vis_out)
    phys_out = {}
    physics_checks(meta, checks, phys_out)
    pack.close()

    verdict = verdict_of(checks, repro, null_out, meta)
    qc = {
        "schema": "roadcheck.qc.v0",
        "pack_id": unpack.pack_id_from_path(zip_path),
        "pack_path": str(Path(zip_path).resolve()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": meta.get("version"),
        "meta_brief": {k: meta.get(k) for k in
                       ("sample_count", "event_count", "duration_s", "sample_rate_hz",
                        "sweep_count", "low_speed_event_count")},
        "detector": {"noise_kind": noise_kind,
                     "refractory": "blind_zone_1.2m(>=v0.7)" if ver >= (0, 7)
                     else "fixed_1500ms",
                     "smooth_tau_s": TAU_SMOOTH},
        "checks": checks,
        "reproduction": repro,
        "periodicity": per,
        "signal": ms,
        "null_hypothesis": null_out,
        "vision": vis_out,
        "physics": phys_out,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("%s_qc.json" % qc["pack_id"])
    out_path.write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
    return qc, out_path


# ------------------------------------------------------------------ 自检
def selftest():
    """合成信号验证复现机：已知 5 个注入冲击 => 恰好 5 个事件；纯噪声 => 0 事件。"""
    rng = np.random.default_rng(7)
    sr = 60.0
    n = 60 * 60
    t = np.arange(n) * (1000.0 / sr)
    hp = rng.normal(0, 0.15, n)
    for c in (5, 15, 25, 35, 45):          # 5 个冲击，幅度 6（远超 4σ 门限）
        i0 = int(c * sr)
        for k in range(-3, 10):
            ii = i0 + k
            if 0 <= ii < n:
                hp[ii] += 6.0 * math.exp(-((k - 1) / 2.5) ** 2)
    spd = np.full(n, 5.0)
    sm = ema_smooth(t, hp)
    thr, kind = threshold_series(t, sm, spd, (0, 8))
    evs, _ = detect_replay(t, sm, thr, spd, (0, 8), "ride")
    assert len(evs) == 5, "自检失败：应检出 5 个事件，实际 %d" % len(evs)
    hp2 = rng.normal(0, 0.15, n)
    sm2 = ema_smooth(t, hp2)
    thr2, _ = threshold_series(t, sm2, spd, (0, 8))
    evs2, _ = detect_replay(t, sm2, thr2, spd, (0, 8), "ride")
    assert len(evs2) == 0, "自检失败：纯噪声应 0 事件，实际 %d" % len(evs2)
    print("selftest OK：5 冲击信号 -> %d 事件；纯噪声 -> 0 事件；门限口径 %s"
          % (len(evs), kind))
    return True


def main():
    ap = argparse.ArgumentParser(description="RoadCheck 采集包体检（T1 qc_pack）")
    ap.add_argument("--zip", action="append", help="采集包 zip（可重复）")
    ap.add_argument("--all", action="store_true", help="扫描 workspace/road_check{,/raw}")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "qc"))
    ap.add_argument("--surrogates", type=int, default=20, help="每种替代信号次数")
    ap.add_argument("--sweep-cap", type=int, default=300)
    ap.add_argument("--no-surrogates", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.selftest:
        selftest()
        return

    paths = list(args.zip or [])
    if args.all or not paths:
        ws = REPO_ROOT.parent
        paths += sorted((ws / "road_check").glob("*.zip"))
        paths += sorted((ws / "road_check" / "raw").glob("*.zip"))
    if not paths:
        ap.print_help()
        sys.exit(1)

    out_dir = Path(args.out)
    lines = ["# RoadCheck 包体检汇总 %s" % datetime.now().strftime("%Y-%m-%d %H:%M"), ""]
    for p in paths:
        qc, out_path = run_pack(p, out_dir, n_each=args.surrogates,
                                sweep_cap=args.sweep_cap,
                                run_surrogates=not args.no_surrogates)
        v = qc["verdict"]
        print("%s | v%s | 复现%s | %s | %.0fs -> %s"
              % (qc["pack_id"], qc["version"], qc["reproduction"].get("status"),
                 v["summary"], qc["elapsed_s"], out_path.name))
        fails = [c["item"] for c in qc["checks"] if c["status"] == "fail"]
        lines.append("## %s（v%s）— **%s**" % (qc["pack_id"], qc["version"], v["summary"]))
        lines.append("")
        lines.append("- 复现：%s（回放 %d / meta %d / csv %d，噪声口径 %s）"
                     % (qc["reproduction"].get("status"),
                        qc["reproduction"].get("replayed_event_count"),
                        qc["reproduction"].get("meta_event_count"),
                        qc["reproduction"].get("pack_events_csv_rows"),
                        qc["detector"]["noise_kind"]))
        if qc["periodicity"]:
            lines.append("- 事件 %d｜min(iv) %sms｜CV %s｜crest 中位 %s｜crest≥4 %s"
                         % (qc["periodicity"].get("events"), qc["periodicity"].get("min_interval_ms"),
                            qc["periodicity"].get("cv"), qc["signal"].get("crest_median"),
                            qc["signal"].get("crest_ge4_ratio")))
        if qc["null_hypothesis"]:
            nh = qc["null_hypothesis"]
            lines.append("- 零假设：真实 %d vs 噪声 %.1f(平稳)/%.1f(包络) => %s"
                         % (nh.get("real_events"), nh.get("stationary_events_mean", 0),
                            nh.get("envelope_events_mean", 0),
                            "含信息" if nh.get("informative") else "**不含病害信息**"))
        if qc["vision"].get("blur_ratio") is not None:
            lines.append("- 帧质量：模糊帧 %.0f%%｜暗帧 %.0f%%｜亮度中位 %s｜亮度趋势 r %s"
                         % ((qc["vision"].get("blur_ratio") or 0) * 100,
                            (qc["vision"].get("dark_ratio") or 0) * 100,
                            qc["vision"].get("brightness_median"),
                            qc["vision"].get("brightness_time_r")))
        if fails:
            lines.append("- fail 项：%s" % "；".join(fails))
        lines.append("")
    md = out_dir / "qc_summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("汇总 -> %s" % md)


if __name__ == "__main__":
    main()
