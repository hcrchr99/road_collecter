# -*- coding: utf-8 -*-
"""自检规则 R1–R5 与置信度推导（规划 §3.2）。

★ 铁律：规则写死在代码里，不允许模型自由发挥；置信度由证据推导，不由模型声明。
每条规则返回 RuleFire(rule_id, reason, action)；action 由 loop 映射为确定性取证动作。

R5（历史趟冲突）依赖 T7 cross_trip——本期空实现，接口保留、恒不触发。
"""
from __future__ import annotations

from collections import namedtuple

from .. import imu_features

RuleFire = namedtuple("RuleFire", "rule reason action")

# 词表分层（对齐黄金集口径，规划 §6.2：减速带/井盖/标线/接缝均为非病害）：
# 缺陷类 = 黄金集 24 条真值同口径（粗糙 14 / 坑洼 7 / 纵缝 2 / 横缝 1）；
# 构造物类 = 真实存在但非病害的语义类。R1/R3 只对缺陷类触发——
# 构造物与 IMU 的「冲突」是语义分类问题，不是证据问题（实测 ev2 减速带被误降置信）。
DEFECT_LABELS = {"坑洼", "粗糙路面", "纵向裂缝", "横向裂缝", "龟裂"}
STRUCTURE_LABELS = {"减速带", "井盖", "标线", "设计接缝"}

# R1 判据（★ 2026-09-26 按实测分布重标定，偏离规划原文并留档）：
# 规划原文「时长 <100ms 且 crest 高」在 v0.8 数据上无判别力——事件时长被门限
# 量化到 60~90ms（实测 1229 包 P25~P75 = 66~83ms，88% 事件 <100ms，crest 中位 18.8），
# 该条件命中 114/130，循环会全开。
# 改用唯一有实测标定的判别量：跨车检验 §5.4 的 post_ratio > 0.089
# （真实凹陷产生「冲击 + 持续位移」，F1=0.70）。R1 语义变为：
# 视觉报病害但 IMU 无持续位移成分 → 与「真实凹陷必有回弹」冲突，看波形验证。
POST_RATIO_THRESHOLD = imu_features.POST_RATIO_THRESHOLD


def r1_conflict(feat) -> bool:
    """R1 的 IMU 侧前提：报了病害，但波形无持续位移成分（post_ratio 数据可得且 ≤0.089）。"""
    pr = feat.get("post_ratio")
    return pr is not None and not feat.get("post_ratio_positive")

# R2 清晰度阈值（与 tools/qc.py 同一标定：480x360 原始灰度，<50 失焦/模糊）
BLUR_LAP_VAR = 50.0


def frame_usable(kf) -> bool:
    """L0 quality 块口径的兜底可用性（真实清晰度由 tools._lap_var 按需补算，
    这里只用 L0 已有字段：blocked + 暗帧。废帧亮度 <25、可辨 ≥40 为实测标定）。"""
    q = kf.get("quality", {})
    return (not q.get("blocked")) and q.get("brightness", 128) >= 25.0


def check(rec, review, state) -> list:
    """对一次判读结果跑 R1–R5。state: loop 维护的取证状态 dict。

    state 约定字段：
      frames_shown   本轮已喂给模型的帧数（含关键帧）
      actions_done   已执行过的 action 集合（防重复取证死循环）
      rounds         已进行的补充取证轮数
    返回 RuleFire 列表（空 = 通过，可落终判）。
    """
    fires = []
    label = review.get("visual_label", "")
    feat = rec["imu_features"]

    # ---- R1 语义冲突：视觉报病害，但 IMU 波形无持续位移成分（真实凹陷应有「下陷-回弹」）
    if label in DEFECT_LABELS and r1_conflict(feat):
        fires.append(RuleFire(
            "R1", "视觉报「%s」但 IMU 无持续位移成分（post_ratio %.3f ≤ %.3f），"
                  "与真实凹陷「先下陷后回弹」预期冲突，需看冲击后帧与带符号波形验证"
                  % (label, feat.get("post_ratio"), POST_RATIO_THRESHOLD),
            "r1_post_wave"))

    # ---- R2 帧证据不足：可用关键帧 ≤1
    usable = state.get("usable_kf", len([k for k in rec["keyframes"] if frame_usable(k)]))
    if usable <= 1:
        action = "r2_more_frames" if "r2_more_frames" not in state["actions_done"] \
            else "r2_escalate"
        fires.append(RuleFire(
            "R2", "可用帧 ≤1（其余 blocked 或 lap_var<%.0f），证据不足" % BLUR_LAP_VAR,
            action))

    # ---- R3 与 IMU 矛盾：视觉报病害，但 IMU 证据弱（近似"视觉独有"，需独立帧互证）
    if label in DEFECT_LABELS and imu_features.evidence_tier(feat.get("crest")) == "弱":
        action = "r3_two_frames" if "r3_two_frames" not in state["actions_done"] \
            else None
        if action:
            fires.append(RuleFire(
                "R3", "视觉报「%s」但 IMU 证据弱（crest %s），要求至少 2 张独立帧一致，"
                      "否则降置信" % (label, feat.get("crest")),
                action))

    # ---- R4 低置信未复核：模型自报 low → 强制再取一轮证据（不改词表，只加证据）
    if review.get("confidence") == "low" and state["rounds"] < 3:
        fires.append(RuleFire(
            "R4", "模型自报置信 low，强制补充取证后再判", "r4_more_evidence"))

    # ---- R5 与历史趟冲突（阶段 C 前置：cross_trip 本期恒空，接口保留）
    # ct = tools.cross_trip(lat, lon, 15.0)
    # if label in DEFECT_LABELS and len([o for o in ct["observations"] if not o.get("defect")]) >= 2:
    #     fires.append(RuleFire("R5", "历史 ≥2 趟同位置无观测", "r5_downweight"))
    return fires


def derive_confidence(rec, review, state, escalations) -> tuple:
    """置信度推导（护栏：不由模型声明）。返回 (confidence, 依据说明)。

    规划 §3.2：双模态互证 = high；单模态 + 多帧一致 = mid；单帧孤证 / 与 IMU 矛盾 = low。
    多帧"一致"的工程近似：最终证据中可用帧 ≥2 且 R3 未火（帧间一致性未逐帧投票，
    如实记录在 trace，不夸大）。
    """
    label = review.get("visual_label", "")
    feat = rec["imu_features"]
    tier = imu_features.evidence_tier(feat.get("crest"))
    fired = {f.rule for f in state.get("last_fires", [])}
    frames_ok = state.get("usable_kf_final", 0)

    if escalations:
        return "low", "升级人工复核（%s）" % "；".join(escalations[-1:])

    if label not in DEFECT_LABELS:
        # 非缺陷结论（平路/遮挡/构造物类）：置信看帧证据是否充分
        return ("high", "非病害结论且帧证据可用" if frames_ok >= 2
                else "mid" if frames_ok == 1 else "low")

    # 病害类
    if fired & {"R1"}:
        return "low", "R1 未消解：视觉与 IMU 语义冲突（单点毛刺型）"
    if tier in ("强", "中") and frames_ok >= 2:
        return "high", "双模态互证（IMU %s 档 + %d 帧可用）" % (tier, frames_ok)
    if frames_ok >= 2 and "R3" not in fired:
        return "mid", "单模态 + 多帧可用（IMU 证据弱，未达互证）"
    return "low", "单帧孤证或与 IMU 矛盾（可用帧 %d，IMU %s 档）" % (frames_ok, tier)
