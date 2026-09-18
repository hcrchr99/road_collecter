#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路面体检 / RoadCheck - 双模态融合决策 v0.1

核心论点（方案文档里可以直接用）：
    IMU 感知的是「是否引起垂直颠簸」，也就是病害的**深度与结构**；
    视觉感知的是「路面表面长什么样」，也就是病害的**类型**。
    两者不是重复信息，而是互补维度。

    - IMU 抓不到不引起颠簸的病害：纵向裂缝、横向裂缝、龟裂、浅层修补。
      这些在真实路面病害中占比很高（RDD2022 里裂缝类标注远多于坑洼）。
    - 视觉从单目图像无法可靠估计"坑有多深"，也怕夜间、逆光、积水反光；
      而颠簸幅度恰好是深度的直接物理证据。

    融合后病害类型的可识别覆盖从单模态的 4/7 提升到 7/7。

自检：
    python fusion.py --selftest

端到端跑通（先 analyze.py 再 vision.py，最后融合）：
    python fusion.py --imu out/events.csv --vision vis/vision_events.json --outdir fused
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

# 本项目要识别的路面病害/设施全集
UNIVERSE = ["坑洼", "减速带", "井盖", "粗糙路面", "纵向裂缝", "横向裂缝", "龟裂"]

# 各模态在设计上能感知到的类型
IMU_CAN = {"坑洼", "减速带", "井盖", "粗糙路面"}
VISION_CAN = {"坑洼", "井盖", "粗糙路面", "纵向裂缝", "横向裂缝", "龟裂"}

# IMU 侧的"冲击型"与"持续型"事件
IMPULSE_IMU = {"坑洼", "井盖"}
SUSTAINED_IMU = {"减速带", "粗糙路面"}
EVENT_IMU = IMPULSE_IMU | SUSTAINED_IMU

# 视觉侧把裂缝细分成三类时的映射关键字
CRACK_KEYS = {"纵向": "纵向裂缝", "横向": "横向裂缝", "龟裂": "龟裂"}

# 视觉独有病害与 IMU 事件的空间去重半径（米）。
# 同一个坑洼常同时被 IMU 颠簸触发（事件帧）和周期扫描帧看到，
# 不去重会把这处病害算两次，直接夸大病密度。
DEDUP_M = 20.0


def _norm_crack(vision_label, why=""):
    """视觉只说"裂缝"时，尽量用判据里的方向信息细分成三类。"""
    if vision_label != "裂缝":
        return vision_label
    for k, v in CRACK_KEYS.items():
        if k in (why or ""):
            return v
    return "龟裂"


def fuse(imu_label, vision_label, imu_peak=None, vision_conf=None, vision_why=""):
    """融合决策，返回 (最终标签, 置信度 0~1, 规则号, 依据说明)。

    imu_label 为 None 表示该位置 IMU 没有检出事件（不是缺失数据）。
    """
    vc = vision_conf if vision_conf is not None else 0.5
    vl = _norm_crack(vision_label, vision_why)

    # ---- 有视觉、无 IMU 事件：视觉独有的病害，或时间对齐偏差
    if imu_label is None:
        if vl in ("纵向裂缝", "横向裂缝", "龟裂"):
            return vl, min(0.85, 0.55 + vc * 0.3), "R1", \
                "视觉独有能力：裂缝不引起垂直颠簸，IMU 原理上抓不到"
        if vl == "粗糙路面":
            return "粗糙路面", min(0.75, 0.45 + vc * 0.3), "R2", \
                "视觉独有能力：纹理粗糙但未达到 IMU 的速度门控/幅度门控阈值"
        if vl == "坑洼":
            return "疑似浅坑", 0.40, "R3", \
                "视觉看到凹陷但 IMU 未检出颠簸：可能是浅坑、被水覆盖，或摄像头与车轮存在时间对齐偏差（低置信，建议复采）"
        if vl == "井盖":
            return "井盖", 0.35, "R3", "视觉疑似井盖但无颠簸，可能是划线标记或阴影（低置信）"
        return "平路", min(0.9, 0.5 + vc * 0.4), "R4", "双模态一致判定为正常路面"

    # ---- 有 IMU 事件
    if vl == "平路":
        if imu_label in IMPULSE_IMU:
            return "井盖/减速带", 0.70, "R5", \
                "IMU 有冲击但视觉未见表面破损 → 属结构性凸起（井盖、减速带、修补凸台），不是坑洞。单靠 IMU 会误判为坑洼"
        return imu_label, 0.65, "R6", \
            "IMU 判定为持续型颠簸，视觉未发现破损 → 保留 IMU 结论（搓板路/沉陷）"

    if vl == "坑洼":
        if imu_label in IMPULSE_IMU:
            return "坑洼", 0.95, "R7", \
                "双模态互证：IMU 检出冲击 + 视觉检出凹陷，置信度最高"
        return "粗糙路面破损", 0.60, "R8", \
            "视觉见坑洼但 IMU 为持续型颠簸 → 可能是密集成片的坑槽，或 IMU 该处速度过低"

    if vl in ("纵向裂缝", "横向裂缝", "龟裂"):
        if imu_label in IMPULSE_IMU:
            return vl + "(含局部沉降)", 0.65, "R9", \
                "视觉见裂缝 + IMU 有冲击 → 裂缝处存在局部沉降或错台，属需要优先处理的复合病害"
        return vl, 0.75, "R10", "视觉见裂缝 + IMU 持续型颠簸 → 网裂伴随路面变形"

    if vl == "粗糙路面":
        return "粗糙路面", 0.85, "R11", "双模态互证：视觉纹理粗糙 + IMU 持续颠簸"

    if vl == "井盖":
        return "井盖", 0.80, "R12", "视觉见井盖 + IMU 有冲击，一致"

    return imu_label, 0.55, "R13", "视觉判定不明确，回退到 IMU 结论"


def coverage_table():
    """单模态与融合的类型覆盖对比，可直接放进方案。"""
    imu_only = IMU_CAN & set(UNIVERSE)
    vis_only = VISION_CAN & set(UNIVERSE)
    fused = imu_only | vis_only
    return {
        "universe": len(UNIVERSE),
        "imu": sorted(imu_only), "imu_n": len(imu_only),
        "vision": sorted(vis_only), "vision_n": len(vis_only),
        "fused": sorted(fused), "fused_n": len(fused),
        "imu_missing": sorted(set(UNIVERSE) - imu_only),
        "vision_missing": sorted(set(UNIVERSE) - vis_only),
    }


# ---------------------------------------------------------------- 端到端
def load_imu(path):
    """读取 IMU 事件表，兼容采集端 events.csv 与 analyze.py 输出的 events.csv。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for idx, r in enumerate(rows):
        try:
            i = int(float(r.get("id", "")))
        except (TypeError, ValueError):
            i = idx            # analyze.py 早期输出没有 id 列，退化为行号
        label = (r.get("final_label") or r.get("manual") or r.get("auto_label")
                 or r.get("auto") or r.get("type") or r.get("final_type") or "").strip()
        out[i] = {
            "label": label or None,
            "peak": _f(r.get("peak")), "duration": _f(r.get("duration_ms") or r.get("duration")),
            "lat": _f(r.get("lat")), "lon": _f(r.get("lon")),
            "speed": _f(r.get("speed_kmh") or r.get("spd")),
            "noise": label in ("误报", "正常", "平路"),
            "t_ms": _f(r.get("t_ms") or r.get("t")),
        }
    return out


def align_by_time(imu, vis, tol_ms=1200):
    """把视觉事件按时间戳对齐到 IMU 事件上。

    为什么必须做这一步：IMU 侧和视觉侧的 id 来自不同来源，本质上不可比。
    - 采集端路径：IMU events.csv 的 id 与事件帧文件名绑定，两侧 id 一致。
    - analyze.py 路径：它从原始波形**重新检测**事件，id 是自己的行号。
      即使这次恰好和采集端一致（都是 0..8），那也只是巧合——
      一旦检测结果多一个/少一个事件，全部 id 就会整体错位。
    所以不依赖 id 相同，统一按时间戳匹配；匹配不上的 IMU 事件分配远离视觉 id 的键，
    以免与扫描帧的负数 id 冲突。
    """
    if not imu:
        return imu
    vis_event = {k: v for k, v in vis.items()
                 if v.get("source") != "sweep" and v.get("t_ms") is not None}
    if not vis_event:
        return imu                       # 没有可对齐的视觉事件帧

    out, used = {}, set()
    time_match = 0
    for vid in sorted(vis_event):
        v = vis_event[vid]
        best, bd = None, None
        for k, I in imu.items():
            if k in used or I.get("t_ms") is None:
                continue
            d = abs(I["t_ms"] - v["t_ms"])
            if bd is None or d < bd:
                best, bd = k, d
        if best is not None and bd is not None and bd <= tol_ms:
            out[vid] = imu[best]
            used.add(best)
            time_match += 1

    base = (max(vis_event) + 1000) if vis_event else 0
    extra = 0
    for k, I in imu.items():
        if k not in used:
            extra += 1
            out[base + extra] = I
    id_same = len(set(imu) & set(vis_event))
    print("[fusion] 按时间戳对齐（±%d ms）：匹配 %d 个；id 相同的有 %d 个（不作为依据）；"
          "%d 个 IMU 事件无对应视觉帧" % (tol_ms, time_match, id_same, extra))
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_vision(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for e in data.get("events", []):
        try:
            i = int(e.get("id"))
        except (TypeError, ValueError):
            continue
        out[i] = {
            "label": e.get("vision_label"),
            "conf": e.get("vision_conf"),
            "why": e.get("vision_why", ""),
            "frames": e.get("vision_frames", 0),
            "source": e.get("source", "event"),
            "lat": _f(e.get("lat")), "lon": _f(e.get("lon")),
            "t_ms": _f(e.get("t_ms")),
        }
    return out


def haversine_m(a_lat, a_lon, b_lat, b_lon):
    import math
    r = math.pi / 180.0
    dlat = (b_lat - a_lat) * r
    dlon = (b_lon - a_lon) * r
    h = (math.sin(dlat / 2) ** 2
         + math.cos(a_lat * r) * math.cos(b_lat * r) * math.sin(dlon / 2) ** 2)
    return 2 * 6371000 * math.asin(min(1.0, math.sqrt(h)))


def merge(imu, vision, dedup_m=DEDUP_M):
    """合并两路结果。

    重要：周期扫描帧发现的"视觉独有"病害，若落在某个 IMU 事件附近，
    应当归并到该事件上，而不是重复计数——同一个坑洼既被 IMU 颠簸触发、
    又被扫描帧看到，是本项目最常见的重复来源。
    """
    ids = sorted(set(imu) | set(vision))
    rows = []

    # 先收集所有 IMU 事件的位置，用于给视觉独有记录做空间去重
    imu_pts = [(i, I["lat"], I["lon"]) for i, I in imu.items()
               if I.get("lat") is not None and I.get("lon") is not None and not I.get("noise")]

    for i in ids:
        I = imu.get(i)
        V = vision.get(i)

        if I and I["noise"]:
            rows.append({"id": i, "final": "误报", "conf": 0.9, "rule": "R0",
                         "reason": "人工已标定为误报，直接剔除",
                         "imu": I["label"], "vision": V["label"] if V else None,
                         "lat": I["lat"], "lon": I["lon"], "t_ms": I["t_ms"],
                         "source": "event"})
            continue

        imu_label = I["label"] if I else None
        if imu_label in ("", None):
            imu_label = None
        v_label = V["label"] if V else None

        # 视觉独有记录（来自扫描帧，id 为负）：先做空间去重
        if I is None and V is not None and V.get("lat") is not None:
            near = None
            for j, la, lo in imu_pts:
                if haversine_m(V["lat"], V["lon"], la, lo) <= dedup_m:
                    near = j
                    break
            if near is not None:
                rows.append({"id": i, "final": "重复观测", "conf": 0.0, "rule": "R15",
                             "reason": "该处病害已由 IMU 事件 #%s 记录，扫描帧重复看到，已归并" % near,
                             "imu": None, "vision": v_label,
                             "lat": V["lat"], "lon": V["lon"], "t_ms": V.get("t_ms"),
                             "source": "sweep", "merged_into": near})
                continue

        if not V:
            rows.append({"id": i, "final": imu_label or "未知", "conf": 0.5, "rule": "R14",
                         "reason": "该事件没有可用视觉帧，退化为纯 IMU 判定",
                         "imu": imu_label, "vision": None,
                         "lat": I["lat"] if I else None, "lon": I["lon"] if I else None,
                         "t_ms": I["t_ms"] if I else None, "source": "event"})
            continue

        final, conf, rule, reason = fuse(
            imu_label, v_label,
            imu_peak=I["peak"] if I else None,
            vision_conf=V["conf"], vision_why=V["why"])
        rows.append({
            "id": i, "final": final, "conf": round(float(conf), 3),
            "rule": rule, "reason": reason,
            "imu": imu_label, "vision": v_label,
            "vision_frames": V["frames"],
            "peak": I["peak"] if I else None,
            "lat": I["lat"] if I else (V.get("lat") if V else None),
            "lon": I["lon"] if I else (V.get("lon") if V else None),
            "t_ms": I["t_ms"] if I else (V.get("t_ms") if V else None),
            "source": "event" if I else "sweep",
        })
    return rows


def write_out(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    if not rows:
        print("没有可融合的记录")
        return
    keys = ["id", "final", "conf", "rule", "imu", "vision", "reason", "source",
            "merged_into", "peak", "lat", "lon", "t_ms", "vision_frames"]
    with open(os.path.join(outdir, "fused_events.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 只输出聚合后的病害点，不含原始轨迹（合规要求）。
    # 「重复观测」与「平路」「误报」一样不入图，避免同一处病害重复计数。
    EXCLUDE = {"平路", "误报", "未知", "重复观测"}
    feats = []
    for r in rows:
        if r["final"] in EXCLUDE or r["lat"] is None or r["lon"] is None:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {k: r[k] for k in ("id", "final", "conf", "rule", "imu",
                                             "vision", "source", "t_ms") if k in r},
        })
    with open(os.path.join(outdir, "fused_events.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

    cov = coverage_table()
    summary = {
        "records": len(rows),
        "final_distribution": {str(k): int(v) for k, v in Counter(r["final"] for r in rows).items()},
        "rule_distribution": {str(k): int(v) for k, v in Counter(r["rule"] for r in rows).items()},
        "imu_events": sum(1 for r in rows if r["source"] == "event"),
        "sweep_findings": sum(1 for r in rows if r["source"] == "sweep"),
        "deduped": sum(1 for r in rows if r["rule"] == "R15"),
        "vision_only_findings": sum(1 for r in rows
                                    if r["imu"] is None and r["rule"] not in ("R15", "R0")),
        "dual_confirmed": sum(1 for r in rows if r["rule"] == "R7"),
        "mapped_defect_points": len(feats),
        "coverage": cov,
    }
    with open(os.path.join(outdir, "fused_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------- 自检
def selftest():
    print("=" * 70)
    print("自检：融合决策矩阵 -> 类型覆盖度 -> 置信度单调性")
    print("=" * 70)

    print("[1/4] 类型覆盖度（单模态 vs 融合）：")
    cov = coverage_table()
    print("      病害类型全集（%d 类）：%s" % (cov["universe"], "、".join(UNIVERSE)))
    print("      仅 IMU 可识别 %d 类：%s" % (cov["imu_n"], "、".join(cov["imu"])))
    print("      IMU 原理上抓不到：%s" % "、".join(cov["imu_missing"]))
    print("      仅视觉可识别 %d 类：%s" % (cov["vision_n"], "、".join(cov["vision"])))
    print("      视觉单独抓不到：%s" % "、".join(cov["vision_missing"]))
    print("      融合后可识别 %d 类，覆盖率 %.0f%%（单模态最高 %.0f%%）"
          % (cov["fused_n"], cov["fused_n"] / cov["universe"] * 100,
             max(cov["imu_n"], cov["vision_n"]) / cov["universe"] * 100))
    assert cov["fused_n"] > max(cov["imu_n"], cov["vision_n"]), "融合未带来类型覆盖提升"

    print("\n[2/4] 决策矩阵抽查：")
    cases = [
        (None, "纵向裂缝", 0.7, "视觉独有：裂缝不引起颠簸"),
        (None, "平路", 0.8, "无事件 + 平坦 = 平路"),
        (None, "坑洼", 0.7, "视觉见坑但无颠簸，应降置信"),
        ("坑洼", "坑洼", 0.7, "双模态互证，应最高置信"),
        ("坑洼", "平路", 0.8, "IMU 冲击但表面完好，应是结构性凸起"),
        ("减速带", "平路", 0.8, "持续型颠簸 + 表面完好"),
        ("坑洼", "裂缝", 0.6, "裂缝 + 冲击 = 含局部沉降"),
        ("粗糙路面", "粗糙路面", 0.7, "双模态互证"),
    ]
    for imu, vis, vc, note in cases:
        final, conf, rule, reason = fuse(imu, vis, vision_conf=vc)
        print("      IMU=%-8s 视觉=%-10s -> %-16s conf=%.2f %s  [%s]"
              % (str(imu), vis, final, conf, rule, note))

    print("\n[3/4] 置信度单调性检查：")
    c_dual = fuse("坑洼", "坑洼", vision_conf=0.7)[1]
    c_imu = fuse("坑洼", "平路", vision_conf=0.8)[1]
    c_vis_alone = fuse(None, "坑洼", vision_conf=0.7)[1]
    print("      双模态互证 %.2f  >  单模态(结构凸起) %.2f  >  视觉单独(疑似浅坑) %.2f"
          % (c_dual, c_imu, c_vis_alone))
    assert c_dual > c_imu > c_vis_alone, "置信度未体现证据强度差异"
    print("      单调性成立：互证 > 单模态推断 > 单模态孤证")

    print("\n[4/4] 端到端合并逻辑（6 个 IMU 事件 + 2 条扫描帧记录）：")
    imu = {
        0: {"label": "坑洼", "peak": 10.5, "lat": 31.2300, "lon": 121.4700, "noise": False, "t_ms": 3200},
        1: {"label": "井盖", "peak": 9.1, "lat": 31.2310, "lon": 121.4710, "noise": False, "t_ms": 8100},
        2: {"label": "减速带", "peak": 5.6, "lat": 31.2320, "lon": 121.4720, "noise": False, "t_ms": 15000},
        3: {"label": "误报", "peak": 4.0, "lat": 31.2330, "lon": 121.4730, "noise": True, "t_ms": 20000},
        4: {"label": "粗糙路面", "peak": 3.2, "lat": 31.2340, "lon": 121.4740, "noise": False, "t_ms": 24000},
        5: {"label": "坑洼", "peak": 11.2, "lat": 31.2350, "lon": 121.4750, "noise": False, "t_ms": 30000},
    }
    vision = {
        0: {"label": "坑洼", "conf": 0.72, "why": "暗块连通域", "frames": 6, "source": "event"},
        1: {"label": "平路", "conf": 0.80, "why": "各指标均低", "frames": 6, "source": "event"},
        2: {"label": "平路", "conf": 0.75, "why": "各指标均低", "frames": 5, "source": "event"},
        3: {"label": "平路", "conf": 0.80, "why": "各指标均低", "frames": 4, "source": "event"},
        4: {"label": "粗糙路面", "conf": 0.66, "why": "纹理起伏偏高", "frames": 6, "source": "event"},
        5: {"label": "平路", "conf": 0.70, "why": "各指标均低", "frames": 2, "source": "event"},
        # 扫描帧记录（id 为负）：一条远离所有 IMU 事件的裂缝，一条与事件 #0 重合的坑洼
        -1: {"label": "裂缝", "conf": 0.61, "why": "细长暗结构，纵向主导方向",
             "frames": 1, "source": "sweep", "lat": 31.2400, "lon": 121.4800, "t_ms": 45000},
        -2: {"label": "坑洼", "conf": 0.58, "why": "暗块连通域",
             "frames": 1, "source": "sweep", "lat": 31.2300, "lon": 121.4700, "t_ms": 3300},
    }
    rows = merge(imu, vision)
    for r in rows:
        print("      #%-4s IMU=%-8s 视觉=%-8s -> %-12s conf=%.2f %-5s %s"
              % (r["id"], str(r["imu"]), str(r["vision"]), r["final"],
                 r["conf"], r["rule"], r.get("reason", "")[:34]))
    s = write_out(rows, "out_fused_selftest")

    assert s["vision_only_findings"] == 1, \
        "应恰有 1 个视觉独有病害（#-1 裂缝），实际 %d" % s["vision_only_findings"]
    assert s["deduped"] == 1, "应恰有 1 条被空间去重（#-2），实际 %d" % s["deduped"]
    assert s["dual_confirmed"] == 1, "互证事件数应为 1（#0）"
    assert all(r["final"] != "误报" or r["rule"] == "R0" for r in rows)
    print("\n      汇总：%s" % json.dumps(s["final_distribution"], ensure_ascii=False))
    print("      判断规则分布：%s" % json.dumps(s["rule_distribution"], ensure_ascii=False))
    print("      IMU 事件 %d 个、扫描帧记录 %d 条、空间去重 %d 条、视觉独有病害 %d 处、"
          "双模态互证 %d 处、最终上图病害点 %d 个"
          % (s["imu_events"], s["sweep_findings"], s["deduped"],
             s["vision_only_findings"], s["dual_confirmed"], s["mapped_defect_points"]))

    print("-" * 70)
    print("自检通过。输出：out_fused_selftest/{fused_events.csv, fused_events.geojson, fused_summary.json}")


def main():
    ap = argparse.ArgumentParser(description="路面体检 双模态融合决策 v0.1")
    ap.add_argument("--imu", help="analyze.py 输出的 events.csv，或采集端 events.csv")
    ap.add_argument("--vision", help="vision.py 输出的 vision_events.json")
    ap.add_argument("--outdir", default="fused")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.imu and not args.vision:
        ap.print_help()
        sys.exit(1)

    imu = load_imu(args.imu) if args.imu else {}
    vis = load_vision(args.vision) if args.vision else {}
    print("载入 IMU 事件 %d 个、视觉记录 %d 条" % (len(imu), len(vis)))
    imu = align_by_time(imu, vis)

    rows = merge(imu, vis)
    s = write_out(rows, args.outdir)
    if s:
        print("融合后病害分布：%s" % json.dumps(s["final_distribution"], ensure_ascii=False))
        print("IMU 事件 %d 个、扫描帧记录 %d 条、空间去重 %d 条、视觉独有病害 %d 处、"
              "双模态互证 %d 处、最终上图病害点 %d 个"
              % (s["imu_events"], s["sweep_findings"], s["deduped"],
                 s["vision_only_findings"], s["dual_confirmed"], s["mapped_defect_points"]))
        print("输出：%s/{fused_events.csv, fused_events.geojson, fused_summary.json}" % args.outdir)


if __name__ == "__main__":
    main()
