# -*- coding: utf-8 -*-
"""工具层（规划 §4.1）：T2/T4/T5/T7 + 波形/帧取证的原语。

设计约束：
- 全部为确定性代码，模型不直接调用工具——由 selfcheck 规则触发、loop 执行
  （规划 §3.2：「取证动作」由规则表 prescribe，模型只做判读）。
- zipfile.read 非线程安全：read_frame 由调用方持锁（沿用 judge_pack 的 zip_lock 约定）。
- ★ 不得上传位置数据：T7 cross_trip 本期返回空实现，且任何工具输出不含 lat/lon。
"""
from __future__ import annotations

import numpy as np

from .. import imu_features
from .selfcheck import r1_conflict

FRAME_NEAR_MS = 400        # T4 取帧时允许的最大偏移误差（超出则报无帧）
WAVE_MAX_POINTS = 48       # T5 波形序列最多点数（控制 token）


def evidence_tier_of(feat) -> str:
    return imu_features.evidence_tier(feat.get("crest"))


class AgentTools:
    """一个采集包的工具集视图。"""

    def __init__(self, pack, records, t_arr, hp, vz):
        self.pack = pack
        self.by_id = {r["event_id"]: r for r in records}
        self.frames_by_ev = pack.list_event_frames()
        self.t_arr = t_arr
        self.hp = hp
        self.vz = vz
        self._lap_cache = {}

    # -------------------------------------------------------------- T2
    def list_events(self, flt: str = "all"):
        """事件清单 + 筛选理由。flt ∈ all/low_speed/blocked/conflict/uncertain。

        conflict = R1 候选（报病害才会触发：无持续位移成分）；
        uncertain = 开循环候选（IMU 证据弱 / 低速 / conflict）。
        """
        out = []
        for r in self.by_id.values():
            if r["blocked"]:
                if flt in ("all", "blocked"):
                    out.append({"event_id": r["event_id"], "filter": "blocked"})
                continue
            feat = r["imu_features"]
            reason = None
            if r.get("low_speed"):
                reason = "low_speed"
            elif r1_conflict(feat):
                reason = "conflict"
            elif evidence_tier_of(feat) == "弱":
                reason = "uncertain"
            if reason and flt in ("all", "uncertain", reason):
                out.append({"event_id": r["event_id"], "filter": reason})
        return out

    # -------------------------------------------------------------- T4
    def read_frame(self, event_id: int, offset_ms: int = 0, crop: float = 1.0,
                   zoom: float = 1.0, preprocess: str = "none"):
        """读事件帧：取最接近 t0+offset_ms 的一帧，可裁剪/放大/预处理。

        crop: 保留的画面比例（0<c<=1），锚定在画面**中下部**（缺陷本体所在，
              与采集端挂载几何一致）；zoom: 放大倍数（纯 resize）。
        返回 (jpeg_bytes, meta)；无可用帧返回 (None, meta)。
        """
        from ..judge import preprocess_frame

        rec = self.by_id[event_id]
        t0 = rec["t0_ms"]
        fl = self.frames_by_ev.get(event_id, [])
        if not fl:
            return None, {"event_id": event_id, "offset_ms": offset_ms, "found": False,
                          "reason": "包内无该事件帧"}
        best = min(fl, key=lambda f: abs((f[0] - t0) - offset_ms))
        actual = best[0] - t0
        data = self.pack.read_frame(best[1])
        if crop < 1.0 or zoom != 1.0:
            import cv2
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                c = min(max(float(crop), 0.1), 1.0)
                ch, cw = int(h * c), int(w * c)
                # 中下部锚定：水平居中，垂直取下半（y0 = h-ch）
                y0, x0 = h - ch, (w - cw) // 2
                img = img[y0:y0 + ch, x0:x0 + cw]
                if zoom > 1.0:
                    img = cv2.resize(img, None, fx=float(zoom), fy=float(zoom),
                                     interpolation=cv2.INTER_CUBIC)
                import cv2 as _cv2
                ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    data = buf.tobytes()
        return preprocess_frame(data, preprocess), {
            "event_id": event_id, "offset_ms": offset_ms, "actual_offset_ms": actual,
            "crop": crop, "zoom": zoom, "preprocess": preprocess, "found": True,
        }

    # -------------------------------------------------------------- T5
    def waveform(self, event_id: int, window_ms: int = 1200):
        """事件附近带符号垂直分量（vz）与 hp 的数值序列 + 派生判读（判 R1 用）。

        窗口 = [t0-200, t0+window_ms]。输出含极值/极性/衰减时长，模型不得重新解读。
        """
        rec = self.by_id[event_id]
        t0 = rec["t0_ms"]
        dur = rec["imu_features"].get("duration_ms", 0)
        i0 = int(np.searchsorted(self.t_arr, t0 - 200, side="left"))
        i1 = int(np.searchsorted(self.t_arr, t0 + max(window_ms, dur + 400), side="right"))
        if i1 <= i0:
            return {"event_id": event_id, "text": "波形窗口内无样本", "series": []}
        t = self.t_arr[i0:i1]
        vz = self.vz[i0:i1]
        hp = self.hp[i0:i1]
        step = max(1, len(t) // WAVE_MAX_POINTS)
        series = [{"dt_ms": int(t[k] - t0), "vz": round(float(vz[k]), 1),
                   "hp": round(float(hp[k]), 2)} for k in range(0, len(t), step)]
        # 极性与极值（与 imu_features 同口径）
        vz_min, vz_max = float(vz.min()), float(vz.max())
        bump = "凸起" if abs(vz_max) > abs(vz_min) else "凹陷"
        peak = float(np.abs(hp).max())
        # 衰减时长：事件起点后 |hp| 首次持续低于 30% 峰值的时刻
        below = np.abs(hp) < 0.3 * peak
        decay_ms = None
        for k in range(len(below)):
            if below[k] and below[min(k + 3, len(below) - 1)]:
                decay_ms = int(t[k] - t0)
                break
        text = (
            "补充波形（确定性提取）：事件 #%d 起前后 %.0fms 窗\n"
            "- vz 极值：min %.1f / max %.1f（极性判定：%s）\n"
            "- hp 峰值 %.2f；衰减至 30%% 峰值耗时 %s\n"
            "- 序列(dt_ms,vz,hp)：%s"
            % (event_id, window_ms, vz_min, vz_max, bump, peak,
               ("%dms" % decay_ms) if decay_ms is not None else "窗内未衰减",
               " ".join("(%d,%s,%s)" % (p["dt_ms"], p["vz"], p["hp"]) for p in series[:24]))
        )
        return {"event_id": event_id, "text": text, "series": series,
                "bump": bump, "vz_min": vz_min, "vz_max": vz_max,
                "decay_ms": decay_ms}

    # -------------------------------------------------------------- T7（阶段 C 前置，本期空实现）
    def cross_trip(self, lat: float, lon: float, radius_m: float):
        """历史趟在同位置的观测列表。本期返回空（接口预留，聚合原则见规划 §3.3）。"""
        return {"observations": [], "note": "cross_trip 未启用（阶段 C）"}

    # -------------------------------------------------------------- 证据组装
    def _lap_var(self, zip_path) -> float:
        """480x360 原始灰度全图拉普拉斯方差（与 tools/qc.py 同口径，带缓存）。
        L0 的 quality 块没有 lap_var 字段（只有 std/uniq/cast/brightness/blocked），
        清晰度必须自己算——qc 诊断已证明帧自检测不了暗与糊。"""
        if zip_path not in self._lap_cache:
            import cv2
            arr = np.frombuffer(self.pack.read_frame(zip_path), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self._lap_cache[zip_path] = 0.0
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self._lap_cache[zip_path] = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
        return self._lap_cache[zip_path]

    def usable_keyframes(self, rec):
        return [k for k in rec["keyframes"]
                if not k["quality"]["blocked"]
                and k["quality"].get("brightness", 128) >= 25.0
                and self._lap_var(k["zip_path"]) >= 50.0]

    def extra_frames_evidence(self, event_id, n=3, exclude_roles=("pre", "impact", "post_proxy"),
                              preprocess="clahe"):
        """R2/R4 取证：从包内该事件全部帧中取关键帧之外的 n 帧（CLAHE）。"""
        from ..judge import preprocess_frame

        rec = self.by_id[event_id]
        t0 = rec["t0_ms"]
        used = {k["zip_path"] for k in rec["keyframes"]}
        cand = [f for f in self.frames_by_ev.get(event_id, []) if f[1] not in used]
        if not cand:
            return []
        if len(cand) > n:
            idx = np.linspace(0, len(cand) - 1, n).astype(int)
            cand = [cand[i] for i in dict.fromkeys(idx)]
        return [((f[0] - t0), preprocess_frame(self.pack.read_frame(f[1]), preprocess))
                for f in cand]
