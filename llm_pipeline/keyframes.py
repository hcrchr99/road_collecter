# -*- coding: utf-8 -*-
"""关键帧选取（开发文档 §2.1 修正版）。

★ 实测修正（2026-09-25，对 roadcheck_tang_20260924_1229 全包核查）：
  采集端环形缓存的帧偏移范围为 t0−1200ms ～ t0+154ms，全部在冲击前/即时，
  文档原设想的「t0+200ms 冲击后帧」实际不存在。
  ⇒ 三帧规则修正为：
    A 冲击前   ：最接近 t0−400ms 的帧（缺陷本尊在画面中前方，最完整）
    B 冲击即时 ：最接近 t0 的帧（缺陷已在画面下缘；85/130 事件该帧在冲击后
                 ≤154ms——环形缓存整体偏早所致，不影响判读）
    C 最后视野 ：剩余帧中 offset 最大（最晚）者；当最晚帧已被 B 选走时
                 即为次晚帧（实测 127/130 如此）——语义是「更早的接近视野」，
                 三帧恒不重复
  帧间隔中位约 200ms（6 帧 / 1.2s），A/B 目标命中间隔中位 ≤30ms。
"""
import re

RE_FRAME = re.compile(r"^frames/ev(\d+)_t(\d+)\.jpg$")


def select_keyframes(t0_ms: int, frames, n_frames_expected=6):
    """从事件帧清单中选 3 个关键帧。

    frames: [(t_ms, zip_path)] 升序
    返回: [{"role": "pre|impact|post_proxy", "t_ms", "zip_path",
            "offset_ms", "target_ms"}]；帧不足时尽量补齐（role 顺序优先 A,B,C）
    """
    if not frames:
        return []
    offsets = [(t - t0_ms, t, p) for t, p in frames]

    def nearest(target, taken):
        cands = [o for o in offsets if o[1] not in taken]
        if not cands:
            return None
        return min(cands, key=lambda o: (abs(o[0] - target), o[0]))

    picked, used = [], set()
    # A: 冲击前 t0-400
    a = nearest(-400, used)
    if a:
        picked.append(("pre", -400, a))
        used.add(a[1])
    # B: 冲击即时 t0
    b = nearest(0, used)
    if b:
        picked.append(("impact", 0, b))
        used.add(b[1])
    # C: 剩余帧中最晚者（最后视野帧；保证与 A/B 不重复）
    rest = [o for o in offsets if o[1] not in used]
    if rest:
        c = max(rest, key=lambda o: o[0])
        picked.append(("post_proxy", None, c))
        used.add(c[1])
    else:
        latest = max(offsets, key=lambda o: o[0])
        picked.append(("post_proxy", None, latest))

    out = []
    for role, target, (off, t, p) in picked:
        out.append({
            "role": role,
            "target_ms": target,
            "t_ms": t,
            "offset_ms": off,
            "gap_ms": abs(off - target) if target is not None else None,
            "zip_path": p,
        })
    # 输出顺序固定为 pre / impact / post_proxy
    order = {"pre": 0, "impact": 1, "post_proxy": 2}
    out.sort(key=lambda r: order[r["role"]])
    return out
