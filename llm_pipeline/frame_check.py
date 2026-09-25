# -*- coding: utf-8 -*-
"""帧自检：Python 侧复现采集端 index.html frameQuality()（L830-875）。

判据（逐行对照 JS 原实现，2026-09-25 核对）：
  - 60×45 降采样（JS canvas.drawImage 线性重采样 → 此处用 PIL BOX/面积平均，
    对判据结论的影响可忽略）
  - uniq  = 该降采样图上唯一 RGB 颜色数
  - std   = 灰度（0.299r+0.587g+0.114b）标准差
  - cast  = RGB 三通道均值的最大两两差（单色遮挡物会显著偏色）
  - flat   = std<9.0 且 uniq<400   → 遮挡/未出画
  - tinted = cast>40               → 单色遮挡
  - suspect = flat 或 tinted
另按开发文档 §2.1 补充亮度判据：mean<40 → 暗（暗光废帧实测：亮度<25 全废、
≥40 可辨，v0.8 首包验收结论）→ 单独标记 dark，不并入 suspect（因白天包
占比应极低；是否作为 blocked 判据之一由调用方决定）。
"""
import io

from PIL import Image

W, H = 60, 45


def self_check(jpeg_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((W, H), Image.BOX)
    px = list(img.getdata())

    n = len(px)
    seen = set()
    grays = []
    sr = sg = sb = 0
    for r, g, b in px:
        seen.add((r, g, b))
        grays.append(0.299 * r + 0.587 * g + 0.114 * b)
        sr += r
        sg += g
        sb += b
    mean = sum(grays) / n
    std = (sum((x - mean) ** 2 for x in grays) / n) ** 0.5
    mr, mg, mb = sr / n, sg / n, sb / n
    cast = max(abs(mr - mg), abs(mg - mb), abs(mr - mb))
    uniq = len(seen)

    flat = std < 9.0 and uniq < 400
    tinted = cast > 40
    dark = mean < 40.0
    suspect = flat or tinted

    reason = ""
    if suspect:
        reason = ("画面几乎无纹理，疑似镜头被遮挡或未出画" if flat
                  else "画面严重偏色，疑似镜头被单色遮挡物挡住")
    elif dark:
        reason = "画面过暗（亮度均值<40），暗光判读不可靠"

    return {
        "uniq": uniq,
        "std": round(std, 1),
        "cast": round(cast, 1),
        "brightness": round(mean, 1),
        "flat": flat,
        "tinted": tinted,
        "dark": dark,
        "suspect": suspect,        # 与采集端 frameQuality 同口径
        "blocked": suspect or dark,  # L0 判定是否进入 L1/L2
        "reason": reason,
    }
