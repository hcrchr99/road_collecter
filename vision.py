#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路面体检 / RoadCheck - 视觉侧 v0.1

两条后端：
  1) classical（默认，只需要 OpenCV）：纹理/边缘/暗斑/结构张量相干性，零模型依赖，立刻可跑
  2) yolo（装了 ultralytics 自动启用）：在 RDD2022 上微调的目标检测，识别
     D00 纵向裂缝 / D10 横向裂缝 / D20 龟裂 / D40 坑洼

设计哲学与 IMU 侧的规则分类器一致：先用一个能跑通的基线占住位置，
模型作为可替换后端接入。基线在合成图上表现良好不代表真实数据可用。

自检：
    python vision.py --selftest

处理采集端导出的 ZIP：
    python vision.py --zip roadcheck_20260918_2130.zip --outdir vis
"""

import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter

# Windows 上重定向 stdout 到文件时默认走 GBK(cp936)，会把中文报告写成乱码。
# 强制 UTF-8，保证 `python vision.py --selftest > report.txt` 产出可读报告。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ---------------------------------------------------------------- 配置
ROI_TOP = 0.35        # 裁掉上方（天空/远景），只保留路面
ROI_SIDE = 0.08       # 左右各裁一点（车把/支架）
ILLUM_SIGMA = 15.0    # 光照背景估计的高斯尺度（远大于病害尺度）
DARK_DROP = 0.15      # 相对周围亮度下降超过该比例才算"暗"
MIN_BLOB_PX = 60      # 连通域面积下限（滤掉噪点与纹理颗粒）
BH_T = 28             # 黑帽响应的绝对门限（norm8 灰度单位）。不能用 Otsu：
                      # 近均匀图像上 Otsu 会把噪声随意切开，实测把 60% 像素判成裂缝
SEVERITY_RANK = {"平路": 0, "粗糙路面": 1, "裂缝": 2, "井盖": 3, "减速带": 3, "坑洼": 4}

# RDD2022 四类 -> 本项目病害体系
RDD_MAP = {"D00": "裂缝", "D10": "裂缝", "D20": "裂缝", "D40": "坑洼"}
RDD_NAME = {"D00": "纵向裂缝", "D10": "横向裂缝", "D20": "龟裂", "D40": "坑洼"}


# ---------------------------------------------------------------- 图像读取
def read_image(data):
    """data 可以是路径、bytes 或 ndarray"""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (bytes, bytearray)):
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.imread(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图像：%s" % data)
    return img


def road_roi(img):
    """裁出路面区域。

    摄像头要么朝前（画面里大量天空/远景），要么朝下。统一取下部中央区域，
    是应对"不知道用户怎么架手机"最稳的做法。
    """
    h, w = img.shape[:2]
    y0 = int(h * ROI_TOP)
    x0 = int(w * ROI_SIDE)
    roi = img[y0:h, x0:w - x0]
    if roi.size == 0:
        roi = img
    return cv2.resize(roi, (320, 240), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 特征
def _largest_blob(mask):
    """返回 (最大连通域面积, 总面积, 面积占比, 最大连通域的细长比)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0, mask.size, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= MIN_BLOB_PX
    if not keep.any():
        return 0, mask.size, 0.0, 0.0
    i = int(np.argmax(np.where(keep, areas, 0))) + 1
    a = int(stats[i, cv2.CC_STAT_AREA])
    w = max(int(stats[i, cv2.CC_STAT_WIDTH]), 1)
    h = max(int(stats[i, cv2.CC_STAT_HEIGHT]), 1)
    elong = max(w, h) / max(min(w, h), 1)
    return a, int(mask.size), a / float(mask.size), float(elong)


def features(img):
    """路面特征。

    关键设计：先做**光照归一化**再判断明暗。直接用"低于均值 N 倍标准差"这类
    绝对门限是错的——在纯噪声图像上它天然产生固定误报率（实测恒为 18%），
    会把平滑路面全判成坑洼。归一化后再配合连通域面积过滤，才既抗光照又抗噪点。
    """
    roi = road_roi(img)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)          # 去传感器噪点，保留路面纹理
    g = gray.astype(np.float32)

    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=ILLUM_SIGMA)
    norm = g / np.maximum(bg, 1.0)                    # 相对周围亮度，≈1 为正常
    norm8 = np.clip(norm * 128.0, 0, 255).astype(np.uint8)

    # 1) 纹理对比度
    gray_std = float(np.std(norm8))
    lap_var = float(np.var(cv2.Laplacian(norm8, cv2.CV_32F)))
    edges = cv2.Canny(norm8, 20, 60)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    # 2) 坑洼：比周围暗的大块连通区域
    dark = (norm < 1.0 - DARK_DROP).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark_area, total, dark_blob_ratio, dark_elong = _largest_blob(dark)
    dark_ratio = float(np.count_nonzero(dark)) / dark.size

    # 3) 裂缝：黑帽运算提取细长暗结构。
    #    这里必须用绝对门限，不能用 Otsu——近均匀图像上 Otsu 会把噪声当"目标"切开，
    #    实测把平滑路面的 60% 像素判成裂缝，特征完全失效。
    bh = cv2.morphologyEx(norm8, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    cm = (bh > BH_T).astype(np.uint8) * 255
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    crack_area, _, crack_blob_ratio, crack_elong = _largest_blob(cm)
    crack_ratio = float(np.count_nonzero(cm)) / cm.size

    # 4) 方向相干性：区分有方向的裂缝与各向同性的粗糙纹理
    f32 = norm8.astype(np.float32)
    ix = cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(f32, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float(np.sum(ix * ix)), float(np.sum(iy * iy)), float(np.sum(ix * iy))
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    coherence = float((l1 - l2) / (l1 + l2)) if (l1 + l2) > 1e-9 else 0.0
    orient = float(0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy))) if tr > 1e-9 else 0.0

    hist = cv2.calcHist([norm8], [0], None, [32], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))

    return {
        "gray_std": round(gray_std, 2),
        "lap_var": round(lap_var, 1),
        "edge_density": round(edge_density, 4),
        "dark_ratio": round(dark_ratio, 4),
        "dark_blob_ratio": round(dark_blob_ratio, 4),
        "dark_blob_px": dark_area,
        "dark_elong": round(dark_elong, 2),
        "crack_ratio": round(crack_ratio, 4),
        "crack_blob_ratio": round(crack_blob_ratio, 4),
        "crack_px": crack_area,
        "crack_elong": round(crack_elong, 2),
        "entropy": round(entropy, 3),
        "coherence": round(coherence, 4),
        "orient_deg": round(orient, 1),
    }


# ---------------------------------------------------------------- 经典 CV 基线
def classify_classical(f):
    """启发式分类，返回 (标签, 置信度 0~1, 判据说明)。

    阈值按合成图特征均值设定，真实数据上必须重新标定——
    它存在的意义是占住基线位置，不是替代模型。

    判别顺序与依据：
      坑洼 = "大块且不细长"的暗连通域（凹陷阴影成片）
      裂缝 = "细长"的暗结构（细长比高）
      两类的暗斑面积都可能超阈值，所以必须用细长比而不是面积来区分。
    """
    if f["dark_blob_px"] >= MIN_BLOB_PX and f["dark_blob_ratio"] > 0.008 and f["dark_elong"] < 4.0:
        return "坑洼", min(0.9, 0.45 + f["dark_blob_ratio"] * 8), \
            "存在相对周围明显偏暗的大块连通区域（%d px，细长比 %.1f）" % (f["dark_blob_px"], f["dark_elong"])
    if f["crack_px"] >= MIN_BLOB_PX and f["crack_elong"] >= 3.0 and f["crack_ratio"] > 0.003:
        dirname = "纵向" if abs(f["orient_deg"]) < 45 or abs(f["orient_deg"]) > 135 else "横向"
        return "裂缝", min(0.85, 0.35 + f["crack_ratio"] * 12), \
            "检测到细长暗结构（细长比 %.1f，%s主导方向）" % (f["crack_elong"], dirname)
    if f["gray_std"] > 1.5:
        return "粗糙路面", min(0.8, 0.3 + f["gray_std"] / 20.0), "纹理起伏偏高（归一化灰度标准差 %.2f）" % f["gray_std"]
    return "平路", 0.8, "各项病害指标均低"


# ---------------------------------------------------------------- YOLO 后端
class YoloBackend:
    """在 RDD2022 上微调过的检测器。装好 ultralytics 后自动可用。"""

    def __init__(self, weights):
        from ultralytics import YOLO       # 延迟导入，未安装时不影响经典基线
        self.model = YOLO(weights)
        self.weights = weights

    def __call__(self, img, conf=0.25):
        res = self.model.predict(img, conf=conf, verbose=False)[0]
        names = res.names
        dets = []
        for b in res.boxes:
            code = str(names[int(b.cls)])
            label = RDD_MAP.get(code.split("_")[0].upper())
            if label is None:
                continue
            dets.append({
                "code": code,
                "label": label,
                "detail": RDD_NAME.get(code.split("_")[0].upper(), code),
                "conf": round(float(b.conf), 3),
                "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
            })
        return dets


def try_yolo(weights):
    if not weights or not os.path.exists(weights):
        return None
    try:
        return YoloBackend(weights)
    except Exception as e:
        print("[vision] YOLO 后端不可用（%s），改用经典 CV 基线" % e)
        return None


def classify_yolo(dets):
    if not dets:
        return "平路", 0.6, "检测器未发现病害"
    best = max(dets, key=lambda d: (SEVERITY_RANK.get(d["label"], 0), d["conf"]))
    names = "、".join(sorted({d["detail"] for d in dets}))
    return best["label"], best["conf"], "检测到 %s" % names


# ---------------------------------------------------------------- 事件窗口聚合
def worst_of(items):
    """在一段事件窗口的所有帧里取最严重的判断，返回 (标签, 置信度, 判据)。

    items 支持 [(label, conf)] 或 [(label, conf, why)] 两种形式。

    为什么取最严重而不是取多数：视觉与 IMU 存在时间对齐误差
    （摄像头朝向前方、坑洼可能只有一两帧入画），取最严重能避免漏判，
    代价是可能把偶发误检放大——所以必须靠融合决策里的交叉验证来兜。
    """
    if not items:
        return "平路", 0.0, "无可用帧"
    best = max(items, key=lambda r: (SEVERITY_RANK.get(r[0], 0), r[1]))
    why = best[2] if len(best) > 2 else "窗口内最严重判断"
    return best[0], float(best[1]), why


# ---------------------------------------------------------------- 自检
def synth_road(kind, seed=0, size=(320, 240)):
    """合成四种路面纹理，用于验证特征与基线是否真的能区分它们。

    注意：用**空间相关**的噪声（高斯模糊后的噪声）模拟沥青骨料纹理，
    而不是逐像素独立噪声。逐像素噪声不是沥青的样子，而且会让拉普拉斯方差
    被噪声主导，把"平滑路面"算成高频最高的一类——这是很容易踩的坑。
    """
    rng = np.random.default_rng(seed)
    h, w = size

    def aggregate(amp, ksize):
        n = rng.normal(0, amp, (h, w)).astype(np.float32)
        return cv2.GaussianBlur(n, (ksize, ksize), 0)

    base = np.full((h, w), 122.0, dtype=np.float32) + aggregate(2.0, 3)

    if kind == "平路":
        img = base
    elif kind == "粗糙路面":
        img = np.full((h, w), 122.0, dtype=np.float32) + aggregate(11.0, 5)
    elif kind == "裂缝":
        img = base.copy()
        for _ in range(3):
            x0 = int(rng.integers(30, w - 30))
            pts = np.array([[[x0 + int(7 * np.sin(y / 14.0)), y]]
                            for y in range(0, h, 8)], np.int32)
            cv2.polylines(img, [pts], False, 30.0, 2)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif kind == "坑洼":
        img = base.copy()
        # 必须落在 road_roi 裁剪后仍然完整的区域内，否则坑洼被裁掉一半、特征失真
        cx = int(rng.integers(80, 240))
        cy = int(rng.integers(120, 200))
        cv2.ellipse(img, (cx, cy), (42, 27), 15, 0, 360, 34.0, -1)     # 凹陷主体
        cv2.ellipse(img, (cx, cy), (42, 27), 15, 0, 360, 165.0, 3)     # 破损亮边
        img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        raise ValueError(kind)

    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def selftest():
    print("=" * 66)
    print("自检：合成四种路面纹理 -> 特征提取 -> 经典 CV 分类 -> 混淆矩阵")
    print("=" * 66)

    kinds = ["平路", "粗糙路面", "裂缝", "坑洼"]
    n_per = 40
    rows = []
    for k in kinds:
        for s in range(n_per):
            img = synth_road(k, seed=hash((k, s)) % 100000)
            f = features(img)
            lab, conf, why = classify_classical(f)
            rows.append({**f, "truth": k, "pred": lab, "conf": conf})

    print("[1/3] 各类特征均值（阈值应据此设定）：")
    keys = ["gray_std", "edge_density", "dark_blob_ratio", "dark_blob_px", "dark_elong",
            "crack_ratio", "crack_px", "crack_elong", "entropy"]
    print("      %-8s %8s %11s %14s %11s %9s %10s %9s %10s %8s"
          % ("真值", *[k[:9] for k in keys]))
    for k in kinds:
        sub = [r for r in rows if r["truth"] == k]
        vals = [np.mean([r[kk] for r in sub]) for kk in keys]
        print("      %-8s %8.2f %11.4f %14.4f %11d %9.2f %10.4f %9.1f %10.2f %8.3f"
              % (k, *vals))

    print("\n[2/3] 混淆矩阵（行为真值，列为预测）：")
    print("      %-10s %s" % ("", "".join("%-10s" % k for k in kinds)))
    total_ok = 0
    for k in kinds:
        cells = []
        for p in kinds:
            n = sum(1 for r in rows if r["truth"] == k and r["pred"] == p)
            cells.append(n)
        total_ok += cells[kinds.index(k)]
        print("      %-10s %s" % (k, "".join("%-10d" % c for c in cells)))
    acc = total_ok / len(rows)
    print("\n      整体准确率 %.3f（合成图上的上限参考，真实数据会明显更低）" % acc)
    assert acc >= 0.80, "经典 CV 基线准确率 %.3f 过低，特征或阈值需要调整" % acc

    print("\n[3/3] 窗口聚合策略验证（取窗口内最严重判断）：")
    cases = [
        [("平路", 0.7), ("平路", 0.8), ("坑洼", 0.6)],
        [("平路", 0.9), ("平路", 0.9)],
        [("粗糙路面", 0.7), ("裂缝", 0.6), ("平路", 0.8)],
    ]
    for c in cases:
        lab, conf, why = worst_of(c)
        print("      %-46s -> %s (%.2f)" % (str(c), lab, conf))

    print("-" * 66)
    print("自检通过。")
    print("说明：RDD2022 有中国摩托车视角子集（1,977 训练图 / 4,650 标注），")
    print("      用 ultralytics 在它上面微调后，加 --weights 即可切到 YOLO 后端。")


# ---------------------------------------------------------------- 处理采集包
def process_zip(zip_path, outdir, weights=None, conf=0.25):
    """处理采集端的 ZIP。

    两类画面分开处理：
      - 事件帧 frames/evN_tXXXX.jpg：由 IMU 触发，用于确认事件的病害类型
      - 周期扫描帧 frames/sweep_tXXXX.jpg：与事件无关的定时抓拍。
        裂缝不引起垂直颠簸、永远不会触发事件，只能靠扫描帧发现。
        扫描帧里检出裂缝/坑洼时，输出一条 id 为负数的"视觉独有"记录，
        交给 fusion.py 与 IMU 事件做空间去重后再定性。
    """
    os.makedirs(outdir, exist_ok=True)
    backend = try_yolo(weights)
    mode = "yolo(%s)" % os.path.basename(weights) if backend else "classical"

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        events, sweeps = [], []
        if "events.csv" in names:
            import pandas as pd
            events = pd.read_csv(io.BytesIO(z.read("events.csv"))).to_dict("records")
        if "sweeps.csv" in names:
            import pandas as pd
            sweeps = pd.read_csv(io.BytesIO(z.read("sweeps.csv"))).to_dict("records")

        ev_frames = sorted(n for n in names
                           if n.startswith("frames/ev") and n.lower().endswith((".jpg", ".png")))
        sw_frames = sorted(n for n in names
                           if n.startswith("frames/sweep") and n.lower().endswith((".jpg", ".png")))

        per_event = {}
        for n in ev_frames:
            base = os.path.basename(n)
            try:
                eid = int(base.split("_")[0].replace("ev", ""))
            except ValueError:
                continue
            img = read_image(z.read(n))
            f = features(img)
            if backend:
                lab, c, why = classify_yolo(backend(img, conf))
            else:
                lab, c, why = classify_classical(f)
            per_event.setdefault(eid, []).append({"file": n, "label": lab, "conf": c,
                                                  "why": why, "feat": f})

        sweep_rows = []
        sw_meta = {int(s["t_ms"]): s for s in sweeps if _int_ok(s.get("t_ms"))}
        for n in sw_frames:
            base = os.path.basename(n)
            try:
                t = int(base.replace("sweep_t", "").split(".")[0])
            except ValueError:
                continue
            img = read_image(z.read(n))
            f = features(img)
            if backend:
                lab, c, why = classify_yolo(backend(img, conf))
            else:
                lab, c, why = classify_classical(f)
            sweep_rows.append({"file": n, "t_ms": t, "label": lab, "conf": c,
                               "why": why, "meta": sw_meta.get(t, {}), "feat": f})

    out_events = []
    for ev in events:
        eid = int(ev.get("id", -1))
        fr = per_event.get(eid, [])
        lab, c, why = worst_of([(x["label"], x["conf"], x["why"]) for x in fr])
        driver = None
        if fr:
            driver = max(fr, key=lambda x: (SEVERITY_RANK.get(x["label"], 0), x["conf"]))["file"]
        row = dict(ev)
        row["source"] = "event"
        row["vision_label"] = lab
        row["vision_conf"] = round(float(c), 3)
        row["vision_why"] = why
        row["vision_frames"] = len(fr)
        row["vision_driver_frame"] = driver
        row["vision_all_labels"] = [x["label"] for x in fr]
        row["vision_detail"] = [{"file": x["file"], "label": x["label"], "conf": x["conf"]} for x in fr]
        out_events.append(row)

    # 扫描帧里的病害 → 视觉独有记录（负数 id）
    only = [r for r in sweep_rows if SEVERITY_RANK.get(r["label"], 0) >= 2]
    for k, r in enumerate(only, start=1):
        m = r["meta"]
        out_events.append({
            "id": -k, "source": "sweep", "t_ms": r["t_ms"],
            "lat": m.get("lat"), "lon": m.get("lon"), "speed_kmh": m.get("spd"),
            "vision_label": r["label"], "vision_conf": round(float(r["conf"]), 3),
            "vision_why": r["why"], "vision_frames": 1,
            "vision_driver_frame": r["file"], "vision_all_labels": [r["label"]],
            "note": "由周期扫描帧发现，IMU 未在此处检出颠簸事件",
        })

    payload = {
        "mode": mode,
        "source": os.path.basename(zip_path),
        "event_frames": len(ev_frames),
        "sweep_frames": len(sw_frames),
        "vision_only_findings": len(only),
        "events": out_events,
    }
    with open(os.path.join(outdir, "vision_events.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if out_events:
        import pandas as pd
        pd.DataFrame(out_events).drop(columns=["vision_detail"], errors="ignore").to_csv(
            os.path.join(outdir, "vision_events.csv"), index=False, encoding="utf-8-sig")

    ev_rows = [r for r in out_events if r["source"] == "event"]
    print("后端 %s｜事件帧 %d 张 / 事件 %d 个｜扫描帧 %d 张 / 视觉独有病害 %d 处"
          % (mode, len(ev_frames), len(ev_rows), len(sw_frames), len(only)))
    print("事件视觉判定分布：%s"
          % dict(Counter(r["vision_label"] for r in ev_rows)))
    if only:
        print("扫描帧发现的病害：%s" % dict(Counter(r["label"] for r in only)))
    print("输出：%s/{vision_events.json, vision_events.csv}" % outdir)
    return out_events


def _int_ok(v):
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description="路面体检 视觉侧 v0.1")
    ap.add_argument("--zip", help="采集端导出的 ZIP")
    ap.add_argument("--image", help="单张图片，只打印特征与判定")
    ap.add_argument("--weights", help="YOLO 权重路径（在 RDD2022 上微调得到）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--outdir", default="vis")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.image:
        img = read_image(args.image)
        f = features(img)
        print(json.dumps(f, ensure_ascii=False, indent=2))
        backend = try_yolo(args.weights)
        print("判定：%s" % (classify_yolo(backend(img, args.conf)) if backend
                            else classify_classical(f)))
        return
    if args.zip:
        process_zip(args.zip, args.outdir, args.weights, args.conf)
        return
    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
