# -*- coding: utf-8 -*-
"""L0 特征提取主流程：包 → 逐事件特征块 + 判读任务清单 + 核对清单。

产出（默认落 _repo/out/llm_pipeline/l0/）：
  features.jsonl  每事件一条：帧清单/关键帧/帧自检/IMU 特征/特征文本块
  checklist.md    与 meta.json 的数字核对清单（M-L1 验收）
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, frame_check, imu_features, keyframes, unpack


def _samples_arrays(samples):
    """samples.csv 行 → (t_ms, hp, vz) numpy 数组。"""
    t = np.array([int(float(r["t_ms"])) for r in samples], dtype=np.int64)
    hp = np.array([float(r["hp"]) for r in samples], dtype=float)
    vz = np.array([float(r["vz"]) for r in samples], dtype=float)
    return t, hp, vz


def extract(pack: "unpack.Pack", progress=None) -> dict:
    """对包做全量 L0 提取。返回 result dict（含 records/checklist/summary）。"""
    meta = pack.read_meta()
    events = pack.read_events()
    samples = pack.read_samples()
    frames_by_ev = pack.list_event_frames()

    t_arr, hp, vz = _samples_arrays(samples)

    records = []
    for ev in events:
        eid = int(ev["id"])
        t0 = int(float(ev["t_ms"]))
        kf = keyframes.select_keyframes(t0, frames_by_ev.get(eid, []))

        # 帧自检：只检关键帧（不全量解码）
        kf_checked = []
        for k in kf:
            qc = frame_check.self_check(pack.read_frame(k["zip_path"]))
            kf_checked.append({**k, "quality": qc})

        blocked = all(k["quality"]["blocked"] for k in kf_checked)
        feat = imu_features.compute_event_features(ev, t_arr, hp, vz)
        block = imu_features.build_feature_block(ev, feat)

        rec = {
            "event_id": eid,
            "t0_ms": t0,
            "duration_ms": feat["duration_ms"],
            "frames_total": int(ev.get("frames") or len(frames_by_ev.get(eid, []))),
            "keyframes": kf_checked,
            "blocked": blocked,
            "imu_features": feat,
            "imu_block": block,
            "auto_label": feat["auto_label"],
            "low_speed": feat["low_speed"],
            # blocked 事件不进入 L1/L2，直接终判
            "visual_label_preset": config.BLOCKED_LABEL if blocked else None,
        }
        records.append(rec)
        if progress:
            progress(eid)

    checklist = build_checklist(pack, meta, events, samples, frames_by_ev, records)
    return {"pack_id": pack.pack_id, "meta": meta, "records": records,
            "checklist": checklist}


def build_checklist(pack, meta, events, samples, frames_by_ev, records) -> dict:
    """M-L1 验收：与 meta.json 逐项核对。"""
    ev_frames_total = sum(len(v) for v in frames_by_ev.values())
    n_kf = [len([k for k in r["keyframes"]]) for r in records]
    kf3 = sum(1 for x in n_kf if x == 3)
    gaps_ab = [k["gap_ms"] for r in records for k in r["keyframes"]
               if k["role"] in ("pre", "impact") and k["gap_ms"] is not None]
    blocked_cnt = sum(1 for r in records if r["blocked"])
    suspects = Counter()
    for r in records:
        for k in r["keyframes"]:
            if k["quality"]["suspect"]:
                suspects[k["role"]] += 1

    fq = meta.get("frame_quality", {})
    checks = [
        ("事件数", len(events), meta.get("event_count")),
        ("事件帧总数", ev_frames_total, None),  # meta 不含该数，仅记录
        ("samples 行数", len(samples), meta.get("sample_count")),
        ("扫描帧数", len(pack.list_sweep_frames()), meta.get("sweep_count")),
        ("包版本", meta.get("version"), meta.get("version")),
        ("3 关键帧事件数", kf3, len(events)),
        ("低速事件数", sum(1 for r in records if r["low_speed"]),
         meta.get("low_speed_event_count")),
        ("blocked 事件数(L0 全部关键帧不可判)", blocked_cnt, None),
        ("关键帧 pre/impact 命中间隔中位 ms",
         int(np.median(gaps_ab)) if gaps_ab else None, "<=30 (计划验收线)"),
    ]
    return {
        "checks": [{"item": a, "value": b, "expected": c} for a, b, c in checks],
        "keyframe_suspect_by_role": dict(suspects),
        "meta_frame_quality": {k: fq.get(k) for k in
                               ("checked", "suspect", "suspect_ratio", "usable")},
        "blocked_events": [r["event_id"] for r in records if r["blocked"]],
    }


def run(pack_path=None, out_dir=None):
    pack_path = Path(pack_path or config.DEFAULT_PACK)
    out_dir = Path(out_dir) if out_dir else config.OUTPUT_DIR / "l0"
    out_dir.mkdir(parents=True, exist_ok=True)

    with unpack.Pack(pack_path) as pack:
        result = extract(pack)

    # features.jsonl
    feats_path = out_dir / "features.jsonl"
    with open(feats_path, "w", encoding="utf-8") as f:
        for r in result["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # checklist.md
    cl = result["checklist"]
    lines = [
        "# M-L1 核对清单 · %s" % result["pack_id"],
        "",
        "| 核对项 | 本包实测 | 期望/参照 |",
        "|---|---|---|",
    ]
    for c in cl["checks"]:
        lines.append("| %s | %s | %s |" % (c["item"], c["value"],
                                           c["expected"] if c["expected"] is not None else "—"))
    lines += [
        "",
        "关键帧自检 suspect 分布（按角色）：%s" % cl["keyframe_suspect_by_role"],
        "meta.frame_quality：%s" % cl["meta_frame_quality"],
        "blocked 事件：%s" % (cl["blocked_events"] or "无"),
        "",
        "> 注：证据强度分档阈值为占位值（TODO-calibration），见 imu_features.py。",
    ]
    cl_path = out_dir / "checklist.md"
    with open(cl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return result, feats_path, cl_path
