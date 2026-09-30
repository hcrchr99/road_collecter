#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""忠实回放采集端的 IMU 事件检测算法，定位"为什么没有检出/多检出事件"。

★ 2026-09-26 重写：检测实现统一收敛到 tools/qc.py（按 meta.version 分叉三套口径，
  已在 9 个真实包上验证：v0.8 包 388/388 精确复现，v0.6 包 152/152 精确复现）。
  本脚本不再自带公式，只做诊断呈现——避免出现与采集端漂移的第三份实现
  （旧版停在 quantile 门限 + 600ms 不应期，对 v0.8/v0.9 包会算出完全错误的数字）。

当前口径（采集端 v0.9，与 v0.8 检测部分一致）：
  门限   = max(1.2, 4 × 最近 1 秒 |smooth| 的 RMS)（本底下限 0.02）
  开始   = smooth > 门限 且 距上次事件结束 > 盲区时间（1.2 m ÷ 车速，钳制 250~1500 ms；
           速度不可用取 800 ms）。速度不再作为开启条件，低速事件只打 low_speed 标记。
  结束   = 任一帧 smooth < 门限×0.5 且时长 > 60 ms，或 3000 ms
  ≤v0.7 旧包自动切换到历史口径（分位门限 / 固定不应期 / 速度门控丢弃）。

用法：
    python replay_detect.py <pack.zip>
    python replay_detect.py            # 用环境变量 ROADCHECK_ZIP
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

# Windows 上重定向 stdout 到文件时默认走 GBK(cp936)，会把中文报告写成乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tools.qc import (  # noqa: E402
    detect_replay, ema_smooth, rolling_rms, threshold_series, ver_tuple,
)
from llm_pipeline import unpack  # noqa: E402

SRC = (sys.argv[1] if len(sys.argv) > 1
       else os.environ.get("ROADCHECK_ZIP", ""))

if not SRC or not Path(SRC).exists():
    print("用法: python replay_detect.py <pack.zip>  （或设 ROADCHECK_ZIP）")
    sys.exit(1)

with unpack.Pack(SRC) as pack:
    meta = pack.read_meta()
    events_csv = pack.read_events()
    samples = pack.read_samples()
    pack_id = pack.pack_id

ver = ver_tuple(meta.get("version"))
mode = (meta.get("collection") or {}).get("mode", "")

t_ms = np.array([int(float(r["t_ms"])) for r in samples], dtype=np.float64)
hp = np.array([float(r["hp"]) for r in samples], dtype=float)
spd = np.array([float(r.get("speed_ms") or 0) for r in samples], dtype=float)

smooth = ema_smooth(t_ms, hp)
thr, noise_kind = threshold_series(t_ms, smooth, spd, ver)
events, missed = detect_replay(t_ms, smooth, thr, spd, ver, mode)

rms1 = rolling_rms(t_ms, smooth, 1000.0)
over = float(np.mean(np.abs(smooth) > thr))

print("=" * 86)
print("采集包 %s（采集端 v%s）" % (pack_id, meta.get("version")))
print("=" * 86)
print("样本 %d / 时长 %.1f s / 采样率 %.1f Hz | 噪声口径 %s"
      % (len(samples), t_ms[-1] / 1000.0,
         1000.0 / np.median(np.diff(t_ms)), noise_kind))
print()
print("【信号】smooth 峰 %.2f | 门限 中位 %.2f / 最大 %.2f | 超门限样本 %.2f%%"
      % (float(np.abs(smooth).max()), float(np.median(thr)), float(thr.max()),
         over * 100))
print("       1s-RMS 中位 %.3f（本底量级，v0.8 实测健康值 ~0.6%% 超门限）"
      % float(np.median(rms1)))
print()

n_meta = int(meta.get("event_count") or 0)
print("【回放】检出 %d 个事件 | meta 记录 %d | events.csv %d 行 %s"
      % (len(events), n_meta, len(events_csv),
         "✅ 一致" if len(events) == n_meta == len(events_csv) else "⚠ 不一致，先怀疑复现口径"))

if events:
    iv = np.diff([e[0] for e in events])
    crest = [e[2] / max(rms1[e[3]], 1e-9) for e in events]
    print("       min(iv) %.0f ms | CV %.2f | crest 中位 %.2f | crest≥4 占比 %.1f%%"
          % (float(iv.min()) if len(iv) else 0,
             float(np.std(iv) / np.mean(iv)) if len(iv) else 0,
             float(np.median(crest)),
             100.0 * float(np.mean([c >= 4 for c in crest]))))

for k, label in (("speed_gate", "速度门控丢弃"), ("refractory", "不应期吞掉"),
                 ("in_event", "事件期间吞掉")):
    m = missed.get(k, [])
    if m:
        print("       %s：%d 个样本" % (label, len(m)))

if ver < (0, 7):
    print()
    print("【注意】本包为 v%s（旧口径）。历史结论：旧公式 max(1.2, 4×20%分位) ≡ 1.013σ，"
          % meta.get("version"))
    print("       事件数不含病害信息（零假设检验已在 6 个旧包复证）。仅存档，不进判读。")
else:
    print()
    print("【诊断】门限健康。若事件数异常，优先核对：")
    print("       1) meta.threshold_stats.floor_active_ratio 是否接近 1（自适应项失效）")
    print("       2) 采样中断（手指按屏会让 devicemotion 停摆）")
    print("       3) 光照（事件触发抓帧不可再生，判据见 tools/qc.py 的 vision 组）")

print()
print("【逐秒概况】（每 5 秒）")
print("   %6s %10s %10s %9s %10s" % ("t(s)", "门限", "smooth峰", "超门限%", "速度km/h"))
step = 5
for a in range(0, int(t_ms[-1] / 1000) + 1, step):
    m = (t_ms >= a * 1000) & (t_ms < (a + step) * 1000)
    if not m.any():
        continue
    print("   %6d %10.2f %10.2f %8.1f%% %10.1f"
          % (a, float(thr[m].mean()), float(smooth[m].max()),
             100.0 * float(np.mean(np.abs(smooth[m]) > thr[m])),
             float(spd[m].mean()) * 3.6))
