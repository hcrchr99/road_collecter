# -*- coding: utf-8 -*-
"""
忠实回放采集端的 IMU 事件检测算法，定位"为什么没有检出事件"。

逐行复刻 collector.html 的 detect() 逻辑：
  门限 = max(1.2, 4 × quantile(|smooth|, 20%))   窗口为「最近 3 秒」（尾部窗口，非居中）
  开始事件条件 = 速度 >= 8 km/h  且  smooth > 门限  且  距上次事件结束 > 600 ms
  结束事件条件 = smooth < 门限 × 0.5（至少 60 ms）  或  持续超过 3000 ms
"""
import json
import zipfile

import numpy as np
import pandas as pd

SRC = r"C:\Users\Admin\Downloads\roadcheck_20260918_2127.zip"

# 与 collector.html 的 CFG 完全一致
TAU_SMOOTH = 0.010
NOISE_WIN_S = 3.0
NOISE_Q = 0.20
NOISE_K = 4.0
MIN_THRESHOLD = 1.2
RELEASE_RATIO = 0.5
REFRACTORY_MS = 600
MAX_EVENT_MS = 3000
MIN_SPEED_MS = 8 / 3.6

z = zipfile.ZipFile(SRC)
s = pd.read_csv(z.open("samples.csv"))
t_ms = s["t_ms"].to_numpy(dtype=float)
hp = s["hp"].to_numpy(dtype=float)
spd = s["speed_ms"].to_numpy(dtype=float)

# 复刻 S.smooth：对 hp 做 tau=10ms 的指数平滑（采集端检测用的就是它）
dt = np.diff(t_ms, prepend=t_ms[0]) / 1000.0
dt[0] = 1 / 60
dt = np.clip(dt, 1e-4, 0.2)
smooth = np.zeros_like(hp)
acc = 0.0
for i in range(len(hp)):
    a = np.exp(-dt[i] / TAU_SMOOTH)
    acc = acc * a + hp[i] * (1 - a)
    smooth[i] = acc

# 尾部 3 秒窗口的分位数门限
thr = np.zeros_like(hp)
win = []
for i in range(len(t_ms)):
    win.append(abs(smooth[i]))
    while win and t_ms[i] - t_ms[i - len(win) + 1] > NOISE_WIN_S * 1000 and len(win) > 1:
        win.pop(0)
    if len(win) > 1 and t_ms[i] - t_ms[i - len(win) + 1] > NOISE_WIN_S * 1000:
        win.pop(0)
    q = np.quantile(win, NOISE_Q)
    thr[i] = max(MIN_THRESHOLD, NOISE_K * max(q, 0.02))

# 状态机（跑两遍：带速度门控 / 不带速度门控）
def run(gate):
    events = []
    in_ev, t0, peak, last_end = False, 0.0, 0.0, -1e9
    for i in range(len(t_ms)):
        v, t, sp = smooth[i], t_ms[i], spd[i]
        moving = (sp >= MIN_SPEED_MS) if gate else True
        if not in_ev:
            if moving and v > thr[i] and (t - last_end) > REFRACTORY_MS:
                in_ev, t0, peak = True, t, v
        else:
            peak = max(peak, v)
            if (v < thr[i] * RELEASE_RATIO and t - t0 > 60) or (t - t0) > MAX_EVENT_MS:
                events.append((t0 / 1000.0, (t - t0), peak, thr[i]))
                in_ev, last_end = False, t
    return events


print("=" * 86)
print("采集包 %s" % SRC.split("\\")[-1])
print("=" * 86)
print("样本 %d 个 / 时长 %.1f s / 采样率 %.1f Hz"
      % (len(s), t_ms[-1] / 1000.0, 1000.0 / np.median(np.diff(t_ms))))
print()
print("【信号强度】")
print("  smooth   min %8.3f  max %8.3f  std %7.3f" % (smooth.min(), smooth.max(), smooth.std()))
print("  门限     min %8.3f  max %8.3f  mean %7.3f" % (thr.min(), thr.max(), thr.mean()))
print("  超门限样本数  %d / %d  (%.1f%%)"
      % (int(np.sum(smooth > thr)), len(smooth), 100.0 * np.mean(smooth > thr)))
print()
print("【速度门控】门限 = %.3f m/s = %.1f km/h" % (MIN_SPEED_MS, MIN_SPEED_MS * 3.6))
print("  速度 min %.3f  中位 %.3f  max %.3f m/s" % (spd.min(), np.median(spd), spd.max()))
print("  通过门控的样本数  %d / %d  (%.1f%%)"
      % (int(np.sum(spd >= MIN_SPEED_MS)), len(spd), 100.0 * np.mean(spd >= MIN_SPEED_MS)))
print()

ev_gate = run(True)
ev_nogate = run(False)

print("=" * 86)
print("【回放结果】")
print("  带速度门控（＝你实际跑出来的）：%d 个事件" % len(ev_gate))
print("  去掉速度门控（仅看信号）：      %d 个事件" % len(ev_nogate))
print()
if ev_nogate:
    print("  去掉门控后本应检出的事件：")
    print("     %10s %10s %10s %12s" % ("起始(s)", "时长(ms)", "峰值", "当时门限"))
    for t0, d, pk, th in ev_nogate:
        print("     %10.2f %10.0f %10.2f %12.2f" % (t0, d, pk, th))
print()
print("=" * 86)
print("【诊断】")
if len(ev_gate) == 0 and len(ev_nogate) > 0:
    print("  根因：速度门控。信号本身完全够（软件层峰值远超门限），")
    print("        但全程速度读数为 0，moving 恒为 false，永远无法开启事件。")
    print("  性质：规则问题（门控阈值对静止/低速测试场景不适用），不是程序 bug。")
elif len(ev_gate) == 0 and len(ev_nogate) == 0:
    print("  根因：门限。信号峰值未超过自适应门限，需要检查噪声底估计。")
else:
    print("  速度门控未造成完全抑制，需进一步分析。")

# 逐秒看门限与速度
print()
print("【逐秒概况】")
print("   %6s %10s %10s %10s %8s" % ("t(s)", "门限", "smooth峰", "超门限%", "速度km/h"))
for a in range(0, int(t_ms[-1] / 1000) + 1):
    m = (t_ms >= a * 1000) & (t_ms < (a + 1) * 1000)
    if not m.any():
        continue
    print("   %6d %10.2f %10.2f %9.0f%% %8.1f"
          % (a, thr[m].mean(), smooth[m].max(),
             100.0 * np.mean(smooth[m] > thr[m]), spd[m].mean() * 3.6))
