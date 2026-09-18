# -*- coding: utf-8 -*-
"""生成一份与采集端导出格式完全一致的 CSV，用于验证 analyze.py 的真实输入路径。"""
import numpy as np
import pandas as pd

from analyze import synth

raw, truth = synth(seed=11, dur=90.0, spacing=8.0)
rng = np.random.default_rng(3)
n = len(raw)

out = pd.DataFrame({
    "t_ms": raw["t"].round(0).astype(int),
    "ax": raw["ax"].round(4),
    "ay": raw["ay"].round(4),
    "az": raw["az"].round(4),
    "gx": (0.5 * rng.standard_normal(n)).round(4),
    "gy": (0.5 * rng.standard_normal(n)).round(4),
    "gz": (0.5 * rng.standard_normal(n)).round(4),
    "lat": raw["lat"].round(7),
    "lon": raw["lon"].round(7),
    "speed_ms": raw["speed"].round(3),
    "hp": np.abs(raw["az"] - 9.81).round(4),
})
out.to_csv("sample_run.csv", index=False, encoding="utf-8-sig")
print("已生成 sample_run.csv：%d 行 × %d 列，注入真值事件 %d 个" % (len(out), len(out.columns), len(truth)))
print("列名：%s" % ", ".join(out.columns))
