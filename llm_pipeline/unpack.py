# -*- coding: utf-8 -*-
"""解包：zipfile 直读内存，不做落盘解压。

- 采集端导出 STORE 模式 zip，zipfile 原生可读；zip 内路径恒为正斜杠
- 所有 CSV 带 BOM（采集端 JS 导出），一律 utf-8-sig
"""
import csv
import io
import json
import re
import zipfile
from pathlib import Path

# 帧文件命名约定（与 tools/label_tool.html L323-324 一致）：
#   frames/ev{id}_t{ms}.jpg   事件触发帧（frames 列是帧数，不是文件名！）
#   frames/sweep_t{ms}.jpg    周期扫描帧
RE_EVENT_FRAME = re.compile(r"^frames/ev(\d+)_t(\d+)\.jpg$")
RE_SWEEP_FRAME = re.compile(r"^frames/sweep_t(\d+)\.jpg$")


def pack_id_from_path(path) -> str:
    """roadcheck_tang_20260924_1229.zip → roadcheck_tang_20260924_1229"""
    return Path(path).stem


class Pack:
    """一个采集包的只读视图。用 with 语句使用。"""

    def __init__(self, path):
        self.path = Path(path)
        self.pack_id = pack_id_from_path(path)
        self.z = zipfile.ZipFile(path)

    def close(self):
        self.z.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------ 结构化数据
    def read_meta(self) -> dict:
        return json.loads(self.z.read("meta.json").decode("utf-8-sig"))

    def read_csv(self, name: str):
        """返回 list[dict]（字符串值）。"""
        text = self.z.read(name).decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))

    def read_events(self):
        return self.read_csv("events.csv")

    def read_samples(self):
        return self.read_csv("samples.csv")

    # ------------------------------------------------------ 帧清单
    def list_event_frames(self) -> dict:
        """{event_id(int): [(t_ms(int), zip内路径), ...]}，t 升序。"""
        out = {}
        for name in self.z.namelist():
            m = RE_EVENT_FRAME.match(name)
            if m:
                out.setdefault(int(m.group(1)), []).append((int(m.group(2)), name))
        for eid in out:
            out[eid].sort()
        return out

    def list_sweep_frames(self):
        """[(t_ms, zip内路径)]，t 升序。"""
        out = []
        for name in self.z.namelist():
            m = RE_SWEEP_FRAME.match(name)
            if m:
                out.append((int(m.group(1)), name))
        out.sort()
        return out

    def read_frame(self, zip_path: str) -> bytes:
        return self.z.read(zip_path)
