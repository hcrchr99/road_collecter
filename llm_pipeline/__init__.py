# -*- coding: utf-8 -*-
"""RoadCheck LLM 判读管线 · L0 特征提取层（M-L1）

确定性代码，无任何 LLM 调用：
  zip 数据包 → 解包 → 关键帧选取 → 帧自检 → IMU 派生特征块 → 判读任务清单

设计依据：RoadCheck_LLM判读管线_开发文档.md §2.1 / §4
"""
