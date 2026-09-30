# -*- coding: utf-8 -*-
"""判读智能体（规划 §3.2 阶段 B）。

内核 = 取证循环（loop）+ 工具集（tools）+ 自检规则（selfcheck）。
铁律：
- 自检规则 R1–R5 写死在代码里，不允许模型自由发挥；
- 置信度由证据推导，不由模型声明；
- 输出契约 roadcheck.vision_review.v0 不变，LLM 结论恒为 source: llm；
- 单事件最多 3 轮取证、≤8 次模型调用，超限一律 ESCALATE（复核队列只增不减）。
"""
