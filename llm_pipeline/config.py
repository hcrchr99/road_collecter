# -*- coding: utf-8 -*-
"""路径、模型注册表、环境变量名。所有可变配置集中在此。"""
import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
# 工作区根（_repo 的上一级）
WORKSPACE = Path(__file__).resolve().parent.parent.parent
REPO = WORKSPACE / "_repo"

# 采集包目录（工作区惯例：road_check/ 放所有 zip）
PACKS_DIR = WORKSPACE / "road_check"
# 标注数据目录
LABELS_DIR = REPO / "labels"
# 管线输出目录
OUTPUT_DIR = REPO / "out" / "llm_pipeline"
# 报告目录
REPORTS_DIR = REPO / "reports"

# 默认实验包（M-L1 / M-L2 评测包）
DEFAULT_PACK = PACKS_DIR / "roadcheck_tang_20260924_1229.zip"
# 第一黄金集：130 条全量人工标注（tang，2026-09-25）
DEFAULT_GOLD = LABELS_DIR / "roadcheck_20260924_1229.labels.part-tang.csv"

# ---------------------------------------------------------------- API key 环境变量
ENV_ZHIPU = "ZHIPU_API_KEY"       # 智谱 bigmodel.cn（GLM-4.6V）
ENV_DASHSCOPE = "DASHSCOPE_API_KEY"  # 阿里百炼（Qwen3-VL 系列）
ENV_DEEPSEEK = "DEEPSEEK_API_KEY"    # DeepSeek（V4.1 Flash，原生多模态）
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# ---------------------------------------------------------------- 模型注册表（§7.3 对照实验）
# role: L2=主力判读 / L1=粗筛门卫；provider 决定用哪个 client
# timeout: 单次 API 调用超时（秒）；本地视觉模型生成慢，放宽到 600
# 价格快照 2026-09-25（元/M tokens，以官网当期为准）：
#   glm-5.3-flash  0.8/2.8（原生多模态，320B-A18B，1M 上下文；9-9 前促销 0.4/1.4 已结束）
#   glm-5.3-flashx 2/7（同款高速版）
#   glm-4.6v       约 $0.3/M 输入（开发文档 §3 快照）
#   glm-4.6v-flash 免费（视觉模型，官网 FAQ：输入/输出/缓存全免）
#   deepseek-v4-1-flash  $0.15-0.30/M 输入、$0.60-1.20/M 输出（峰谷分时，输入重）；
#     原生多模态（图像+文本一体），1M 上下文；思考模式默认开启（temperature 被忽略），
#     hosted API 用 reasoning_effort=low 控制思考预算
MODELS = {
    "glm-5.3-flash": {
        "provider": "openai_compat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": ENV_ZHIPU,
        "model": "glm-5.3-flash",
        "role": "L2",
        "concurrency": 4,
        "timeout": 180,
        # glm-5.3-flash 始终思考，不支持 disabled（错误码 1210），只接受 low/high/max。
        # 判读是结构化分类任务，用 low 档省输出成本（实测 thinking off 报 400）
        "extra_body": {"thinking": {"type": "enabled", "effort": "low"}},
    },
    "glm-4.6v-flash": {
        "provider": "openai_compat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": ENV_ZHIPU,
        "model": "glm-4.6v-flash",
        "role": "L1/免费基线",
        "concurrency": 2,  # 免费档限流更严，压低并发
        # 4.6 系未实测 thinking 参数兼容性，用模型默认，避免 400
    },
    "glm-4.6v": {
        "provider": "openai_compat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": ENV_ZHIPU,
        "model": "glm-4.6v",
        "role": "L2",
        "concurrency": 4,
        "timeout": 180,
    },
    "qwen3-vl-32b": {
        "provider": "openai_compat",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": ENV_DASHSCOPE,
        "model": "qwen3-vl-32b",
        "role": "L2",
        "concurrency": 4,
        "timeout": 180,
    },
    "deepseek-v4.1f": {
        # 正式模型名 deepseek-flash（= V4.1 Flash，原生多模态；API 报错提示
        # 有效名为 deepseek-flash / deepseek-v4-pro，deepseek-v4-1-flash 不被接受）。
        # 思考模式默认开启：temperature 被忽略，用 reasoning_effort=low 控制输出成本
        "provider": "openai_compat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": ENV_DEEPSEEK,
        "model": "deepseek-flash",
        "role": "L2/跨厂商对照",
        "concurrency": 4,
        "timeout": 180,
        "extra_body": {"reasoning_effort": "low"},
    },
    "qwen3-vl-8b": {
        # 2026-09-25 暂停：本地推理 ~2-5 min/事件（串行 130 条 >4h），云端正选已够用。
        # 保留注册项与基础设施（ollama + 派生模型 qwen3-vl-8b-16k），
        # 将来要低成本冒烟或断网调试时可直接恢复
        "provider": "openai_compat",
        "base_url": OLLAMA_BASE_URL,
        "api_key_env": None,  # 本地 ollama 无需 key
        "concurrency": 1,     # 单 GPU 串行
        # 用派生模型 qwen3-vl-8b-16k（ollama_Modelfile.qwen，num_ctx=16384）：
        # 默认 4096 窗口下 3 张关键帧视觉 token 占满上下文，生成只剩 ~18 token → 空响应
        "model": "qwen3-vl-8b-16k",
        "role": "L1/L2 对照（暂停）",
        "timeout": 600,       # 本地视觉生成慢，放宽超时
    },
}

# ---------------------------------------------------------------- schema v1.2 词表（labels/schema.md 为唯一真源）
# v1.2 新增「标线」：热熔标线边缘隆起压过产生真实振动但非病害；
# 只能由人工/LLM 视觉判读产生，autoType（纯 IMU）值域不含它
LABELS = [
    "坑洼", "减速带", "井盖", "设计接缝", "粗糙路面",
    "纵向裂缝", "横向裂缝", "龟裂", "平路", "遮挡/无法判定",
    "标线",
]
BLOCKED_LABEL = "遮挡/无法判定"

# 人工仲裁修正层（schema §4：结论写进第三份文件，不改原始标注）
DEFAULT_GOLD_ADJUDICATED = LABELS_DIR / "roadcheck_20260924_1229.labels.adjudicated.csv"
