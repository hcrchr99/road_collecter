# -*- coding: utf-8 -*-
"""统一模型调用封装。

GLM-4.6V（bigmodel.cn）、Qwen3-VL（DashScope）、Qwen3-VL-8B（本地 ollama）
三者都提供 OpenAI 兼容接口 ⇒ 一个客户端类 + config.MODELS 注册表即可，
不需要每个厂商一个文件。

JSON 合规硬要求（开发文档 §5）：
  - 开启 response_format=json_object（结构化输出模式）
  - 解析失败重试一次，仍失败标 parse_failed 显式丢弃，不硬塞
  - 限流/超时：指数退避重试（网络类错误与解析失败分开计）
"""
import base64
import json
import os
import time

from .. import config


class ParseError(Exception):
    pass


class ModelClient:
    def __init__(self, model_key: str):
        cfg = config.MODELS[model_key]
        self.model_key = model_key
        self.model = cfg["model"]
        self.base_url = cfg["base_url"]
        self.role = cfg["role"]
        self.extra_body = cfg.get("extra_body") or {}
        self.timeout = int(cfg.get("timeout", 120))
        key = None
        if cfg["api_key_env"]:
            key = os.environ.get(cfg["api_key_env"])
            if not key:
                raise RuntimeError(
                    "缺少 API key：请设置环境变量 %s（模型 %s）" % (cfg["api_key_env"], model_key))
        # 延迟导入，L0 阶段无需安装 openai
        from openai import OpenAI
        self.client = OpenAI(base_url=cfg["base_url"], api_key=key or "local")

    # ------------------------------------------------------------ 消息构造
    @staticmethod
    def build_messages(system_prompt, user_text, frame_jpegs):
        """frame_jpegs: [(role_tag, bytes)]，按 pre/impact/post 顺序。"""
        content = [{"type": "text", "text": user_text}]
        for _tag, data in frame_jpegs:
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,%s" % b64}})
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": content}]

    # ------------------------------------------------------------ 调用
    def call_json(self, messages, max_retries_net=3, timeout=None):
        """调用并解析 JSON。网络错误指数退避；解析失败仅重试一次。

        返回 (dict|None, meta)；
        meta: {"status": "ok|parse_failed|net_failed", "attempts": n, "error": str}
        timeout 缺省用 config 里该模型的配置。
        """
        if timeout is None:
            timeout = self.timeout
        parse_attempts = 0
        net_attempt = 0
        last_err = ""
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=timeout,
                    extra_body=self.extra_body or None,
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:  # 网络/限流/请求错误
                net_attempt += 1
                last_err = "%s: %s" % (type(e).__name__, e)
                # 4xx 属请求级永久错误（参数/鉴权/格式），重试无意义，立即返回
                name = type(e).__name__
                if name in ("BadRequestError", "AuthenticationError",
                            "PermissionDeniedError", "NotFoundError",
                            "UnprocessableEntityError"):
                    return None, {"status": "net_failed", "attempts": net_attempt,
                                  "error": last_err}
                if net_attempt > max_retries_net:
                    return None, {"status": "net_failed", "attempts": net_attempt,
                                  "error": last_err}
                time.sleep(2 ** net_attempt)  # 指数退避：2/4/8s
                continue
            # 解析
            try:
                obj = json.loads(text)
                if not isinstance(obj, dict) or "visual_label" not in obj:
                    raise ValueError("缺少 visual_label 字段")
                return obj, {"status": "ok", "attempts": net_attempt + parse_attempts + 1,
                             "error": ""}
            except Exception as e:
                parse_attempts += 1
                last_err = "%s: %s" % (type(e).__name__, e)
                if parse_attempts >= 2:
                    return None, {"status": "parse_failed", "attempts": parse_attempts,
                                  "error": last_err, "raw": text[:500]}


def get_client(model_key: str) -> ModelClient:
    return ModelClient(model_key)
