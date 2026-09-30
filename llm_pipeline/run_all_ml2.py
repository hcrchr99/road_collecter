# -*- coding: utf-8 -*-
"""M-L2 全量运行器：顺序跑指定模型 + 最终评测。

设计为**独立进程**运行（双击 bat / Start-Process），
断点续跑由 judge.judge_pack 自带（reviews.jsonl 已有决定性结论的事件自动跳过）。

输出同时写控制台与 run_<模型>.log（控制台窗口关了日志仍在，便于排障）。

用法：
  python -m llm_pipeline.run_all_ml2 glm-5.3-flash glm-4.6v-flash glm-4.6v
  python -m llm_pipeline.run_all_ml2 --tag v2 --prompt prompts/judge_system_v2.md glm-4.6v qwen3-vl-8b
"""
import json
import os
import sys
import time

from . import config, evaluate, judge, unpack


class Tee:
    """stdout 同时写控制台与日志文件。"""

    def __init__(self, stream, log_path):
        self.stream = stream
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass  # 控制台编码问题不影响日志
        self.log.write(s)
        self.log.flush()

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.log.flush()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="模型 key（或 key_v2 这类带 tag 的目录名）")
    ap.add_argument("--tag", default=None, help="输出目录后缀，隔离不同提示词版本")
    ap.add_argument("--prompt", default=None, help="系统提示词文件路径")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None,
                    help="每事件喂给模型的帧数（>3 从包内全部帧均匀抽样）")
    ap.add_argument("--preprocess", default="none", choices=["none", "clahe"],
                    help="帧预处理：clahe = 自适应直方图均衡")
    args = ap.parse_args()

    # 日志落盘（每个运行器进程一个日志文件）
    log_name = "run_%s.log" % ("_".join(args.models) + ("_" + args.tag if args.tag else ""))
    log_path = config.OUTPUT_DIR / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.stdout, log_path)

    print("==== run_all_ml2 %s tag=%s prompt=%s ===="
          % (time.strftime("%Y-%m-%d %H:%M:%S"), args.tag, args.prompt), flush=True)

    features_path = config.OUTPUT_DIR / "l0" / "features.jsonl"
    records = [json.loads(l) for l in
               open(features_path, encoding="utf-8").read().splitlines() if l.strip()]
    print("records=%d models=%s" % (len(records), args.models), flush=True)

    # 模型 key 与输出目录名分离；目录名 = base + tag（评测读同一目录）
    run_keys = []
    for m in args.models:
        base = m if m in config.MODELS else m.rsplit("_", 1)[0]
        if base not in config.MODELS:
            print("跳过 %s：不在模型注册表" % m, flush=True)
            continue
        dirname = base + ("_" + args.tag if args.tag else "")
        run_keys.append((base, dirname))

    with unpack.Pack(config.DEFAULT_PACK) as pack:
        for base, dirname in run_keys:
            cfg = config.MODELS[base]
            if cfg.get("api_key_env") and not os.environ.get(cfg["api_key_env"]):
                print("跳过 %s：未设置 %s" % (base, cfg["api_key_env"]), flush=True)
                continue
            print("== START %s (dir=%s) %s ==" % (base, dirname,
                                                  time.strftime("%H:%M:%S")), flush=True)
            stats = judge.judge_pack(base, pack, records, limit=args.limit,
                                     out_tag=args.tag, prompt_file=args.prompt,
                                     frames_n=args.frames, preprocess=args.preprocess)
            print("== DONE %s %s ==" % (base, time.strftime("%H:%M:%S")), flush=True)
            print(json.dumps(stats, ensure_ascii=False), flush=True)

    eval_models = [dirname for _, dirname in run_keys]
    path, rows = evaluate.render_report(eval_models)
    print("评测报告：%s" % path, flush=True)
    for r in rows:
        if r.get("missing"):
            print("  %-16s （未跑）" % r["model"], flush=True)
        else:
            print("  %-16s 一致率=%s%% 幻觉率=%s%% JSON合规=%s%%"
                  % (r["model"], r["agreement"], r["hallucination"],
                     r["json_compliance"]), flush=True)


if __name__ == "__main__":
    main()
