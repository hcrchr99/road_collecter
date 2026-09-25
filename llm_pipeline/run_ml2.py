# -*- coding: utf-8 -*-
"""M-L2 入口：跑指定模型（或全部）的 L2 判读 + 评测对照表。

用法：
  python -m llm_pipeline.run_ml2 --model glm-4.6v [--limit 10]
  python -m llm_pipeline.run_ml2 --model all            # 三个模型都跑
  python -m llm_pipeline.run_ml2 --eval-only            # 只重算评测
"""
import argparse
import json
import sys

from . import config, judge


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="RoadCheck L2 判读 + 评测（M-L2）")
    ap.add_argument("--model", default="all",
                    help="glm-4.6v | qwen3-vl-32b | qwen3-vl-8b | all")
    ap.add_argument("--limit", type=int, default=None, help="冒烟：只跑前 N 个未判事件")
    ap.add_argument("--fresh", action="store_true", help="清空该模型已有判读重跑")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    models = list(config.MODELS.keys()) if args.model == "all" else [args.model]

    if not args.eval_only:
        from . import l0_extract, unpack
        features_path = config.OUTPUT_DIR / "l0" / "features.jsonl"
        if not features_path.exists():
            print("L0 结果不存在，先自动跑 run_ml1 ...")
            l0_extract.run()
        records = [json.loads(l) for l in
                   open(features_path, encoding="utf-8").read().splitlines() if l.strip()]
        for m in models:
            cfg = config.MODELS[m]
            if cfg["api_key_env"] and not __import__("os").environ.get(cfg["api_key_env"]):
                print("跳过 %s：未设置 %s" % (m, cfg["api_key_env"]))
                continue
            print("== 判读：%s（%d 条任务）==" % (m, len(records)))
            with unpack.Pack(config.DEFAULT_PACK) as pack:
                stats = judge.judge_pack(m, pack, records,
                                         limit=args.limit, fresh=args.fresh)
            print("完成：%s" % json.dumps(stats, ensure_ascii=False))

    from . import evaluate
    path, _rows = evaluate.render_report(models)
    print("评测报告：%s" % path)


if __name__ == "__main__":
    main()
