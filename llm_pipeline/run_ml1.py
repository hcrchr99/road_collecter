# -*- coding: utf-8 -*-
"""M-L1 入口：只跑 L0 特征提取，无任何 LLM 调用。

用法（在 _repo/ 下）：
  python -m llm_pipeline.run_ml1                 # 默认包
  python -m llm_pipeline.run_ml1 --pack <zip>    # 指定包
"""
import argparse
import sys

from . import config, l0_extract


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 中文输出
    ap = argparse.ArgumentParser(description="RoadCheck L0 特征提取（M-L1）")
    ap.add_argument("--pack", default=str(config.DEFAULT_PACK))
    args = ap.parse_args()

    result, feats_path, cl_path = l0_extract.run(args.pack)
    n = len(result["records"])
    blocked = sum(1 for r in result["records"] if r["blocked"])
    print("pack_id            : %s" % result["pack_id"])
    print("特征块条数          : %d" % n)
    print("blocked（不进 L1/L2）: %d" % blocked)
    print("特征块输出          : %s" % feats_path)
    print("核对清单输出        : %s" % cl_path)
    for c in result["checklist"]["checks"]:
        print("  %-28s %s  (参照: %s)" % (c["item"], c["value"], c["expected"]))


if __name__ == "__main__":
    main()
