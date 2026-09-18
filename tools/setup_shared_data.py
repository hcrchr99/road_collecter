#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享数据目录脚手架

在共享盘上一次性建好约定的目录结构。团队协作开始前每台机器跑一次即可，
重复执行是安全的（只补缺，不覆盖已有内容）。

用法：
    python tools/setup_shared_data.py --root D:/shared/roadcheck
    python tools/setup_shared_data.py --root /Volumes/team/roadcheck

依赖：无
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

README_RAW = """# raw/ —— 采集包原件（只读）

这里的每一份采集包都是**原始证据**，一旦落盘：

- 不复写
- 不重命名
- 不在原地编辑

发现问题时不要修文件，而是：

1. 原包保持不动
2. 把诊断写进 `<pack_id>.notes.md`
3. 需要修正时产出新文件 `<pack_id>.fixed.zip`

原因：任何报告里的数字都要能回溯到当时用的那份原始数据。覆盖原包会让
历史结论永远对不上。

## 目录内容

- `*.zip` —— 采集端导出的采集包
- `*.notes.md` —— 该包的诊断备注（可选）
- `MANIFEST.csv` —— 采集包登记表（不含轨迹坐标，可安全入库到 Git）

## 权限建议

本目录在共享盘上应设为**只读**，只有采集者首次写入。新包由采集者
用 `tools/register_pack.py` 登记后放入。
"""

README_LABELS = """# labels/ —— 标注文件

## 铁律：一人一档

**绝不允许两个 agent 往同一个文件里追加行。** 按 `pack_id` 切分：

    <pack_id>.labels.csv              整包归一个人标
    <pack_id>.part-<name>.csv         多人分帧区间标

需要汇总时用只读的合并脚本产出 `labels.all.csv`，这个文件本身也不该手写。

## 字段定义

见仓库内 `labels/schema.md`。所有人生成的标注文件必须遵守该 schema，
否则合并时字段对不上。

## 状态流转

    文件不存在       → 未开始
    labels.csv 存在   → 进行中
    label_status=done → 完成（在 MANIFEST.csv 里更新）
"""

README_GOLDEN = """# golden/ —— 黄金集（冻结后不得修改）

这里是纯人工标注的评估基准。一旦开始用它评估模型，就**不能再改**。

## 冻结约定

标完后在对应目录的 README.md 里写明：

    - 标注人
    - 标注日期
    - 依据的 schema 版本
    - 内容哈希
    - 状态：FROZEN

**要改就新建 `golden.v2/`**，并说明改了什么、为什么。这样任何一次评估
结果都能对应到确切的黄金集版本。

## 为什么冻结

如果一边评估一边改黄金集，那所有历史准确率数字都失去意义 —— 你无法
区分「模型变好了」和「标准变松了」。
"""

README_ROOT = """# roadcheck —— 团队共享数据区

本目录服务于「多人 + 多 agent」协作。上游规范见仓库内
`docs/协作与数据共享规范.md`。

## 三条通道

| 通道 | 放什么 | 是否公开 |
|---|---|---|
| GitHub 仓库 | 代码、报告、文档、schema | 公开 |
| 本共享盘 | 采集包、标注、黄金集 | 仅团队内部 |
| 本地工作区 | 脚本运行产物（out/ vis/ fused/） | 不共享 |

## 核心原则

1. **按敏感度分层**，不按文件类型分层
2. **共享「输入 + 脚本」，不共享中间产物** —— 产物脚本可重造
3. **源数据只读** —— 见 raw/README.md

## 目录

    raw/      采集包原件（只读）+ MANIFEST.csv 登记表
    labels/   标注文件（一人一档）
    golden/   黄金集（冻结后不可改）

## 隐私红线

`raw/` 与 `golden/` 内含真实 GPS 轨迹，**只在本共享区内流转**：

- 不得提交到任何公开仓库
- 不得放进演示视频或提交材料
- 对外只发布聚合后的病害点

## 常用命令

    # 登记新采集包
    python tools/register_pack.py <包路径> --collected-by alice --mode bike --storage tdrive

    # 看待标注的
    python tools/register_pack.py --list --status todo

    # 标完更新状态
    python tools/register_pack.py --update <pack_id> --label-status done --labeled-by bob
"""


def write_if_absent(path, content):
    if os.path.exists(path):
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return True


def main():
    ap = argparse.ArgumentParser(description="建共享数据目录结构")
    ap.add_argument("--root", required=True, help="共享盘上的根目录，如 D:/shared/roadcheck")
    ap.add_argument("--force-readme", action="store_true",
                    help="覆盖已存在的 README（默认不覆盖，防止冲掉别人写的内容）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    dirs = [
        root,
        os.path.join(root, "raw"),
        os.path.join(root, "labels"),
        os.path.join(root, "golden"),
    ]

    created_dirs, created_files, kept = [], [], []
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            created_dirs.append(d)

    targets = [
        (os.path.join(root, "README.md"), README_ROOT),
        (os.path.join(root, "raw", "README.md"), README_RAW),
        (os.path.join(root, "labels", "README.md"), README_LABELS),
        (os.path.join(root, "golden", "README.md"), README_GOLDEN),
    ]
    for p, content in targets:
        if args.force_readme and os.path.exists(p):
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            created_files.append(p)
        elif write_if_absent(p, content):
            created_files.append(p)
        else:
            kept.append(p)

    print("共享数据目录已就绪：%s" % root)
    if created_dirs:
        print("\n新建目录：")
        for d in created_dirs:
            print("  %s" % os.path.relpath(d, root))
    if created_files:
        print("\n新建说明文件：")
        for p in created_files:
            print("  %s" % os.path.relpath(p, root))
    if kept:
        print("\n已存在、未改动（加 --force-readme 可覆盖）：")
        for p in kept:
            print("  %s" % os.path.relpath(p, root))

    print("")
    print("下一步：")
    print("  1. 把 raw/ 在共享盘上设为只读")
    print("  2. 采集包放入 raw/，用 tools/register_pack.py 登记")
    print("  3. 确认各人的 pack_id 分工不重叠")
    return 0


if __name__ == "__main__":
    sys.exit(main())
