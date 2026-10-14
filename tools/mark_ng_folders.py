#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性迁移脚本：把 data/process_data 下“已检测出 NG”的旧文件夹改名加 _NG 后缀。

背景：现场已存在的旧检测结果文件夹（命名还是 时间_车型，且没带结果标记），
      本脚本读每个文件夹里的 summary.json 判断该车是否 NG，是则重命名为 ..._NG。
      同时把 SQLite 记录里存的 proc_dir 同步改成新名，保证看板仍能正确读取。

用法（在工控机项目根目录执行，python 用现场 venv 或任意 python 均可，本脚本只用标准库）：
    python tools/mark_ng_folders.py            # 预览模式：只打印将要改的，不改任何东西
    python tools/mark_ng_folders.py --apply    # 真正执行改名 + 改库

安全约束：
  - 只动 data/process_data/ 下符合“时间戳_车型”(如 20261014_121530_8X) 命名规律的真实车文件夹；
  - 测试文件夹(test_board / real_* / verify_*) 与已经带 _NG / _RECHECK 后缀的都不动；
  - 没有 summary.json（无法判定 NG）的文件夹一律跳过，绝不误删/误改。
"""
import argparse
import json
import os
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_ROOT = os.path.join(ROOT, "data", "process_data")
DB_PATH = os.path.join(ROOT, "data", "inspection.db")
OLD_RE = re.compile(r"^\d{8}_\d{6}_")          # 旧命名：时间戳_车型 开头（真实车）
RESULT_SUFFIX = ("_NG", "_RECHECK")


def read_ok(folder):
    """读该文件夹 summary.json 的顶层 ok 字段；返回 (ok, 说明)。"""
    sj = os.path.join(folder, "summary.json")
    if not os.path.isfile(sj):
        return None, "no_summary"
    try:
        with open(sj, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:                       # 损坏的 json 不当成 NG
        return None, "json_err:%s" % e
    ok = data.get("ok", None)
    return ok, "ok" if ok is not None else "no_ok_field"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="真正执行改名+改库（默认仅预览，不改动任何文件）")
    args = ap.parse_args()

    if not os.path.isdir(PROC_ROOT):
        print("[迁移] 找不到 %s，请在项目根目录运行" % PROC_ROOT)
        sys.exit(1)

    entries = sorted(os.listdir(PROC_ROOT))
    ng_list, skip = [], 0
    for name in entries:
        full = os.path.join(PROC_ROOT, name)
        if not os.path.isdir(full):
            continue
        if name.endswith(RESULT_SUFFIX):          # 已标记过
            skip += 1
            continue
        if not OLD_RE.match(name):                # 非真实车文件夹（测试目录等）
            skip += 1
            continue
        ok, _why = read_ok(full)
        if ok is not False:                        # 正常车 / 无法判定 → 不动
            if ok is None:
                skip += 1
            continue
        ng_list.append((name, name + "_NG"))       # ok == False → NG

    print("[迁移] 扫描 %d 项，命中待标记 NG 文件夹 %d 个，跳过 %d 个"
          % (len(entries), len(ng_list), skip))
    if not ng_list:
        print("[迁移] 无需要改动的文件夹，结束")
        return

    print("[迁移] %s" % ("预览(加 --apply 才执行)" if not args.apply else "执行改名+改库中..."))
    for name, new_name in ng_list:
        old_rel = os.path.join("data", "process_data", name)
        new_rel = os.path.join("data", "process_data", new_name)
        print("   %s  ->  %s" % (name, new_name))
        if not args.apply:
            continue
        try:
            os.rename(os.path.join(PROC_ROOT, name),
                      os.path.join(PROC_ROOT, new_name))
        except Exception as e:
            print("   [失败] 改名 %s: %s" % (name, e))
            continue
        # 同步数据库 proc_dir，避免看板读旧路径失效
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("UPDATE records SET proc_dir=? WHERE proc_dir=?",
                             (new_rel, old_rel))
                conn.commit()
        except Exception as e:
            print("   [警告] 数据库 proc_dir 更新失败（文件夹已改名）: %s" % e)

    print("[迁移] 完成。%s"
          % ("（预览模式，未改动任何文件）" if not args.apply else "（已改名+改库）"))


if __name__ == "__main__":
    main()
