#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性迁移脚本：把 data/process_data 下的旧检测文件夹统一重命名为新格式，并补上 NG/RECHECK 标记。

背景：
  工控机现场已存在的旧检测结果文件夹，命名有两种历史格式：
    A. pin_车型       （如 LHG1234_8X）—— 最老部署版本生成，名里只有 PIN + 车型
    B. 时间_车型      （如 20261014_121530_8X）—— 中间版本生成，名里只有 时间 + 车型
  新格式（web_capture.py 现已采用）为：时间_车型_滑橇_PIN，并在检测结论后置 _NG / _RECHECK 后缀。
  本脚本把 A/B 两类旧文件夹全部重建为新全名，并按 summary.json 补 NG/RECHECK 标记；
  同时把 SQLite records 表里的 proc_dir 同步改成新名，保证看板仍能正确读取。

  注：旧 summary.json 只含 per_photo/ok/loc_ok/message，没有“时间”和“滑橇号”，
      因此新名里的“时间”优先取数据库 records.timestamp（最准），取不到则退回文件夹修改时间；
      “滑橇号”按旧名(或PIN)在数据库里反查，记录已被清理(KEEP_RECORDS)则留空；
      “PIN”旧格式 A 名里就有，格式 B 则从数据库反查。

用法（在工控机项目根目录执行，python 用现场 venv 或任意 python 均可，本脚本只用标准库）：
    python tools/mark_ng_folders.py            # 预览模式：只打印将要改的，不改任何东西
    python tools/mark_ng_folders.py --apply    # 真正执行改名 + 改库

安全约束：
  - 只动 data/process_data/ 下的“真实车文件夹”；测试目录(test_/real_/verify_*/stage*/tmp*/sample*) 一律跳过；
  - 已经是新全名格式(时间_车型_滑橇_PIN，含或不含结果后缀)的，跳过（幂等，重跑不叠加）；
  - 没有 summary.json（无法判定 NG）的文件夹：仍改名，但不强行加 _NG；
  - 空目录 / 不含任何检测产物(.bmp/.jpg/.png/.json)的目录：视为非车文件夹，跳过，绝不误改；
  - 绝不做删除操作。
"""
import argparse
import datetime
import json
import os
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_ROOT = os.path.join(ROOT, "data", "process_data")
DB_PATH = os.path.join(ROOT, "data", "inspection.db")

TEST_PREFIXES = ("test_", "real_", "verify_", "stage", "tmp", "sample")
RESULT_SUFFIX = ("_NG", "_RECHECK")
TIME_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)$")   # 时间_车型 或 时间_车型_滑橇_PIN
PINKEY_RE = re.compile(r"^([^_]+)_([^_]+)$")       # pin_车型（无时间）


def _safe_name(s):
    """与 web_capture.py 的 _safe_name 保持一致：文件名安全片段。"""
    s = "" if s is None else str(s)
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")
    return s[:40] or "x"


def _looks_like_car_folder(full):
    """只迁移确实含有检测产物(图片或 json)的目录，避免误改空/杂目录。"""
    try:
        for fn in os.listdir(full):
            if fn.lower().endswith((".bmp", ".jpg", ".jpeg", ".png", ".json")):
                return True
    except Exception:
        return False
    return False


def _classify(name):
    """返回 (category, base, suffix)。
    category: 'new' 已是新全名; 'old_time_key' 时间_车型; 'old_pin_key' pin_车型;
              'test' 测试目录; 'skip' 其它忽略。
    base: 去掉结果后缀后的真实车名；suffix: 原结果后缀或 ''。
    """
    suffix = ""
    base = name
    for suf in RESULT_SUFFIX:
        if name.endswith(suf):
            suffix = suf
            base = name[: -len(suf)]
            break
    if base.startswith(TEST_PREFIXES):
        return "test", base, suffix
    m = TIME_RE.match(base)
    if m:
        segs = m.group(3).split("_")
        if len(segs) >= 3:
            return "new", base, suffix      # 时间_车型_滑橇_PIN（已迁移）
        return "old_time_key", base, suffix  # 时间_车型
    if PINKEY_RE.match(base):
        return "old_pin_key", base, suffix   # pin_车型
    return "skip", base, suffix


def _db_lookup(old_rel):
    """按旧 proc_dir（兼容 / 与 \\ 两种分隔）反查 timestamp/skid/pin。
    若旧名含 PIN（pin_车型），再退一步用 pin 反查（取最新一条）。
    返回 (timestamp_iso, skid, pin)，取不到的字段为 None。
    """
    if not os.path.isfile(DB_PATH):
        return None, None, None
    old_fwd = old_rel.replace("\\", "/")
    old_bwd = old_rel.replace("/", "\\")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT timestamp, skid, pin FROM records "
                "WHERE proc_dir=? OR proc_dir=?",
                (old_fwd, old_bwd)).fetchone()
            if row and (row[0] or row[1] or row[2]):
                return row[0], row[1], row[2]
            m = PINKEY_RE.match(os.path.basename(old_fwd))
            if m:
                pin = m.group(1)
                row = conn.execute(
                    "SELECT timestamp, skid, pin FROM records WHERE pin=? "
                    "ORDER BY id DESC LIMIT 1", (pin,)).fetchone()
                if row:
                    return row[0], row[1], row[2]
    except Exception as e:
        print("   [警告] 数据库查询失败: %s" % e)
    return None, None, None


def _result_suffix_from_summary(folder):
    """读 summary.json 判定结果后缀：ok==False -> _NG；否则 loc_ok==False -> _RECHECK。
    无 summary / 无法判定 -> ''（不强标记）。老 summary 可能无 loc_ok 字段，按可靠处理。
    """
    sj = os.path.join(folder, "summary.json")
    if not os.path.isfile(sj):
        return ""
    try:
        with open(sj, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    ok = data.get("ok", None)
    loc_ok = data.get("loc_ok", True)
    if ok is False:
        return "_NG"
    if loc_ok is False:
        return "_RECHECK"
    return ""


def _build_new_base(base, cat, db_ts, db_skid, db_pin):
    """根据旧名类别 + 数据库信息，构造新全名（不含结果后缀）：时间_车型_滑橇_PIN。"""
    if cat == "old_pin_key":
        pin, key = base.rsplit("_", 1)
        skid = db_skid or ""
        time_str = None
    else:  # old_time_key
        m = TIME_RE.match(base)
        time_str = "%s_%s" % (m.group(1), m.group(2))
        key = m.group(3)
        skid = db_skid or ""
        pin = db_pin or ""
    # 时间：优先数据库抓拍时间，否则文件夹修改时间，再否则当前时间
    if db_ts:
        try:
            time_str = _stamp_from_dt(datetime.datetime.fromisoformat(db_ts))
        except Exception:
            pass
    if not time_str:
        try:
            mt = os.path.getmtime(os.path.join(PROC_ROOT, base))
            time_str = _stamp_from_dt(datetime.datetime.fromtimestamp(mt))
        except Exception:
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 与 web_capture.py 保持一致：时间原样，车型/滑橇/PIN 经 _safe_name
    parts = [time_str, _safe_name(key)]
    if skid:
        parts.append(_safe_name(skid))
    if pin:
        parts.append(_safe_name(pin))
    return "_".join(parts)


def _stamp_from_dt(dt):
    return dt.strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="真正执行改名+改库（默认仅预览，不改动任何文件）")
    args = ap.parse_args()

    if not os.path.isdir(PROC_ROOT):
        print("[迁移] 找不到 %s，请在项目根目录运行" % PROC_ROOT)
        sys.exit(1)

    entries = sorted(os.listdir(PROC_ROOT))
    todo, skipped = [], 0
    for name in entries:
        full = os.path.join(PROC_ROOT, name)
        if not os.path.isdir(full):
            continue
        cat, base, old_suffix = _classify(name)
        if cat in ("test", "skip", "new"):
            skipped += 1
            continue
        if not _looks_like_car_folder(full):
            skipped += 1
            continue
        old_rel = os.path.join("data", "process_data", name)
        db_ts, db_skid, db_pin = _db_lookup(old_rel)
        new_base = _build_new_base(base, cat, db_ts, db_skid, db_pin)
        suffix = old_suffix or _result_suffix_from_summary(full)
        new_name = new_base + suffix
        if new_name == name:
            skipped += 1
            continue
        tsrc = "db" if db_ts else "mtime"
        ssrc = "has_skid" if db_skid else "no_skid"
        todo.append((name, new_name, tsrc, ssrc))

    mode = "预览(加 --apply 才执行)" if not args.apply else "执行改名+改库中"
    print("[迁移] 扫描 %d 项，待迁移 %d 个，跳过 %d 个 (%s)"
          % (len(entries), len(todo), skipped, mode))
    no_skid = sum(1 for t in todo if t[3] == "no_skid")
    mtime_fb = sum(1 for t in todo if t[2] == "mtime")
    if no_skid:
        print("[迁移] 注意：%d 个文件夹在数据库查不到滑橇号(可能已被清理)，新名将不含滑橇段。"
              % no_skid)
    if mtime_fb:
        print("[迁移] 注意：%d 个文件夹在数据库查不到抓拍时间，时间取自文件夹修改时间。"
              % mtime_fb)

    for name, new_name, tsrc, ssrc in todo:
        print("   %s  ->  %s   [时间:%s 滑橇:%s]"
              % (name, new_name, tsrc, ssrc))
        if not args.apply:
            continue
        try:
            os.rename(os.path.join(PROC_ROOT, name),
                      os.path.join(PROC_ROOT, new_name))
        except Exception as e:
            print("   [失败] 改名 %s: %s" % (name, e))
            continue
        old_rel_fwd = os.path.join("data", "process_data", name).replace("\\", "/")
        old_rel_bwd = old_rel_fwd.replace("/", "\\")
        new_rel_fwd = os.path.join("data", "process_data", new_name).replace("\\", "/")
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE records SET proc_dir=? WHERE proc_dir=? OR proc_dir=?",
                    (new_rel_fwd, old_rel_fwd, old_rel_bwd))
                conn.commit()
        except Exception as e:
            print("   [警告] 数据库 proc_dir 更新失败（文件夹已改名）: %s" % e)

    print("[迁移] 完成。%s"
          % ("（预览模式，未改动任何文件）" if not args.apply else "（已改名+改库）"))


if __name__ == "__main__":
    main()
