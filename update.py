# -*- coding: utf-8 -*-
"""一键在线更新脚本（小白友好）。

运行：在 PyCharm 终端输入  python update.py
作用：从云端下载最新版 zip，自动解压覆盖到本项目目录。
注意：只更新代码文件，你放在 data/raw_images/ 的现场真图不会被删除。
如果下载失败（提示链接失效），请联系助手重建下载链接。
"""
import os
import sys
import ssl
import zipfile
import tempfile
import urllib.request

# 云端最新版下载地址（由助手维护，沙盒重置后可能变化）
UPDATE_URL = ("https://webview.e2b.gz5.sandbox.cloudstudio.club/"
              "car_roof_seal_detection.zip"
              "?x-cs-sandbox-id=7446ed5646564d3ab7a7e775f48ee132"
              "&x-cs-sandbox-port=8000")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print(">>> 正在从云端下载最新版本 ...")
    tmp_zip = os.path.join(tempfile.gettempdir(), "car_roof_update.zip")
    try:
        ctx = ssl._create_unverified_context()
        urllib.request.urlretrieve(UPDATE_URL, tmp_zip, context=ctx)
    except Exception as e:
        print(f">>> 下载失败：{e}")
        print(">>> 链接可能已失效，请联系助手重建下载链接。")
        sys.exit(1)

    print(">>> 下载完成，正在解压更新（你的现场真图不会被删除）...")
    try:
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(BASE_DIR)
    except Exception as e:
        print(f">>> 解压失败：{e}")
        sys.exit(1)

    os.remove(tmp_zip)
    print(">>> 更新完成！请重新打开 / 刷新 PyCharm 项目后运行。")


if __name__ == "__main__":
    main()
