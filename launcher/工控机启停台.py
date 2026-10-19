# -*- coding: utf-8 -*-
"""
工控机一键启停台(GUI, tkinter, Python 自带零额外依赖)
现场人员: 双击桌面"启动启停台"快捷方式 -> 点 [启动] / [停止]。

说明:
- 启停台本身用 pythonw 运行(纯 GUI 无黑框); 拉起的 web_capture/ui.app 用 python.exe
  (带控制台黑框, 里面是实时日志, 点[停止]后程序退出黑框自动消失, 平时请勿手动关黑框)。
- 关闭窗口会先询问是否停止程序, 避免误关导致子进程遗留。
- 仅管进程启停, 绝不读写 PLC(web_capture 本身只读)。
"""
import os
import sys
import webbrowser
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_config as C
import process_manager as PM

REPO_ROOT = C.REPO_ROOT


class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车顶胶条检测 - 工控机启停台")
        self.root.geometry("480x440")
        self.procs = {}  # name -> Popen

        # 状态灯 + 文字
        frm = tk.Frame(root)
        frm.pack(pady=10)
        self.lamp = tk.Label(frm, text="●", font=("Arial", 26), fg="gray")
        self.lamp.pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar(value="未启动")
        tk.Label(frm, textvariable=self.status_var, font=("Arial", 14)).pack(side=tk.LEFT)

        # 大按钮
        btn_frm = tk.Frame(root)
        btn_frm.pack(pady=12)
        self.btn_start = tk.Button(btn_frm, text="▶ 启动", width=14, height=2,
                                   bg="#2e7d32", fg="white", command=self.on_start)
        self.btn_start.pack(side=tk.LEFT, padx=12)
        self.btn_stop = tk.Button(btn_frm, text="■ 停止", width=14, height=2,
                                  bg="#c62828", fg="white", command=self.on_stop,
                                  state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=12)

        # 日志框
        tk.Label(root, text="状态日志:").pack(anchor="w", padx=14)
        self.txt = scrolledtext.ScrolledText(root, height=15, state=tk.DISABLED)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    def _ensure_python(self):
        py = C.find_python()
        if py and C.python_works(py):
            return py
        # 让用户现场选 PyCharm 用的那个 python.exe
        path = filedialog.askopenfilename(
            title="选择工控机 Python 解释器(即 PyCharm 用的 python.exe)",
            filetypes=[("Python 可执行文件", "python.exe")],
        )
        if not path:
            return None
        if not C.python_works(path):
            messagebox.showerror(
                "解释器不可用",
                "该 python.exe 缺少依赖(pypylon / opencv / flask 等),\n"
                "请选择 PyCharm 实际用的那个解释器。",
            )
            return None
        C.save_python(path)
        return path

    def on_start(self):
        py = self._ensure_python()
        if not py:
            self.log("[错误] 未配置可用的 Python 解释器, 已取消启动")
            return
        self.log(f"[信息] 使用解释器: {py}")
        started = 0
        for item in C.LAUNCH_ITEMS:
            try:
                p, log_path = PM.start_item(py, item, REPO_ROOT)
                self.procs[item["name"]] = p
                self.log(f"[启动] {item['name']} (PID={p.pid})  日志: {log_path}")
                started += 1
                if item.get("open_url"):
                    webbrowser.open(item["open_url"])
            except Exception as e:
                self.log(f"[错误] 启动 {item['name']} 失败: {e}")
        if started:
            self._set_running()

    def on_stop(self):
        for name, p in list(self.procs.items()):
            ok = PM.stop_item(p)
            self.log(f"[{'已停止' if ok else '强杀兜底'} ] {name} (PID={p.pid})")
        self.procs.clear()
        self._set_stopped()

    def _set_running(self):
        self.status_var.set("运行中")
        self.lamp.configure(fg="#2e7d32")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

    def _set_stopped(self):
        self.status_var.set("已停止")
        self.lamp.configure(fg="gray")
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)

    def on_close(self):
        if self.procs:
            if not messagebox.askyesno("确认", "程序仍在运行, 关闭窗口会先停止程序吗?"):
                return
            self.on_stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
