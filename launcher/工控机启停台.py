# -*- coding: utf-8 -*-
"""
工控机一键启停台(GUI, tkinter, Python 自带零额外依赖)
现场人员: 双击桌面"启动启停台"快捷方式 -> 点 [启动] / [停止]。

说明:
- 启停台本身用 pythonw 运行(纯 GUI 无黑框); 拉起的 web_capture/ui.app 用 python.exe
  (控制台窗口已设为完全隐藏 SW_HIDE: 不进任务栏、桌面不可见、不会被误关; 控制台对象仍保留,
  以便[停止]发 Ctrl+Break 优雅退出释放相机; 日志全部重定向进文件, 无需看黑框)。
- 关闭窗口会先询问是否停止程序, 避免误关导致子进程遗留。
- 仅管进程启停, 绝不读写 PLC(web_capture 本身只读)。
"""
import os
import sys
import ctypes
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
        # 防呆1: 本 launcher 若仍残留未停止的子进程, 先停掉再起
        if any(p and p.poll() is None for p in self.procs.values()):
            self.log("[防呆] 本启停台仍有未停止的子进程, 先停止...")
            self.on_stop()
        # 防呆2: 清理占用相机的旧 web_capture(别的启停台/手动 cmd 拉的),
        # 防止双开撞锁(报错 The device is controlled by another application)
        self._cleanup_strays()
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

    def _cleanup_strays(self):
        """防呆: 启动前扫描系统里是否已有别的 web_capture.py 占着相机, 有则先优雅停掉。"""
        my_pid = os.getpid()
        exclude = {my_pid}
        for p in self.procs.values():
            if p and p.pid:
                exclude.add(p.pid)
        strays = PM.find_stray_web_capture(exclude)
        if not strays:
            return
        self.log(f"[防呆] 检测到相机已被其他进程占用: PID={strays}, 先优雅停止释放相机...")
        ok = PM.stop_pids(strays)
        if ok:
            self.log("[防呆] 旧进程已释放相机, 可继续启动")
        else:
            self.log("[防呆] 旧进程未能完全停止, 若仍报'被另一程序控制'请到任务管理器结束 python.exe")

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


def _hide_own_console():
    """若本 GUI 进程是被控制台解释器(python.exe/py.exe)拉起的, 启动时隐藏自身控制台窗口,
    保证无论怎么启动 GUI 都无黑框。用 pythonw 启动时无控制台, GetConsoleWindow 返回 0 自动跳过。"""
    if sys.platform != "win32":
        return
    try:
        k = ctypes.windll.kernel32
        hwnd = k.GetConsoleWindow()
        if hwnd:
            k.ShowWindow(hwnd, 0)  # SW_HIDE = 0
    except Exception:
        pass


def main():
    _hide_own_console()
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
