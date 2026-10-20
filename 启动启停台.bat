@echo off
rem 工控机一键启停台入口: 双击(或桌面快捷方式)即用, 打开 GUI(无黑框)
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    rem start "" 让 bat 自己的 cmd 窗口启动后即关闭, 不再残留黑框
    start "" ".venv\Scripts\pythonw.exe" "launcher\工控机启停台.py"
) else (
    echo 未找到仓库 .venv\Scripts\pythonw.exe
    echo 请用 PyCharm 的解释器, 或先在该仓库建 venv: python -m venv .venv
    pause
)
