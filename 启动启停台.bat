@echo off
rem 工控机一键启停台入口: 双击(或桌面快捷方式)即用, 打开 GUI(无黑框)
cd /d "%~dp0"
rem 优先用系统无窗口启动器 pyw.exe(与双击 .py 同一解释器, 自带 tkinter, 无黑框)
rem 背景: 工控机 venv 的 pythonw.exe 有时缺 tkinter 导致静默闪退, 故改走系统 pyw
where pyw.exe >nul 2>&1
if %errorlevel%==0 (
    pyw.exe "launcher\工控机启停台.py"
    goto :eof
)
rem 回退: 仓库 venv 的 pythonw(无窗口)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "launcher\工控机启停台.py"
    goto :eof
)
echo 未找到 pyw.exe, 也未找到 .venv\Scripts\pythonw.exe
echo 请确认 Python 已安装并加入 PATH, 或在该仓库建 venv: python -m venv .venv
pause
