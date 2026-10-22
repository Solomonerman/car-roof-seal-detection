@echo off
rem 工控机一键启停台入口: 双击(或桌面快捷方式)即用, 打开 GUI(尽量无黑框)
rem 说明: GUI 内 _hide_own_console() 会在启动时隐藏控制台黑框, 故即使用 py.exe(带黑框)启动也无碍
cd /d "%~dp0"
set "RUNNER="

rem 1) 优先系统无窗口启动器 pyw.exe(与双击 .py 同 Launcher, 无黑框)
where pyw.exe >nul 2>&1 && set "RUNNER=pyw.exe"
rem 2) 兜底 C:\Windows 下的 Launcher(python 默认安装位置)
if not defined RUNNER if exist "C:\Windows\pyw.exe" set "RUNNER=C:\Windows\pyw.exe"
rem 3) 直接用 py.exe(双击 .py 同解释器; GUI 内 _hide_own_console 会隐藏黑框)
if not defined RUNNER where py.exe >nul 2>&1 && set "RUNNER=py.exe"
if not defined RUNNER if exist "C:\Windows\py.exe" set "RUNNER=C:\Windows\py.exe"
rem 4) 仓库 venv 的 pythonw(最后回退)
if not defined RUNNER if exist ".venv\Scripts\pythonw.exe" set "RUNNER=.venv\Scripts\pythonw.exe"

if defined RUNNER (
    start "" "%RUNNER%" "launcher\工控机启停台.py"
    goto :eof
)
echo 未找到任何可用的 Python 启动器(pyw.exe / py.exe / .venv\Scripts\pythonw.exe)
echo 请确认 Python 已安装并加入 PATH(安装时勾选 Add Python to PATH), 或在该仓库建 .venv
pause
