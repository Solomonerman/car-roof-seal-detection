@echo off
rem 工控机一键启停台入口: 双击即用, 打开 GUI(无黑框, 且窗口必定弹到前台)
rem 要点: 不要用 start 把 GUI 丢到后台(会导致窗口创建却躲在当前窗口后面, 看起来像"没启动"); 改为前台阻塞调用
rem       同时用 /min 自启动自身, 让 bat 自己的 cmd 窗口最小化, 不碍眼
cd /d "%~dp0"

rem 第一次双击: 用最小化方式重新以 hide 参数启动自己, 原窗口立即退出
if "%~1"=="" (
    start "" /min cmd /c ""%~f0" hide"
    exit /b
)

rem ---- 以下为 hide 实例: 前台拉起 GUI ----
rem 解释器优先级: 仓库 .venv(与 PyCharm 用的同一套 python, 必含 tkinter+依赖) -> 系统 pyw/py
rem 注意: 必须先认 .venv, 否则会落到系统 pyw.exe(Python Launcher), 与 PyCharm 解释器不同环境, GUI 可能起不来
set "RUNNER="
if exist ".venv\Scripts\pythonw.exe" set "RUNNER=.venv\Scripts\pythonw.exe"
if not defined RUNNER if exist ".venv\Scripts\python.exe" set "RUNNER=.venv\Scripts\python.exe"
if not defined RUNNER where pyw.exe >nul 2>&1 && set "RUNNER=pyw.exe"
if not defined RUNNER if exist "C:\Windows\pyw.exe" set "RUNNER=C:\Windows\pyw.exe"
if not defined RUNNER where py.exe >nul 2>&1 && set "RUNNER=py.exe"
if not defined RUNNER if exist "C:\Windows\py.exe" set "RUNNER=C:\Windows\py.exe"

if defined RUNNER (
    "%RUNNER%" "launcher\工控机启停台.py"
    goto :eof
)
echo 未找到任何可用的 Python 启动器(仓库 .venv 或系统 py/pyw.exe)
echo 请确认 Python 已安装并加入 PATH(安装时勾选 Add Python to PATH), 或在该仓库建 .venv
pause
