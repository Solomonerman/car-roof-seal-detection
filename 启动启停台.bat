@echo off
rem ============================================================
rem 工控机一键启停台入口（双击即用）
rem 推倒重做版设计要点：
rem   1) 不做任何"自重启 / 最小化自身"的花活 —— 旧版用 start /min 自重启会把 GUI
rem      进程丢到后台, Windows 禁止后台进程抢前台焦点, 导致窗口被藏起来(看起来像没启动)。
rem   2) 优先使用仓库 .venv 的解释器(与 PyCharm 同一套, 已验证可跑 GUI 且依赖齐全)。
rem   3) 用 start "" 把 GUI 以独立前台窗口拉起, 本 bat 的 cmd 窗口随即退出, 不留黑框。
rem   4) GUI 自身再用 _hide_own_console() 隐藏其控制台窗口, 全程无可见黑框。
rem ============================================================
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\pythonw.exe" (
    set "PY=.venv\Scripts\pythonw.exe"
)

if defined PY (
    if not exist "%PY%" goto :nofound
    start "" "%PY%" "launcher\工控机启停台.py"
    exit /b
)

rem 兜底：系统 PATH 里的 python.exe（可能带控制台，GUI 会自行隐藏）
python.exe "launcher\工控机启停台.py"
exit /b

:nofound
echo 未找到 Python 解释器：请确认仓库 .venv 已创建，或系统已安装 Python 并加入 PATH。
pause
exit /b
