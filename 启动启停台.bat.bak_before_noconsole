@echo off
rem ============================================================
rem Launcher entry for seal-detection (double-click to run)
rem Root-cause fix: use ASCII-only content so cmd never garbles it.
rem Old bug: file saved as UTF-8, cmd reads GBK -> the Chinese script
rem name in the start line became garbage -> start failed silently.
rem The GUI script is now renamed to ASCII: launcher\launcher_gui.py
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
    start "" "%PY%" "launcher\launcher_gui.py"
    exit /b
)

rem fallback: system python.exe on PATH
python.exe "launcher\launcher_gui.py"
exit /b

:nofound
echo Python interpreter not found. Please create .venv or install Python and add to PATH.
pause
exit /b
