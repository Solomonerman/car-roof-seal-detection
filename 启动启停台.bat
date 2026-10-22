@echo off
rem ============================================================
rem Launcher entry for seal-detection (double-click to run).
rem No black console window: we hand off to a VBScript that runs
rem the GUI python in a fully hidden window, then this .bat exits
rem immediately so nothing stays on the taskbar.
rem ASCII-only -> never garbled by cmd codepage.
rem ============================================================
cd /d "%~dp0"
start "" wscript.exe "launcher\run_hidden.vbs"
exit /b
