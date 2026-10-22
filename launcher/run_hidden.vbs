' ============================================================
' Hidden launcher for the seal-detection GUI (no console window).
' Called by start.bat via wscript.exe. Runs the GUI python in a
' fully hidden window (window style 0) so no black console appears
' and nothing is left on the taskbar.
' ASCII-only on purpose -> never garbled by cmd/codepage.
' ============================================================
Option Explicit

Dim shell, fso, baseDir, py, gui, cmd
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Base dir = the folder containing this .vbs (repo\launcher\), go up one to repo root
baseDir = fso.GetParentFolderName(WScript.ScriptFullName)   ' ...\launcher
baseDir = fso.GetParentFolderName(baseDir)                   ' ... repo root

' Resolve interpreter: prefer repo .venv python.exe / pythonw.exe
py = baseDir & "\.venv\Scripts\python.exe"
If Not fso.FileExists(py) Then
    py = baseDir & "\.venv\Scripts\pythonw.exe"
End If
If Not fso.FileExists(py) Then
    py = "python.exe"   ' fallback to PATH
End If

gui = baseDir & "\launcher\launcher_gui.py"

' Run hidden: window style 0 = completely hidden, False = do not wait
cmd = """" & py & """ """ & gui & """"
shell.Run cmd, 0, False
