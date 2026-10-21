@echo off
rem Development start of the launcher (venv in the repository root).
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\pythonw.exe" -m fl_launcher
