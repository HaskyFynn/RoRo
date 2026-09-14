@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" check_install.py
if errorlevel 1 goto fail
echo Installation complete. Open launch.cmd to begin.
pause
exit /b 0
:fail
echo Installation failed. See the message above. Python 3.11-3.13 with Tk is recommended.
pause
exit /b 1
