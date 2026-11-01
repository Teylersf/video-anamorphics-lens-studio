@echo off
cd /d "%~dp0"
python tools\anamorphic-studio\launch.py
if errorlevel 1 pause
