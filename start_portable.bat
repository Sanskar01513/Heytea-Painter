@echo off
setlocal enabledelayedexpansion
title Heytea Painter
cd /d "%~dp0"

REM Check if local venv exists
if not exist "venv\" (
    echo [ERROR] 虚拟环境不存在！
    echo 请先运行: setup_portable.bat
    echo.
    pause
    exit /b 1
)

REM Check if Python is in venv
if not exist "venv\python.exe" (
    echo [ERROR] Python 未找到！
    echo 请重新运行: setup_portable.bat
    echo.
    pause
    exit /b 1
)

echo 启动 Heytea Painter...
echo.

REM Use venv Python directly
"%~dp0venv\python.exe" heytea_modern.py

if errorlevel 1 (
    echo.
    echo [ERROR] 程序运行失败
    pause
)
