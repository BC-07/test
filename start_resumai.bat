@echo off
title ResuAI - Personal Data Sheet & Resume Screening System

echo =====================================
echo  ResuAI Startup Script
echo  Version 2.0.0 - Excel Support
echo =====================================
echo.

echo Starting ResuAI...
echo.

cd /d "c:\Users\Lenar Yolola\OneDrive\Desktop\ResuAI__"

echo Current directory: %CD%
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Starting application...
echo Open your browser to: http://localhost:5000
echo.
echo Default Login:
echo Email: admin@resumeai.local
echo Password: admin123
echo.
echo Press Ctrl+C to stop the server
echo =====================================
echo.

python start.py

echo.
echo Application stopped.
pause