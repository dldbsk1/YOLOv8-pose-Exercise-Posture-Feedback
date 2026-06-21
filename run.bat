@echo off

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12 is not installed.
    echo Please install Python 3.12 from https://www.python.org/downloads/
    pause
    exit /b
)

if not exist .venv (
    py -3.12 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m streamlit run app_pose.py

pause