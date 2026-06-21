@echo off

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app_pose.py

pause