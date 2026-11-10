@echo off
REM Starts the InsightFlow dashboard (the backend must be running)
call venv\Scripts\activate
streamlit run frontend/streamlit_app.py
