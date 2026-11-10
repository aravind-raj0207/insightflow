@echo off
REM Starts the InsightFlow backend (run setup steps from README first)
call venv\Scripts\activate
uvicorn app.main:app --reload
