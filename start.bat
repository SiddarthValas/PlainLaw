@echo off
setlocal
REM PlainLaw: double-click to start (Windows). First run sets up a private Python environment (5-10 minutes).
cd /d "%~dp0"
title PlainLaw
echo Starting PlainLaw...

REM find a real Python 3 (the "py" launcher first; plain "python" can be a Microsoft Store shortcut that does nothing)
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY (
  python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo Python 3 was not found.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH" on the first screen.
  echo Then double-click start.bat again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: installing what PlainLaw needs. This happens once and takes 5-10 minutes...
  %PY% -m venv .venv
  if errorlevel 1 goto setupfail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements-run.txt
  if errorlevel 1 goto setupfail
)

REM optional: a Groq API key in groq_key.txt (one line) switches on AI understanding of messages
if exist groq_key.txt (
  set "LLM_BASE=https://api.groq.com/openai/v1"
  set /p LLM_KEY=<groq_key.txt
  echo AI understanding: on ^(Groq^)
) else (
  echo AI understanding: off ^(rules and search only^). Add groq_key.txt to switch it on.
)

REM open the browser after the app has had time to load
start "" cmd /c "timeout /t 25 >nul & start http://localhost:8000"
echo.
echo PlainLaw will open in your browser at http://localhost:8000 in about 25 seconds.
echo Keep this window open. Close it to stop PlainLaw.
echo.
".venv\Scripts\python.exe" -m uvicorn api:app --port 8000
echo.
echo PlainLaw stopped. If you see an error above, take a screenshot of this window.
pause
exit /b 0

:setupfail
echo.
echo Setup failed. Check the internet connection and run start.bat again.
echo If it fails again, take a screenshot of this window.
if exist .venv rmdir /s /q .venv
pause
exit /b 1
