#!/bin/bash
# PlainLaw: double-click to start (macOS). First run sets up a private Python environment (a few minutes).
cd "$(dirname "$0")"
echo "Starting PlainLaw..."
# stop an older PlainLaw still running on port 8000, so the new version is the one that opens
OLD=$(lsof -ti :8000 2>/dev/null); if [ -n "$OLD" ]; then echo "Stopping the old PlainLaw..."; kill $OLD 2>/dev/null; sleep 2; fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Install it from https://www.python.org/downloads/ and double-click this again."
  read -p "Press Enter to close"; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "First run: installing what PlainLaw needs (one time, a few minutes)..."
  python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip -q && .venv/bin/python -m pip install -r requirements-run.txt
  if [ $? -ne 0 ]; then echo "Setup failed. Check your internet connection and try again."; rm -rf .venv; read -p "Press Enter to close"; exit 1; fi
fi
# optional: put a Groq API key in groq_key.txt (one line) to switch on AI understanding of messages
if [ -f groq_key.txt ]; then
  export LLM_BASE=https://api.groq.com/openai/v1
  export LLM_KEY="$(tr -d ' \r\n' < groq_key.txt)"
  echo "AI understanding: on (Groq)"
else
  echo "AI understanding: off (rules and search only). Add groq_key.txt to switch it on."
fi
# open the browser once the app answers
( until curl -s http://localhost:8000/api/health >/dev/null 2>&1; do sleep 2; done; open http://localhost:8000 ) &
echo "PlainLaw will open in your browser at http://localhost:8000 . Keep this window open; close it to stop."
.venv/bin/python -m uvicorn api:app --port 8000
