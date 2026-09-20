# 📚 AI Study Pack Generator

A multi-stage AI workflow that turns a topic, pasted notes, or an uploaded PDF into a personalised study pack.

**Workflow:** Planner → Content generation (per module) → Assessment → Reviewer → Refiner

Each stage passes context to the next, validates its own output, retries on errors, and degrades gracefully if a step fails.

## Features
- Personalised by level, goal, available days, minutes per day, learning style and weak areas
- Lessons, key terms, worked examples, flashcards, quiz and day-by-day schedule
- Automatic checks plus an AI reviewer that scores the pack and triggers targeted rewrites
- Live progress, review report and workflow trace
- Download the pack as Markdown or JSON

## Tech stack
Python, Streamlit, Groq API (Llama 3.3 70B), pypdf

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Add your key in `.streamlit/secrets.toml` (never commit this file):
```toml
GROQ_API_KEY = "your_key_here"
```
Or paste it in the app sidebar.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to share.streamlit.io and click **Create app**.
3. Select the repo, branch `main`, main file `app.py`.
4. In **Advanced settings → Secrets**, add `GROQ_API_KEY = "your_key_here"`.
5. Click **Deploy**.
