# Mental Health Dashboard Frontend

A standalone SaaS-style dashboard for the AI Mental Health Risk Analyzer.

## How to Run

1. Set your OpenRouter key before starting if you want chat:

   PowerShell:
   `$env:OPENROUTER_API_KEY="your_key_here"`

2. Start the full app (backend + frontend):

   `python api_server.py`

3. Open `http://127.0.0.1:8000/` and submit the input form.

Optional (JSON-only mode):

`python export_dashboard_data.py --sample-index 0 --output frontend/dashboard_data.json`

## Files

- `index.html` — Dashboard layout and structure
- `style.css` — Glassmorphic dark theme styling
- `script.js` — Chart.js rendering and animations

## Data

Update `frontend/dashboard_data.json` (exported by `export_dashboard_data.py`).

## Notes

- Uses Chart.js via CDN.
- Works offline after first load if the CDN is cached.
