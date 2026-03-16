# Public Web Deployment Bundle

## Purpose

This bundle is intended for public deployment so that reviewers and clinicians can open the application directly in a browser without installing Python locally.

## What this bundle contains

- `app.py`: deployable Flask backend
- `clinical_risk_comparison.html`: browser interface
- `clinical_web_model_card.json`: metadata and benchmark configuration
- `runtime/`: model and preprocessor artifacts
- `requirements.txt`: Python dependencies
- `Procfile` and `render.yaml`: direct deployment entrypoints

## Recommended deployment targets

- Render
- Railway
- Fly.io
- Any Linux server that can run `gunicorn`

## Quick deployment on Render

1. Upload this folder to a Git repository.
2. Create a new Render Web Service from that repository.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:$PORT app:app`
4. After deployment, open the generated public URL.

## Local smoke test

```bash
pip install -r requirements.txt
python app.py
```

Then open:

`http://127.0.0.1:8765`
