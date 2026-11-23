# PrenatalAI

A runnable research prototype for prenatal ultrasound workflow exploration and evidence-grounded medical information retrieval.

## Important clinical limitation
This is **not a diagnostic device** and does not detect or rule out Spina Bifida. The included app has no trained or clinically validated ultrasound model. The ultrasound page only accepts and previews images. Do not upload identifiable patient data.

## Quick start (Windows 11)
1. Install Python 3.11.
2. Extract this ZIP and open `PrenatalAI` in VS Code.
3. In PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run frontend/app.py
```

The app runs in local demo mode without API keys or MongoDB.

## Optional local LLM
Install Ollama from https://ollama.com/download, then run:
```powershell
ollama pull llama3.2:1b
```
Set `LLM_PROVIDER=ollama` in `.env`. Without Ollama, the app uses local evidence snippets.

## Optional API
In another terminal:
```powershell
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000/docs.

## Research PDFs
Place trusted, legally obtained, non-identifiable PDFs in `research/`, then use the Research Library page to extract and index text. Verify source quality, publication details, and licensing.

## Current scope
Implemented: Streamlit, FastAPI, local retrieval, PDF extraction, basic safety smoke tests, local JSON case records.
Not implemented: a trained ultrasound classifier, Spina Bifida detection, Grad-CAM, production-grade RAG evaluation, OpenAI integration, clinical validation, or MongoDB UI persistence.

Do not claim disease detection until expert-labelled data, patient-independent evaluation, external validation, and qualified clinical review support that claim.
