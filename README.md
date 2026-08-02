# NOVA Station FAQ Bot

A lightweight Retrieval-Augmented Generation (RAG) backend for **Pranav Sai's** portfolio, built with FastAPI and Groq.

**NOVA** is the central AI assistant aboard **NOVA Station**, where Commander Pranav Sai runs operations. NOVA briefs visitors about the Commander's professional record — synthesizing natural answers instead of echoing documents.

## Features

- **Persona**: Sci-fi space-station assistant personality with strict contact guardrails (Gmail + LinkedIn only, never phone numbers).
- **RAG**: Section-aware chunking of `data/pranav_profile.txt` + in-memory numpy cosine search (no ChromaDB, no heavy vector store).
- **Fast & free**: `fastembed` (ONNX, ~90 MB) for embeddings and Groq's free `llama-3.3-70b-versatile` for generation.
- **Free-tier friendly**: Tiny RAM footprint, embeddings pre-warmed at build time so cold starts stay fast.
- **Security**: `X-API-Key` auth + IP-based rate limiting (10 req/min).
- **Render ready**: Ships with `render.yaml`.

## Getting Started Locally

1. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables** (`.env`):
   ```ini
   GROQ_API_KEY=your_groq_api_key
   API_SECRET_KEY=your_custom_secret_key_for_frontend
   GROQ_MODEL=llama-3.3-70b-versatile   # optional override
   ```

3. **Run the Server:**
   ```bash
   python api.py
   # or: uvicorn api:app --reload
   ```

The first run downloads the embedding model into `fastembed_cache/` (git-ignored).

## Test the RAG engine directly

```bash
python rag_engine.py
```

## API Documentation

Once running, interactive API docs are available at `http://localhost:8000/docs`.
