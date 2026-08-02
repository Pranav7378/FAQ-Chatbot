# NOVA Station FAQ Bot

A lightweight Retrieval-Augmented Generation (RAG) backend for **Pranav Sai's** portfolio, built with FastAPI and Groq.

**NOVA** is the central AI assistant aboard **NOVA Station**, a frontier AI research outpost where Commander Pranav Sai runs operations. NOVA briefs visitors about the Commander's professional record — synthesizing natural, conversational answers instead of echoing documents.

## How it works

```
Visitor question
      │
      ▼
 ask_rag(question)                 rag_engine.py
      │
      ├─ Guardrail: reject phone/sensitive requests before any retrieval
      ▼
 Search                             13 profile sections are embedded once at
      │                             startup (fastembed / all-MiniLM-L6-v2, ONNX).
      ▼                             Cosine top-k (k=3, threshold 0.25) + Jaccard dedupe.
 Retrieve relevant sections
      │
      ▼
 Context: phone number stripped from every injected chunk
      │
      ▼
 Groq (llama-3.3-70b-versatile)     System prompt = NOVA persona + guardrails
      │                             User prompt = dossier + question
      ▼
 Answer: phone numbers redacted from output too
```

## Features

- **Persona**: Sci-fi space-station assistant (NOVA Station) with strict contact guardrails — Gmail and LinkedIn only, phone numbers are never provided, confirmed, or repeated.
- **RAG**: Section-aware chunking of `data/pranav_profile.txt` + in-memory numpy cosine search. No ChromaDB, no LangChain, no heavy vector store.
- **Fast & free**: `fastembed` (ONNX, ~90 MB) for embeddings and Groq's free `llama-3.3-70b-versatile` for generation. Both are free — no paid APIs needed.
- **Free-tier friendly**: Tiny RAM footprint (fits Render's 512 MB), embedding model pre-warmed at build time so cold starts only load into memory.
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

This prints the parsed sections, retrieval scores, and a live answer for sample questions.

## API Usage

Interactive docs: `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your_api_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What experience does Pranav have?"}'
```

Response:

```json
{ "answer": "Commander Pranav is currently a Junior AI Engineer at Sita Corp in Hyderabad..." }
```

`test_api.py` also runs a quick check of auth + a valid request against a running server.

## Deploying to Render (free tier)

1. Create a new Web Service from this repo (or use `render.yaml`).
2. Set env vars in the dashboard: `GROQ_API_KEY` and `API_SECRET_KEY` (Render will prompt you for these; they're not committed).
3. The `buildCommand` pre-downloads the embedding model into `fastembed_cache/`, which ships in the deploy image — so free-tier spin-downs don't trigger a re-download; cold starts only reload the model into RAM.

Free-tier notes:
- The instance sleeps after ~15 min of inactivity; the first request after sleep has a few seconds of cold start.
- The `db/` directory is **not** used anymore; the knowledge base is re-parsed from `data/pranav_profile.txt` on every boot, so edits to the profile take effect on the next deploy.

## Updating the knowledge base

Edit `data/pranav_profile.txt` (keep the existing `----- SECTION TITLE -----` layout) and redeploy. NOVA will answer from the updated dossier.

## Troubleshooting

- **`401 invalid_api_key`** on every request → your `GROQ_API_KEY` is wrong or expired. Verify it with any Groq playground request.
- **`GROQ_API_KEY not found`** → the `.env` file is missing the key (or isn't being loaded).
- **Slow first request after deploy** → the embedding model may still be downloading; check the build logs that `fastembed_cache/` was populated.
- **Answers that say "doesn't cover that"** → the question falls below the retrieval threshold; try rephrasing, or the profile genuinely lacks that info.
