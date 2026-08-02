# NOVA Station FAQ Bot

Hey! This is the backend for my portfolio FAQ bot. You ask about my work, experience, skills, projects — and **NOVA** answers as my AI Portfolio Assistant with a "NOVA Station" space-station theme.

It's a small RAG setup: I wrote my portfolio into `data/pranav_profile.txt`, split it into clean sections, embed them once at startup, and retrieve the most relevant bits for each question before asking Groq to phrase an answer in NOVA's voice.

## The vibe

- **NOVA** is my AI Portfolio Assistant, styled after a futuristic station AI. **NOVA Station** is a *creative interface* — a virtual station representing my engineering journey — not a real place.
  - Projects = "missions", Blogs = "research logs", Architecture diagrams = "blueprints", Certificates = "credentials", Experience = "career timeline".
- She's friendly and conversational (think JARVIS meets a portfolio assistant), but always truthful and grounded — she answers from my portfolio and never exaggerates or invents achievements.
- She's **context-aware**: she remembers the conversation, so she can answer "who are u?", greet visitors with suggested questions, and follow up on what they were just asking about.
- She's strict about contact info: only my Gmail and LinkedIn. Phone numbers never get shared, even if they're in the file.

## Tech stack

- **FastAPI** + **Groq** (`llama-3.3-70b-versatile`) — both free to use.
- **fastembed** (`all-MiniLM-L6-v2`) for embeddings — tiny ONNX model, runs anywhere.
- **numpy** cosine search instead of a heavy vector DB — the profile is one small file, no need for ChromaDB.
- Built for **Render's free tier** — fits in 512 MB, model is pre-warmed at build time so cold starts are quick.

## Run it locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```ini
GROQ_API_KEY=your_groq_api_key
API_SECRET_KEY=your_secret_key_for_the_api
GROQ_MODEL=llama-3.3-70b-versatile   # optional
```

Then:

```bash
python api.py
# or: uvicorn api:app --reload
```

First run downloads the embedding model into `fastembed_cache/` (git-ignored).

## Try it

```bash
python rag_engine.py        # runs the RAG engine directly
```

Or hit the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What experience does Pranav have?"}'
```

Docs are at `http://localhost:8000/docs` once it's running.

### Conversation memory

Send a `session_id` to keep the conversation on the server, or pass the full `history` yourself (stateless, survives restarts):

```json
{
  "question": "tell me more about that one",
  "session_id": "abc123"
}
```

or

```json
{
  "question": "tell me more about that one",
  "history": [
    {"role": "user", "content": "What projects has Pranav built?"},
    {"role": "assistant", "content": "He built a Crop Recommendation System and a Movie Recommendation System."}
  ]
}
```

The response includes the `session_id` you can reuse for the next turn. If a follow-up question doesn't match anything on its own, NOVA reuses your earlier question to find the right context.

## Deploying

`render.yaml` is included — point a Render web service at this repo, set the two env vars, done. On free tier it sleeps after ~15 min idle; the first request after that takes a few extra seconds to wake up.

## Updating my profile

Edit `data/pranav_profile.txt` (keep the `----- SECTION TITLE -----` format) and redeploy. NOVA reads it fresh on every boot.

## Troubleshooting

- **`401 invalid_api_key`** — the `GROQ_API_KEY` is wrong or expired. Grab a new one from the Groq console.
- **`GROQ_API_KEY not found`** — `.env` is missing or not being loaded.
- **"I don't currently have enough information to answer that accurately."** — the question didn't beat the similarity threshold (or the profile really doesn't mention it). Try rephrasing.
