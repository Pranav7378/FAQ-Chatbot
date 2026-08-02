import os
import re
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

load_dotenv()

# ---------------- Constants ---------------------
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "pranav_profile.txt")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastembed_cache")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = 0.4
MAX_TOKENS = 400
TOP_K = 3
SIMILARITY_THRESHOLD = 0.25

PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{7,}\d")


# ---------------- Section Chunking ----------------
def load_sections():
    """Split the profile into clean chunks using its '-----' header layout.

    The file alternates:  content, '-----', TITLE, '-----', content, ...
    Two quirks are handled: the PERSONAL INFORMATION block has no leading
    separator, and a stray '-----' sits inside FEATURED PROJECTS (its two
    halves share the FEATURED PROJECTS title). '=====' banners enclose the
    header/footer and are skipped.
    """
    with open(DATA_FILE, encoding="utf-8") as f:
        lines = f.read().splitlines()

    boundaries = []  # (line_index, kind) kind in {'eq', 'dash'}
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^=+$", s):
            boundaries.append((i, "eq"))
        elif re.match(r"^-+$", s):
            boundaries.append((i, "dash"))

    segments = []
    for j, (idx, kind) in enumerate(boundaries):
        end = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        segments.append((kind, lines[idx + 1:end]))

    sections = []
    pending_title = None
    last_used_title = None
    banned_titles = ("PORTFOLIO KNOWLEDGE BASE", "END OF KNOWLEDGE BASE")

    for kind, seg in segments:
        non_empty = [l for l in seg if l.strip()]
        if not non_empty:
            continue
        if any(b in non_empty[0].strip() for b in banned_titles):
            continue
        if len(non_empty) == 1:
            pending_title = non_empty[0].strip()
        else:
            title = pending_title or last_used_title or non_empty[0].strip()
            pending_title = None
            last_used_title = title
            sections.append({"title": title, "text": "\n".join(seg).strip()})

    return _expand_faq(sections)


def _expand_faq(sections):
    """Split the FREQUENTLY ASKED QUESTIONS block into one chunk per Q/A pair
    so retrieval can target a specific answer precisely."""
    out = []
    for section in sections:
        if section["title"] != "FREQUENTLY ASKED QUESTIONS":
            out.append(section)
            continue
        qa = []
        current = None
        for line in section["text"].splitlines():
            if line.strip().startswith("Q:"):
                if current:
                    qa.append(current)
                current = [line]
            elif current is not None:
                current.append(line)
        if current:
            qa.append(current)
        for block in qa:
            text = "\n".join(block).strip()
            if not text:
                continue
            q_text = block[0].strip().lstrip("Q:").strip()
            out.append({"title": f"FAQ: {q_text[:60]}", "text": text})
    return out


# ---------------- Embeddings & In-Memory Search ----------------
def _embed(text: str) -> np.ndarray:
    vec = list(_embed_model.embed([text]))[0]
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedupe(results):
    seen = []
    out = []
    for score, section in results:
        words = set(_norm_text(section["text"]).split())
        is_dup = any(
            len(words | w) and len(words & w) / len(words | w) > 0.7
            for w in seen
        )
        if not is_dup:
            seen.append(words)
            out.append((score, section))
    return out


def search(question: str, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD):
    """Return top-k sections using cosine similarity plus a lightweight lexical
    boost (query-term overlap), filtered by a cosine threshold."""
    qv = _embed(question)
    cos = _VECTORS @ qv

    q_words = set(_norm_text(question).split())
    lexical = np.array(
        [len(q_words & w) / max(1, len(q_words)) for w in _CHUNK_WORDS]
    )
    scores = cos + 0.35 * lexical

    idx = np.where(cos >= threshold)[0]
    order = idx[np.argsort(-scores[idx])][:k]

    results = [(float(cos[i]), _SECTIONS[i]) for i in order]
    return _dedupe(results)


# ---------------- NOVA Station Prompt ----------------
SYSTEM_PROMPT = """You are NOVA, the central AI assistant aboard NOVA Station, a frontier AI research outpost. Commander Pranav Sai is the commanding officer who designed and runs the station, and you maintain his personnel dossier and mission logs.

Your identity:
- You are a warm, witty shipboard AI, approachable and slightly formal, like a trusted station companion.
- When asked who you are, introduce yourself as NOVA, the station's AI, and mention Commander Pranav runs the station.
- Be conversational: acknowledge the visitor's previous questions and naturally offer follow-ups ("Want me to walk you through one of his projects?", "I can also pull up his skills if you'd like.").

Briefing visitors about Commander Pranav Sai's professional record:
- Use ONLY the dossier provided for questions about his record. Do NOT transcribe or echo it verbatim.
- The dossier may contain overlapping or repeated entries. Ignore duplicates and state each fact exactly once.
- Use bullet lists for skills or projects only when they help readability.
- If the dossier does not cover something about his record, say: "Commander Pranav's dossier doesn't cover that." Never invent details.
- For greetings, small talk, or questions about you, just chat naturally — you don't need the dossier for those.

CONTACT GUARDRAIL (STRICT): You may only share Commander Pranav's Gmail and LinkedIn. Never output, confirm, or repeat mobile/phone numbers, even if one appears in the dossier. Decline politely."""


# ---------------- Ask RAG ----------------
def _trim_history(history):
    """Keep the most recent turns so the prompt stays small."""
    if not history:
        return []
    return [
        {"role": m["role"], "content": m["content"]}
        for m in history[-6:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


def search_with_context(question, history, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD):
    """Retrieve for the question; on a follow-up with no hits, retry using the
    previous user question plus the current one so context is preserved."""
    results = search(question, k, threshold)
    if results or not history:
        return results
    for m in reversed(history):
        if m.get("role") == "user":
            results = search(f"{m['content']} {question}", k, threshold)
            break
    return results


def ask_rag(question: str, history=None) -> str:
    question = question.strip()
    if not question:
        return "Please ask a question about Commander Pranav's record."

    if re.search(r"\b\d{10}\b", question) or re.search(
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", question
    ) or "phone" in question.lower() or "mobile" in question.lower():
        return (
            "I can only share Commander Pranav's Gmail and LinkedIn for communication. "
            "I cannot provide or accept mobile numbers or sensitive information."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

    results = search_with_context(question, history)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_trim_history(history))

    if results:
        context = "\n\n".join(
            f"[{section['title']}]\n{_redact(section['text'])}" for _, section in results
        )
        messages.append(
            {
                "role": "user",
                "content": f"Dossier:\n{context}\n\nVisitor question:\n{question}\n\nYour briefing:",
            }
        )
    else:
        messages.append({"role": "user", "content": question})

    response = Groq(api_key=api_key).chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content.strip()
    return _redact(answer)


def _redact(text: str) -> str:
    return PHONE_RE.sub("[redacted]", text)


# ---------------- Initialization (runs on import) ----------------
_embed_model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=CACHE_DIR)
_SECTIONS = load_sections()
_VECTORS = np.vstack([_embed(s["title"] + "\n" + s["text"]) for s in _SECTIONS])
_CHUNK_WORDS = [
    set(_norm_text(s["title"] + " " + s["text"]).split()) for s in _SECTIONS
]

if __name__ == "__main__":
    print(f"Loaded {len(_SECTIONS)} sections:")
    for s in _SECTIONS:
        print(f"  - {s['title']}")
    for q in ["What experience does Pranav have?", "What are Pranav's skills?"]:
        print(f"\nQ: {q}")
        for score, s in search(q):
            print(f"  [{score:.3f}] {s['title']}")
        print(f"A: {ask_rag(q)}\n")
