import os
from collections import defaultdict, deque
from uuid import uuid4
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from rag_engine import ask_rag

# Load environment variables
load_dotenv()

# Setup Rate Limiting
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(title="Pranav Sai Portfolio API", description="NOVA FAQ Bot API for Portfolio", version="1.0.0")

# Add SlowAPI rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup CORS - In production, replace "*" with your Lovable app's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup API Key Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_key = os.getenv("API_SECRET_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API_SECRET_KEY not configured on server")
    if api_key_header != expected_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: Optional[List[dict]] = None

class AnswerResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None

# In-memory conversation store (per session, capped at ~6 turns).
# Client-supplied `history` is used when present, so it also survives restarts.
SESSION_STORE = defaultdict(lambda: deque(maxlen=12))

def _get_history(body: QuestionRequest) -> list:
    if body.history is not None:
        return body.history
    if body.session_id:
        return list(SESSION_STORE[body.session_id])
    return []

@app.get("/")
def read_root():
    return {"message": "Welcome to NOVA Station — Pranav Sai's AI Portfolio Assistant API. POST /chat to ask about his experience, projects, skills, and more."}

@app.post("/chat", response_model=AnswerResponse)
@limiter.limit("10/minute") # Allow max 10 questions per minute per IP
async def chat_endpoint(request: Request, body: QuestionRequest, api_key: str = Depends(get_api_key)):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        history = _get_history(body)
        answer = await run_in_threadpool(ask_rag, body.question, history)

        session_id = body.session_id or uuid4().hex
        SESSION_STORE[session_id].append({"role": "user", "content": body.question})
        SESSION_STORE[session_id].append({"role": "assistant", "content": answer})

        return {"answer": answer, "session_id": session_id}
    except Exception as e:
        # In a real app you might want to log this error
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
