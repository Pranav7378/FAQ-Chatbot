# Pranav Sai Portfolio FAQ Bot

A robust Retrieval-Augmented Generation (RAG) backend built using FastAPI and LangChain, integrating with Groq's high-performance LLM API. This backend provides the answers for a portfolio FAQ chat interface.

## Features
- **FastAPI**: Asynchronous, high-performance web framework.
- **LangChain & ChromaDB**: Chunking, embeddings, and vector search over a folder of `.txt` or `.pdf` files.
- **Groq Integration**: Incredibly fast inference utilizing `llama-3.1-8b-instant`.
- **Security**: Endpoint protected via `X-API-Key`.
- **Rate Limiting**: Prevent abuse using `slowapi` (IP-based rate limiting).
- **Render Ready**: Includes a `render.yaml` for 1-click deployments as a web service.

## Getting Started Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pranav7378/FAQ-Chatbot.git
   cd FAQ-Chatbot
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory:
   ```ini
   GROQ_API_KEY=your_groq_api_key
   API_SECRET_KEY=your_custom_secret_key_for_frontend
   ```

4. **Run the Server:**
   ```bash
   python api.py
   # Or using uvicorn directly:
   # uvicorn api:app --reload
   ```

## API Documentation
Once running, interactive API documentation is automatically provided by FastAPI at `http://localhost:8000/docs`.
