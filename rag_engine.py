import os
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------------- Constants ---------------------
DATA_DIR = "data"
DB_DIR = "db"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Document Loading ----------------
def load_documents():
    """Load all txt files from data directory into LangChain documents."""
    print("Loading documents from data folder...")
    documents = []
    
    # Get all txt files in data directory
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No txt files found in {DATA_DIR} directory!")
        # Fallback to text files if no txts found
        txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
        if not txt_files:
            print("No documents found in data folder!")
            return documents
        txt_files = txt_files
    
    print(f"Found {len(txt_files)} document(s): {txt_files}")
    
    for file in txt_files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            print(f"Loading {file}...")
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            
            file_docs = loader.load()
            documents.extend(file_docs)
            print(f"Successfully loaded {file} with {len(file_docs)} pages")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

# ---------------- Build Vector DB ----------------
def build_vector_db():
    print("Building or loading vector database...")
    docs = load_documents()
    
    if not docs:
        print("No documents found to process!")
        return None
    
    # Chunking parameters are decent, keep them for now.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        print(f"chunks: {chunk.page_content[:100]}...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ.get("HF_API_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # In a real scenario, you'd check if DB exists before rebuilding
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    db.persist()
    print(f"Vector database built with {len(chunks)} chunks.")
    return db

# ---------------- Groq LLM Setup ----------------
def setup_groq_llm():
    print(f"Loading Groq model...")
    start_time = time.time()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        exit(1)

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=api_key
    )
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return llm

# ---------------- Initialize RAG Components ----------------

# Check if DB directory is populated (simple persistence check)
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    print("Building new vector database from txt files...")
    db = build_vector_db()
else:
    print("Loading existing vector database...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ.get("HF_API_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

if db is None:
    print("Failed to initialize database. Please check if there are txt files in the data folder.")
    exit(1)

retriever = db.as_retriever(search_kwargs={"k": 5})
groq_llm = setup_groq_llm()

# ---------------- Prompt Template ----------------
RAG_PROMPT = """
You are an expert assistant for answering questions about Pranav Sai's portfolio and background.
Answer the user's question using ONLY the provided context from his documents.
- Give bullet points for skills or projects when appropriate
- Summarize education and experience briefly
- Be factual and concise
- If the context doesn't contain relevant information, say "I don't have enough information about that in the provided documents."
- Do NOT invent information

Context:
{context}

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(
    template=RAG_PROMPT,
    input_variables=["context", "question"]
)

# ---------------- Ask RAG Function ----------------
def ask_rag(question: str) -> str:
    print(f"\n--- Processing Question: {question} ---")
    # Retrieve top 5 relevant chunks
    context_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in context_docs])
    print(f"Retrieved {len(context_docs)} relevant context chunk(s).")

    # Prepare input for FLAN-T5
    final_input = {"context": context, "question": question}

    # Generate answer
    chain = prompt_template | groq_llm
    # LangChain's invoke handles the LLM call
    answer = chain.invoke(final_input)

    # Clean answer
    if hasattr(answer, "content"):
        return answer.content.strip()
    elif isinstance(answer, str):
        cleaned_answer = answer.split("Answer:")[-1].strip()
        return cleaned_answer if cleaned_answer else answer.strip()
    else:
        return str(answer).strip()
    
if __name__ == "__main__":
    # Test if the RAG system works
    test_question = "What are Pranav's skills?"
    answer = ask_rag(test_question)
    print(f"Q: {test_question}")
    print(f"A: {answer}")

