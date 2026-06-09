import os

# Optional: load environment from a .env file if python-dotenv is installed.
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Vector store ---
CHROMA_COLLECTION = "syllabus"
PROJECT_ROOT = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

# --- Retrieval ---
N_RESULTS = 8

# --- Documents ---
DOCS_PATH = os.path.join(PROJECT_ROOT, "documents")
