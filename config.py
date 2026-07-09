import os
from dotenv import load_dotenv

# --------------------------------------------------
# Base Directory
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# --------------------------------------------------
# Gemini API
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found.\n"
        "Create a .env file and add:\n"
        "GEMINI_API_KEY=your_api_key"
    )

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "gemini-2.5-flash"
)

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "local"
).lower()

LOCAL_EMBED_MODEL = os.getenv(
    "LOCAL_EMBED_MODEL",
    "all-MiniLM-L6-v2"
)

GEMINI_EMBED_MODEL = os.getenv(
    "GEMINI_EMBED_MODEL",
    "models/gemini-embedding-001"
)

# --------------------------------------------------
# Flask Configuration
# --------------------------------------------------
SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "rag-secret-key"
)

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# --------------------------------------------------
# Project Paths
# --------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

# --------------------------------------------------
# Create Required Directories
# --------------------------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --------------------------------------------------
# PDF Processing
# --------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf"}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

MAX_CONTEXT_CHARS = 14000

# --------------------------------------------------
# Chroma Collections
# --------------------------------------------------
DOCUMENT_COLLECTION = "pdf_documents"
HISTORY_COLLECTION = "chat_history"

# --------------------------------------------------
# Gemini Generation Parameters
# --------------------------------------------------
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 40
MAX_OUTPUT_TOKENS = 2048

# --------------------------------------------------
# Logging
# --------------------------------------------------
LOG_LEVEL = "INFO"

# --------------------------------------------------
# Application Name
# --------------------------------------------------
APP_NAME = "AI RAG Chatbot"
APP_VERSION = "1.0.0"