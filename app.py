import os
import uuid
import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction, SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from dotenv import load_dotenv

# Resolve paths relative to this file, not the current working directory,
# so `python app.py` works the same whether you run it from this folder
# or from somewhere else.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        f"Expected to find it in: {DOTENV_PATH}\n"
        "Fix: copy .env.example to .env in that same folder, then edit .env "
        "and set GEMINI_API_KEY=your_actual_key (no quotes, no spaces around "
        "the '='). Get a key at https://aistudio.google.com/apikey\n"
        "Then restart the app."
    )

genai.configure(api_key=GEMINI_API_KEY)

# EMBEDDING_PROVIDER controls how documents/questions get embedded:
#   "local"  -> sentence-transformers, runs on your machine, free, no rate limit (default)
#   "gemini" -> Google's embedding API (subject to free-tier rate limits; needs billing for volume)
# Generation (answering questions) always uses Gemini either way.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local").lower()

# You can swap these for newer Gemini model names later without touching any
# other code - just change the env vars (or the defaults below).
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
GEN_MODEL = os.environ.get("GEMINI_GEN_MODEL", "gemini-2.5-flash")
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks so context isn't cut mid-thought

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB total per request


# ---------------------------------------------------------------------------
# Gemini embedding function wired into Chroma (batched + retried so the
# free-tier rate limit of ~100 requests/min doesn't get blown through by
# embedding one chunk at a time)
# ---------------------------------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Wraps Gemini's embedding API so Chroma can use it directly."""

    def __init__(self, task_type="retrieval_document", batch_size=20):
        self.task_type = task_type
        self.batch_size = batch_size

    def _embed_batch(self, batch, retries=3):
        import time
        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=EMBED_MODEL,
                    content=batch,  # list -> one API call embeds the whole batch
                    task_type=self.task_type,
                    output_dimensionality=768,
                )
                # genai returns {'embedding': [...]} for a single string and
                # {'embedding': [[...], [...]]} for a list - normalize both.
                emb = result["embedding"]
                if batch and len(batch) == 1 and emb and not isinstance(emb[0], list):
                    emb = [emb]
                return emb
            except Exception as e:
                msg = str(e)
                if "429" in msg and attempt < retries - 1:
                    wait = 32  # free tier resets roughly every 30-60s
                    print(f"Rate limited, waiting {wait}s before retry ({attempt + 1}/{retries})...")
                    time.sleep(wait)
                    continue
                print(f"Embedding error: {e}")
                return [[0.0] * 768 for _ in batch]
        return [[0.0] * 768 for _ in batch]

    def __call__(self, input):
        embeddings = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings


def build_embedding_function():
    if EMBEDDING_PROVIDER == "gemini":
        print(f"Using Gemini embeddings ({EMBED_MODEL}) - subject to API rate limits.")
        return GeminiEmbeddingFunction(task_type="retrieval_document")
    print(f"Using local sentence-transformers embeddings ({LOCAL_EMBED_MODEL}) - "
          f"free, unlimited, runs on your machine. First run downloads the model.")
    return SentenceTransformerEmbeddingFunction(model_name=LOCAL_EMBED_MODEL)


# Chroma requires ONE embedding function per collection. We use the same
# function for both writing documents and querying - fine for both providers
# in practice, since asymmetric query/document modes give only a small boost.
embed_fn = build_embedding_function()

# ---------------------------------------------------------------------------
# Chroma persistent client + collections
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

docs_collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    embedding_function=embed_fn,
)

history_collection = chroma_client.get_or_create_collection(
    name="chat_history",
    embedding_function=embed_fn,
)


def load_full_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_full_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Words that signal "give me the whole picture" rather than "find one specific
# fact" - these questions need broad document coverage, not narrow similarity
# search, or the model only ever sees a handful of the most-similar chunks.
SUMMARY_KEYWORDS = [
    "summary", "summarize", "summarise", "overview", "overall", "entire",
    "whole thesis", "whole document", "whole paper", "full thesis",
    "what is this document about", "what is this thesis about",
    "tell me about this document", "tell me about this thesis",
    "200 word", "300 word", "500 word", "word para", "words para",
]


def is_broad_request(question):
    q = question.lower()
    return any(kw in q for kw in SUMMARY_KEYWORDS)


def get_uploaded_filenames():
    try:
        all_docs = docs_collection.get(include=["metadatas"])
        sources = {m.get("source") for m in all_docs.get("metadatas", []) if m.get("source")}
        return sorted(sources)
    except Exception as e:
        print(f"Could not list uploaded filenames: {e}")
        return []


def find_target_filename(question, filenames):
    q = question.lower()
    for fname in filenames:
        stem = os.path.splitext(fname)[0].lower().replace("_", " ").replace("-", " ")
        if fname.lower() in q or stem in q:
            return fname
    # If there's exactly one document uploaded, assume broad questions mean that one.
    if len(filenames) == 1:
        return filenames[0]
    return None


MAX_BROAD_CONTEXT_CHARS = 14000  # keeps the prompt a reasonable size


def get_broad_document_context(filename):
    """Pulls chunks from across the WHOLE document (ordered by page), evenly
    sampling if it's too large to fit, so a summary reflects the full thesis
    rather than just whatever 5 chunks happened to match the search words."""
    try:
        result = docs_collection.get(
            where={"source": filename},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        print(f"Broad context retrieval error: {e}")
        return ""

    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    if not docs:
        return ""

    pairs = sorted(zip(metas, docs), key=lambda pm: pm[0].get("page", 0))

    total_chars = sum(len(d) for _, d in pairs)
    if total_chars <= MAX_BROAD_CONTEXT_CHARS:
        selected = pairs
    else:
        # Evenly sample chunks across the whole document so the beginning,
        # middle, and end are all represented, rather than just page 1 onward.
        budget_chunks = max(1, MAX_BROAD_CONTEXT_CHARS // CHUNK_SIZE)
        step = max(1, len(pairs) // budget_chunks)
        selected = pairs[::step]

    context = ""
    for meta, doc in selected:
        page = meta.get("page", "?")
        context += f"\n[Source: {filename}, page {page}]\n{doc}\n"
        if len(context) >= MAX_BROAD_CONTEXT_CHARS:
            break
    return context


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    total_chunks = 0
    processed_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            processed_files.append({"filename": file.filename, "error": "not a PDF, skipped"})
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            pages = extract_pdf_text(filepath)
        except Exception as e:
            processed_files.append({"filename": filename, "error": f"could not read PDF: {e}"})
            continue

        ids, texts, metadatas = [], [], []
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                ids.append(str(uuid.uuid4()))
                texts.append(chunk)
                metadatas.append({"source": filename, "page": page_num})

        if texts:
            docs_collection.add(ids=ids, documents=texts, metadatas=metadatas)
            total_chunks += len(texts)

        processed_files.append({"filename": filename, "pages": len(pages), "chunks": len(texts)})

    return jsonify({
        "message": "Files processed",
        "files": processed_files,
        "total_chunks_added": total_chunks,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # 1. Retrieve relevant document chunks.
    # Broad/summary-style questions ("summarize this thesis", "200 word para")
    # get whole-document coverage instead of narrow top-5 similarity search,
    # since a handful of "most similar to the question text" chunks is a poor
    # match for "give me the whole picture" requests.
    doc_context = ""
    used_broad_context = False
    try:
        if docs_collection.count() > 0:
            if is_broad_request(question):
                filenames = get_uploaded_filenames()
                target = find_target_filename(question, filenames)
                if target:
                    doc_context = get_broad_document_context(target)
                    used_broad_context = bool(doc_context)

            if not doc_context:
                doc_results = docs_collection.query(query_texts=[question], n_results=8)
                chunks = doc_results.get("documents", [[]])[0]
                metas = doc_results.get("metadatas", [[]])[0]
                for chunk, meta in zip(chunks, metas):
                    source = meta.get("source", "unknown")
                    page = meta.get("page", "?")
                    doc_context += f"\n[Source: {source}, page {page}]\n{chunk}\n"
    except Exception as e:
        print(f"Doc retrieval error: {e}")

    # 2. Retrieve semantically relevant past Q&A (not just the latest ones)
    history_context = ""
    try:
        hist_count = history_collection.count()
        if hist_count > 0:
            hist_results = history_collection.query(
                query_texts=[question], n_results=min(3, hist_count)
            )
            hist_metas = hist_results.get("metadatas", [[]])[0]
            for meta in hist_metas:
                q = meta.get("question", "")
                a = meta.get("answer", "")
                history_context += f"\nQ: {q}\nA: {a}\n"
    except Exception as e:
        print(f"History retrieval error: {e}")

    coverage_note = (
        "This context was sampled across the whole document to give broad coverage."
        if used_broad_context else
        "This context was retrieved based on similarity to the question, so it may "
        "only cover a few relevant passages rather than the whole document."
    )

    prompt = f"""You are a helpful assistant answering questions using the provided document context and prior conversation history.

DOCUMENT CONTEXT ({coverage_note}):
{doc_context if doc_context else "No relevant document context found."}

RELEVANT PRIOR CONVERSATION:
{history_context if history_context else "No relevant prior conversation found."}

CURRENT QUESTION:
{question}

Instructions:
- Do your best to fully answer the question using the context above, including honoring any requested length or format (e.g. "200 word paragraph").
- If the context only covers part of the document, synthesize the best answer you can from what's there, and briefly note that it's based on partial coverage rather than refusing to answer.
- If the question refers to something discussed earlier (e.g. "what did I ask before"), use the prior conversation section.
- Only say the information isn't available if the context truly has nothing relevant to draw on.

Answer:"""

    try:
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        return jsonify({"error": f"Gemini generation error: {e}"}), 500

    # 3. Store this Q&A into the semantic vector history (for future retrieval)
    qa_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    try:
        history_collection.add(
            ids=[qa_id],
            documents=[question],
            metadatas=[{"question": question, "answer": answer, "timestamp": timestamp}],
        )
    except Exception as e:
        print(f"History store error: {e}")

    # 4. Append to the full chronological transcript (for displaying chat history in UI)
    full_history = load_full_history()
    full_history.append({
        "id": qa_id,
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
    })
    save_full_history(full_history)

    return jsonify({
        "answer": answer,
        "used_doc_context": bool(doc_context),
        "used_broad_context": used_broad_context,
        "used_history_context": bool(history_context),
    })


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(load_full_history())


@app.route("/reset", methods=["POST"])
def reset():
    """Clears uploaded documents AND chat history (vector store + transcript)."""
    global docs_collection, history_collection
    try:
        chroma_client.delete_collection("pdf_documents")
    except Exception:
        pass
    try:
        chroma_client.delete_collection("chat_history")
    except Exception:
        pass
    docs_collection = chroma_client.get_or_create_collection(
        name="pdf_documents", embedding_function=embed_fn
    )
    history_collection = chroma_client.get_or_create_collection(
        name="chat_history", embedding_function=embed_fn
    )
    save_full_history([])
    return jsonify({"message": "All documents and chat history cleared."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)