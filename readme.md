# PDF RAG Chat (Flask + Gemini + Chroma)

Upload multiple PDFs, ask questions about them, and get answers grounded in
the PDF content — with your previous questions/answers remembered and used
as context for follow-ups.

## How it works

- **Embeddings are local by default** (`sentence-transformers`, free,
  unlimited, no API calls) — see the section below if you want to switch to
  Gemini embeddings instead.
- **PDF upload** (`/upload`): each PDF is text-extracted page by page, split
  into overlapping chunks, embedded, and stored in a local persistent Chroma
  vector database (`chroma_db/`).
- **Ask** (`/ask`): your question is embedded, then used to semantically
  search two Chroma collections:
  1. `pdf_documents` — the PDF chunks (top 5 matches)
  2. `chat_history` — your past Q&A pairs (top 3 semantically relevant ones,
     not just the most recent) — this is what lets you ask things like
     "what did I ask about earlier?" or "explain that last answer more"
  Both are stuffed into a prompt sent to Gemini, which generates the answer.
  The new Q&A pair is then stored back into `chat_history` for future turns,
  and appended to a full chronological transcript (`chat_history.json`) that
  the UI loads on page refresh.
- **Reset** (`/reset`): wipes both collections and the transcript.

## Free embeddings vs. Gemini embeddings (rate limits)

By default this app embeds text **locally** using `sentence-transformers`
(model: `all-MiniLM-L6-v2`) — completely free, no API calls, no rate limit,
runs on your CPU. Only answer *generation* uses the Gemini API. The first
run will download the small model file (~90MB) from Hugging Face.

If you'd rather use Gemini's embedding model instead (e.g. for its stronger
multilingual quality), set in `.env`:
```
EMBEDDING_PROVIDER=gemini
```
Gemini's free tier caps `embed_content` at roughly 100 requests/minute. The
app batches chunks (20 per request) and auto-retries on 429s, but embedding
many/large PDFs on the free tier will still be slow or may need billing
enabled. Local embeddings avoid this entirely, so it's the recommended
default unless you have a specific reason to use Gemini's.

## Setup

```bash
cd rag_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a Gemini API key from https://aistudio.google.com/apikey, then:

```bash
cp .env.example .env
# edit .env and paste your key into GEMINI_API_KEY=
```

## Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Notes / things you may want to tweak

- **Models**: defaults are local `all-MiniLM-L6-v2` (embeddings, override via
  `LOCAL_EMBED_MODEL`) and `gemini-2.5-flash` (generation, override via
  `GEMINI_GEN_MODEL`). If you switch `EMBEDDING_PROVIDER=gemini`, that uses
  `models/gemini-embedding-001` by default (override via `GEMINI_EMBED_MODEL`).
  Google retires Gemini model IDs fairly often (e.g. `text-embedding-004`
  and `gemini-2.0-flash` were both shut down in mid-2026) — if you ever get
  a `404 ... is not found for API version v1beta` error, it means the model
  string is retired; check https://ai.google.dev/gemini-api/docs/models and
  https://ai.google.dev/gemini-api/docs/deprecations for the current names
  and swap them into `.env`, no code changes needed. As of this writing,
  `gemini-2.5-flash` is scheduled to shut down Oct 16, 2026 — its
  replacement is `gemini-3.5-flash` (pricier) or `gemini-3.1-flash-lite`
  (cheaper, lower-capability).
- **Chunking**: 800 characters per chunk with 150 character overlap
  (`CHUNK_SIZE` / `CHUNK_OVERLAP` in `app.py`). Increase chunk size for
  denser technical PDFs, decrease for better retrieval granularity.
- **Storage**: everything is local — `uploads/` (raw PDFs), `chroma_db/`
  (vector store), `chat_history.json` (transcript). Delete these folders/
  file any time to fully reset, or use the "Clear All Data" button in the UI.
- **Rate limits**: with local embeddings there's no embedding rate limit at
  all. Each question still makes one Gemini generation call, so keep an eye
  on your Gemini API quota if you switch that model too.
- This is a single-user local app (no auth, no multi-session isolation). If
  you need multiple concurrent users with separate histories, you'd want to
  key the Chroma collections and history file by a session/user ID.
