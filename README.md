# Voidware RAG — Self-hosted API Documentation Chatbot

A production RAG (Retrieval-Augmented Generation) chatbot that answers questions about your API documentation. Built on a budget: free embeddings, free LLM via Google Colab, free reranking tier. No OpenAI required.

**Live demo:** https://ai.vapevoidware.xyz

---

## What it does

You write markdown documentation for your APIs. The system chunks it, embeds it into a vector database, and lets you ask natural language questions like *"What endpoints does the deploy API have?"* or *"How does authentication work?"* — and get accurate, sourced answers.

## Stack

| Component | Tool | Cost |
|---|---|---|
| Embeddings | `nomic-embed-text` via Ollama | Free |
| LLM | `llama3.2:3b` via Ollama | Free |
| GPU runtime | Google Colab T4 | Free |
| Public tunnel | ngrok | Free (1 tunnel) |
| Vector store | pgvector (PostgreSQL extension) | Free |
| Reranker | Cohere `rerank-english-v3.0` | Free (5k calls/month) |
| Framework | LangChain text splitter | Free |
| API layer | FastAPI | Free |

---

## Architecture

```
User → ai_api (public, port 1463)
           │
           ├── /          → Chat interface (HTML)
           ├── /chat      → Proxies to rag_api
           ├── /ingest    → Proxies to rag_api
           ├── /files     → Manage .md documentation files
           └── /config    → Update Ollama ngrok URL
                │
                ▼
         rag_api (localhost only, port 1464)
                │
                ├── Ollama (Google Colab via ngrok)
                │     ├── nomic-embed-text  → embeddings
                │     └── llama3.2:3b       → generation
                │
                ├── pgvector (PostgreSQL)   → vector store
                │
                └── Cohere API             → reranking
```

**RAG pipeline per query:**
1. Embed query → nomic-embed-text
2. Cosine similarity search → top-12 chunks from pgvector
3. Rerank → Cohere keeps top-5
4. Build prompt with context chunks
5. Generate → llama3.2:3b
6. Return answer + source filenames

---

## Prerequisites

- A Linux VPS (tested on Debian 12)
- PostgreSQL installed and running
- Python 3.11+ with a virtual environment
- A Google account (for Colab)
- A free [ngrok account](https://ngrok.com)
- A free [Cohere account](https://cohere.com)

---

## Setup

### 1. Install pgvector on your VPS

```bash
sudo apt-get install -y postgresql-16-pgvector
sudo -u postgres psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Verify:
```bash
sudo -u postgres psql -d your_database -c "\dx vector"
```

### 2. Clone and configure

```bash
git clone https://github.com/Erchodev/voidware-rag
cd voidware-rag
```

Edit the constants at the top of `rag_api/main.py`:

```python
COHERE_API_KEY = "your_cohere_key_here"
PG_DSN         = "postgresql://user:password@localhost/your_database"
```

Edit `ai_api/main.py`:

```python
ADMIN_KEY = "your_chosen_admin_key"
```

### 3. Install dependencies

```bash
source /path/to/your/venv/bin/activate
pip install -r rag_api/requirements.txt
pip install -r ai_api/requirements.txt
```

### 4. Start the services

```bash
# rag_api (internal only)
uvicorn main:app --host 0.0.0.0 --port 1464 --app-dir rag_api

# ai_api (public)
uvicorn main:app --host 0.0.0.0 --port 1463 --app-dir ai_api
```

Point a reverse proxy (nginx, Cloudflare Tunnel, etc.) at port 1463.

### 5. Start Llama on Google Colab

Open a new Colab notebook at [colab.research.google.com](https://colab.research.google.com).

**Runtime → Change runtime type → T4 GPU**

Then run these cells in order:

```python
# Cell 1 — Install Ollama
!sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
```

```python
# Cell 2 — Start Ollama
import subprocess, time, os

subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
time.sleep(2)

env = os.environ.copy()
env["OLLAMA_ORIGINS"] = "*"
env["OLLAMA_HOST"] = "0.0.0.0"
subprocess.Popen(["ollama", "serve"], env=env)
time.sleep(5)
print("Ollama started")
```

```python
# Cell 3 — Pull models (takes 2-5 min)
!ollama pull llama3.2:3b
!ollama pull nomic-embed-text
```

```python
# Cell 4 — Expose via ngrok
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.kill()

ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")  # from dashboard.ngrok.com
tunnel = ngrok.connect(11434)
print("Ollama URL:", tunnel.public_url)
```

Copy the printed URL (e.g. `https://xxxx.ngrok-free.app`).

### 6. Connect VPS to Colab

```bash
curl -X POST http://localhost:1464/config \
  -H "Content-Type: application/json" \
  -d '{"ollama_url": "https://xxxx.ngrok-free.app"}'
```

### 7. Add your documentation

Place markdown files in `ai_api/files/`. One file per API/service, consistent format works best. Or upload via the API:

```bash
curl -X PUT http://localhost:1463/files/my_api.md \
  -H "x-admin-key: your_admin_key" \
  -H "Content-Type: application/json" \
  -d '{"content": "# My API\n\n## Overview\n..."}'
```

### 8. Ingest

```bash
curl -X POST http://localhost:1463/ingest \
  -H "x-admin-key: your_admin_key"
```

This will take a few minutes depending on how many files you have. Vectors are stored in PostgreSQL and persist across restarts — you only need to re-ingest when documentation changes.

### 9. Chat

Open `https://your-domain`, enter your admin key, and start asking questions.

---

## API Reference

### ai_api (public)

All endpoints except `/` require `x-admin-key` header.

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web chat interface |
| GET | `/health` | Service status |
| POST | `/chat` | `{ query }` → `{ answer, sources, chunks_used }` |
| POST | `/ingest` | Trigger full re-ingest of all docs |
| POST | `/config` | `{ ollama_url }` → update ngrok URL at runtime |
| GET | `/files` | List documentation files |
| GET | `/files/{name}` | Read a file |
| PUT | `/files/{name}` | `{ content }` → create/update a file |
| DELETE | `/files/{name}` | Delete a file |

### rag_api (localhost only)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status + chunk count + current Ollama URL |
| GET | `/db-stats` | pgvector table stats (for monitoring) |
| POST | `/chat` | `{ query }` → full RAG pipeline |
| POST | `/ingest` | Chunk + embed + store all docs |
| POST | `/config` | `{ ollama_url }` → update at runtime |

---

## Documentation format

Each file in `ai_api/files/` should follow this structure for best results:

```markdown
# service_name

## Overview
What this service does, tech stack, port, base URL, auth method.

## Endpoints

### GET /endpoint
Description, query params, auth, request body, response shape.

### POST /endpoint
...
```

Consistent structure dramatically improves retrieval quality because the chunker splits on `##` headers.

---

## When Colab restarts

Colab sessions time out after ~12 hours of inactivity. When that happens:

1. Re-run all 4 cells in your notebook
2. Copy the new ngrok URL
3. Update rag_api: `POST /config` with the new URL

Your vectors in PostgreSQL are unaffected — no need to re-ingest.

---

## Limitations

- **Colab session timeout** — not suitable for 24/7 production without a paid GPU or self-hosted Ollama
- **3B model quality** — llama3.2:3b is fast but struggles with complex multi-hop questions; upgrade to 8B on Colab Pro or use OpenAI/Anthropic API for better quality
- **ngrok free tier** — 1 tunnel, URL changes on restart, limited bandwidth
- **Reranker** — Cohere free tier allows 5,000 rerank calls/month

---

## Upgrading to OpenAI (optional)

If you want better quality without managing Colab, swap the LLM and embedding calls in `rag_api/main.py`:

```python
# Replace Ollama embed with OpenAI
from openai import AsyncOpenAI
oai = AsyncOpenAI(api_key="your_key")

async def embed(text):
    r = await oai.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding  # 1536-dim — update EMBED_DIM and recreate table

async def llm(prompt):
    r = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content
```

Note: pgvector table dimension must match the embedding model. Recreate the table if switching models.

---

## License

MIT