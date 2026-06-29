from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2
import httpx
import cohere
import json
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag_api")

app = FastAPI()

# ── Config — fill these in before running ────────────────────────────────────
COHERE_API_KEY = "your_cohere_api_key_here"
PG_DSN         = "postgresql://user:password@localhost/your_database"
FILES_DIR      = Path("../ai_api/files")   # folder containing your .md docs
CONFIG_FILE    = Path("./runtime_config.json")
EMBED_MODEL    = "nomic-embed-text"
LLM_MODEL      = "llama3.2:3b"
EMBED_DIM      = 768
TOP_K          = 12
TOP_N          = 5
CHUNK_SIZE     = 1500
CHUNK_OVERLAP  = 200

co = cohere.ClientV2(COHERE_API_KEY)


def load_cfg():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"ollama_url": "http://localhost:11434"}

def save_cfg(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def ollama_url():
    return load_cfg()["ollama_url"]


def get_conn():
    conn = psycopg2.connect(PG_DSN)
    conn.set_client_encoding("UTF8")
    return conn

def setup_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id           SERIAL PRIMARY KEY,
            filename     TEXT NOT NULL,
            chunk_index  INTEGER NOT NULL,
            content      TEXT NOT NULL,
            embedding    vector({EMBED_DIM})
        );
    """)
    conn.commit()
    try:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx "
            "ON rag_chunks USING hnsw (embedding vector_cosine_ops);"
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.warning("Could not create HNSW index: %s", e)
    cur.close()
    conn.close()

@app.on_event("startup")
def startup():
    setup_db()
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rag_chunks;")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        if count == 0:
            log.warning("rag_chunks is EMPTY — POST /ingest to populate.")
        else:
            log.info("rag_api ready — %s chunks in rag_chunks.", count)
    except Exception as e:
        log.error("Startup DB check failed: %s", e)


OLLAMA_HEADERS = {"ngrok-skip-browser-warning": "true"}

async def embed(text):
    url = ollama_url()
    async with httpx.AsyncClient(timeout=120, verify=False) as client:
        r = await client.post(
            f"{url}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            headers=OLLAMA_HEADERS
        )
        r.raise_for_status()
        return r.json()["embedding"]

async def llm(prompt):
    url = ollama_url()
    async with httpx.AsyncClient(timeout=180, verify=False) as client:
        r = await client.post(
            f"{url}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_ctx": 8192}},
            headers=OLLAMA_HEADERS
        )
        r.raise_for_status()
        return r.json()["response"]

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    return splitter.split_text(text)


class ChatRequest(BaseModel):
    query: str

class ConfigUpdate(BaseModel):
    ollama_url: str


@app.get("/health")
def health():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rag_chunks;")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"status": "ok", "chunks": count, "ollama_url": ollama_url()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/db-stats")
def db_stats():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rag_chunks;")
        total_chunks = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT filename) FROM rag_chunks;")
        total_files = cur.fetchone()[0]
        cur.execute("SELECT filename, COUNT(*) FROM rag_chunks GROUP BY filename ORDER BY filename;")
        files = [{"filename": row[0], "chunks": row[1]} for row in cur.fetchall()]
        cur.close()
        conn.close()
        return {"engine": "pgvector", "table": "rag_chunks",
                "total_chunks": total_chunks, "total_files": total_files, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
def update_config(body: ConfigUpdate):
    cfg = load_cfg()
    cfg["ollama_url"] = body.ollama_url
    save_cfg(cfg)
    return {"status": "ok", "ollama_url": body.ollama_url}


async def embed_with_sem(sem, chunk):
    async with sem:
        return await embed(chunk)


@app.post("/ingest")
async def ingest():
    files = list(FILES_DIR.glob("*.md"))
    if not files:
        return {"status": "no files found", "total_chunks": 0}

    sem = asyncio.Semaphore(5)
    conn = get_conn()
    cur = conn.cursor()
    results = []
    errors = []
    total = 0
    status = "ok"

    try:
        cur.execute("DELETE FROM rag_chunks;")
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                chunks = chunk_text(text)
                if not chunks:
                    results.append({"file": f.name, "chunks": 0})
                    continue
                vectors = await asyncio.gather(*[embed_with_sem(sem, c) for c in chunks])
                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    cur.execute(
                        "INSERT INTO rag_chunks (filename, chunk_index, content, embedding) VALUES (%s, %s, %s, %s::vector)",
                        (f.name, i, chunk, str(vec))
                    )
                    total += 1
                results.append({"file": f.name, "chunks": len(chunks)})
            except Exception as e:
                log.error("Ingest failed for %s: %s", f.name, e)
                errors.append({"file": f.name, "error": str(e)})
        if total == 0 and errors:
            conn.rollback()
            status = "failed"
        else:
            conn.commit()
    finally:
        cur.close()
        conn.close()

    return {"status": status, "files": len(files), "total_chunks": total,
            "details": results, "errors": errors}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        q_vec = await embed(req.query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT filename, content FROM rag_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (str(q_vec), TOP_K)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"answer": "No documentation ingested yet. Call /ingest first.",
                "query": req.query, "sources": [], "chunks_used": 0}

    docs      = [r[1] for r in rows]
    filenames = [r[0] for r in rows]

    try:
        rerank_resp = co.rerank(
            model="rerank-english-v3.0",
            query=req.query,
            documents=docs,
            top_n=TOP_N
        )
        top_idx = [r.index for r in rerank_resp.results]
    except Exception:
        top_idx = list(range(min(TOP_N, len(docs))))

    top_chunks  = [docs[i] for i in top_idx]
    top_sources = [filenames[i] for i in top_idx]
    top_files   = list(dict.fromkeys(top_sources))

    context = "\n\n".join(
        f"[Document {i+1} — {src}]\n{chunk}"
        for i, (src, chunk) in enumerate(zip(top_sources, top_chunks))
    )
    prompt = f"""You are a documentation assistant. Answer the user's question using ONLY the reference documents below.

Instructions:
- Base every statement strictly on the documents. Do not invent endpoints, fields, or behavior not written below.
- When relevant, give endpoints as method + path with parameters and return values.
- Mention which document(s) you used by their filename.
- Be concise and concrete.

=== REFERENCE DOCUMENTS ===
{context}
=== END OF DOCUMENTS ===

User question: {req.query}

Answer:"""

    try:
        answer = await llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")

    return {"answer": answer.strip(), "query": req.query,
            "sources": top_files, "chunks_used": len(top_chunks)}
