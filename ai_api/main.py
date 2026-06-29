from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from pathlib import Path
import httpx

app = FastAPI()

# ── Config — fill these in before running ────────────────────────────────────
ADMIN_KEY  = "your_admin_key_here"
RAG_URL    = "http://localhost:1464"
FILES_DIR  = Path("./files")
INDEX_HTML = Path("./index.html")
FILES_DIR.mkdir(parents=True, exist_ok=True)


def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai_api"}


@app.post("/chat")
async def chat(body: dict, _=Depends(require_admin)):
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            r = await client.post(f"{RAG_URL}/chat", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"rag_api unreachable: {e}")


@app.post("/ingest")
async def ingest(_=Depends(require_admin)):
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            r = await client.post(f"{RAG_URL}/ingest")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.post("/config")
async def update_config(body: dict, _=Depends(require_admin)):
    """Update the Ollama ngrok URL when Colab restarts."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(f"{RAG_URL}/config", json=body)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.get("/files")
async def list_files(_=Depends(require_admin)):
    files = sorted(f.name for f in FILES_DIR.glob("*.md"))
    return {"count": len(files), "files": files}


@app.get("/files/{filename}")
async def get_file(filename: str, _=Depends(require_admin)):
    path = FILES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}


@app.put("/files/{filename}")
async def upsert_file(filename: str, body: dict, _=Depends(require_admin)):
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    (FILES_DIR / filename).write_text(content, encoding="utf-8")
    return {"status": "ok", "filename": filename}


@app.delete("/files/{filename}")
async def delete_file(filename: str, _=Depends(require_admin)):
    path = FILES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    path.unlink()
    return {"status": "deleted", "filename": filename}
