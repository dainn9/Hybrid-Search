# app.py
import redis
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from utils.redis_client import redis_client, settings

from search.hybrid_engine import HybridEngine

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridSearchAPI")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- Init FastAPI ---
app = FastAPI(title="Hotel Hybrid Search API")
app.state.limiter = limiter


# Allow CORS for all origins (frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
# --- Init HybridEngine ---
def get_engine():
    global engine
    if engine is None:
        engine = HybridEngine(collection_name=settings.collection_name)
    return engine

# --- Search endpoint ---
@app.get("/search")
@limiter.limit("10/minute")  # max 10 request / phút / IP
def search_endpoint(
    Request, request,
    query: str = Query(...,min_length=1,max_length=50, description="Search query text"),
    top_k: int = Query(10, description="Number of top results"),
    # alpha: float = Query(None, description="Fusion weight dense vs BM25")
):
    if len(query) > settings.max_query_length:
        raise HTTPException(status_code=400, detail="Query too long")

    engine = get_engine()
    logger.info(f"GET /search query={query}, top_k={top_k}")
    results = engine.hybrid_search(query=query, top_k=top_k)
    return {"query": query, "top_k": top_k,"results": results}

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    try:
        engine = get_engine()
        redis_client.ping()
        test_docs = engine.collection.query(expr="", limit=1)
        return {"status": "ok", "docs_count": len(test_docs)}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8009, reload=True)
