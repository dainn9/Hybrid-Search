# app.py
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from utils.redis_client import redis_client, settings
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from utils.utils import make_api_response
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager


from search.hybrid_engine import HybridEngine

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridSearchAPI")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

engine = None
# --- Init HybridEngine ---
def get_engine():
    global engine
    if engine is None:
        engine = HybridEngine(collection_name=settings.collection_name)
    return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    logger.info("HybridEngine initialized on startup.")
    yield
    logger.info("HybridEngine shutting down.")

# --- Init FastAPI ---
app = FastAPI(title="Hotel Hybrid Search API",
              version="1.0",
              description="API for hybrid search over hotel data using Milvus and Redis.",
              lifespan=lifespan)

app.state.limiter = limiter

# Allow CORS for all origins (frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Search endpoint ---
@app.get("/search")
@limiter.limit("10/minute")  # max 10 request / phút / IP
async def search_endpoint(
    request: Request,
    query: str = Query(...,min_length=1,max_length=500, description="Search query text"),
    top_k: int = Query(10, description="Number of top results"),
):
    if len(query) > settings.max_query_length:
        raise HTTPException(status_code=400, detail="Query too long")

    logger.info(f"GET /search query={query}, top_k={top_k}")

    results = await run_in_threadpool(engine.hybrid_search, query=query, top_k3=top_k)

    return results

# --- Health check endpoint ---
@app.get("/health")
@limiter.limit("1/minute")  # max 1 request / phút / IP
def health_check(request: Request):
    try:
        redis_client.ping()
        test_docs = engine.collection.query(expr="", limit=1)
        return {"status": "ok", "docs_count": len(test_docs)}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}


# --- Exception handler for rate limiting ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content= make_api_response(
            status=429,
            message="Too many requests - Please slow down.",
            extra={
                "retry_after_seconds": 60,
                "path": request.url.path
            }
        )
    )

# --- Run server ---
# if __name__ == "__main__":
#     print("Swagger UI available at http://127.0.0.1:8009/docs")
#     uvicorn.run("app:app", host="127.0.0.1", port=8009, reload=True)

