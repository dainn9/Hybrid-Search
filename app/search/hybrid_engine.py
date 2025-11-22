# search/hybrid_engine_prod.py
import logging
import json
import numpy as np
import pandas as pd
from underthesea import ner
from utils.redis_client import redis_client, settings
from utils.utils import (
    init_milvus_and_model,
    build_bm25_corpus,
    build_bm25_local,
    clean_text_for_query,
    removed_puncts
)

# --- Logging ---
logger = logging.getLogger("HybridSearchAPI")

class HybridEngine:
    def __init__(self, collection_name,
                 bm25_cache_file="cache/bm25_global.pkl"):
        # Init Milvus + embedding model
        self.collection, self.model = init_milvus_and_model(settings.uri_milvus, 
                                                            settings.api_key_milvus, 
                                                            collection_name)
        # Build / load global BM25 corpus (for queries without city filter)
        self.all_docs, self.bm25_corpus_full = build_bm25_corpus(
            self.collection, cache_file=bm25_cache_file
        )

        # City dictionary
        self.cities = {
            "hồ chí minh": "Hồ Chí Minh",
            "tp hồ chí minh": "Hồ Chí Minh",
            "tp.hcm": "Hồ Chí Minh",
            "sài gòn": "Hồ Chí Minh",
            "hà nội": "Hà Nội",
            "đà nẵng": "Đà Nẵng",
            "phú quốc": "Phú Quốc",
            "nha trang": "Nha Trang",
            "hội an": "Hội An",
            "đà lạt": "Đà Lạt",
            "sa pa": "Sa Pa",
            "sapa": "Sa Pa",
            "huế": "Huế",
            "vũng tàu": "Vũng Tàu"
        }

    # --- City detection ---
    def detect_city(self, query: str):
        query_lower = query.lower()
        # Rule-based
        for k, v in self.cities.items():
            if k in query_lower:
                return v
        # NER fallback
        try:
            for word, _, _, tag in ner(query):
                if tag.endswith("LOC"):
                    expr = f'Location == "{word}"'
                    results = self.collection.query(expr=expr, output_fields=["Location"])
                    if results:
                        return word.title()
        except Exception:
            pass
        return None

    # --- Prepare query ---
    def prepare_query(self, query: str):
        city = self.detect_city(query)
        tokenized_query = removed_puncts(query).split()
        expr = ""
        bm25_scores = []
        filtered_results = []

        # Filter by city
        if city:
            expr = f'Location like "%{city.lower()}%"'
            filtered_results = self.collection.query(
                expr=expr,
                output_fields=["HotelID", "Description", "Location"],
            )
            if filtered_results:
                filtered_docs = [r["Description"] for r in filtered_results]
                bm25_local = build_bm25_local(filtered_docs)
                bm25_scores = bm25_local.get_scores(tokenized_query)

        # Nếu không filter city hoặc filter rỗng, dùng global BM25
        if not filtered_results:
            filtered_results = self.all_docs
            bm25_scores = self.bm25_corpus_full.get_scores(tokenized_query)

        # Normalize BM25 scores
        bm25_scores = np.array(bm25_scores)
        if len(bm25_scores) == 0:
            bm25_scores = []
        elif bm25_scores.max() != bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores = [0] * len(bm25_scores)

        semantic_query = query.lower().replace(city.lower(), "").strip() if city else query.lower()
        return semantic_query, expr, bm25_scores, filtered_results

    # --- Hybrid search ---
    def hybrid_search(self, query: str, alpha: float = None, top_k: int = 10, debug: bool = False):
        if alpha is None:
            alpha = settings.alpha

        # Redis cache
        cache_key = f"search:{query}:{top_k}:{alpha}"
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for key={cache_key}")
            return json.loads(cached)
        else:
            logger.info(f"Cache miss for key={cache_key}")

        semantic_query, expr, bm25_scores, filtered_results = self.prepare_query(query)
        semantic_query = clean_text_for_query(semantic_query)

        # BM25 mapping
        bm25_dict = {r["HotelID"]: score for r, score in zip(filtered_results, bm25_scores)}

        # Dense search
        query_emb = self.model.encode([semantic_query], normalize_embeddings=True)
        search_params = search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self.collection.search(
            data=query_emb,
            anns_field="TextForEmbedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["HotelID", "Description", "Location"],
        )

        # Fusion: dense similarity + BM25
        hits = []
        for hits_list in results:
            for hit in hits_list:
                hotel_id = hit.entity.get("HotelID")
                dense_score = hit.distance
                bm25_score = bm25_dict.get(hotel_id, 0)
                final_score = alpha * dense_score + (1 - alpha) * bm25_score
                hits.append({
                    "doc_id": hotel_id,
                    # "title": hit.entity.get("Description", "Unknown"),
                    # "location": hit.entity.get("Location", ""),
                    "score": final_score
                })
                if debug:
                    hits[-1]["dense_score"] = dense_score
                    hits[-1]["bm25_score"] = bm25_score

         # Sort top_k
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:top_k]

        # Cache 24h
        redis_client.setex(cache_key, 86400, json.dumps(hits))

        return hits

# --- Usage example ---
if __name__ == "__main__":
    engine = HybridEngine(collection_name=settings.collection_name)
    query = "Khách sạn Phú Quốc gần biển"
    results = engine.hybrid_search(query, top_k=10)
    df = pd.DataFrame(results)
    print(df.head())