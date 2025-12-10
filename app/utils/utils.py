# utils/production_utils.py
import os
import pickle
import string
import re
import unicodedata
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import connections, Collection
from rank_bm25 import BM25Okapi
from utils.api_response import make_api_response

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def init_milvus_and_model(uri, token, 
                          collection_name="hotels_collection_mpnet_base_v2",
                          embedding_model_name="paraphrase-multilingual-mpnet-base-v2",
                          cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Init Milvus collection and SentenceTransformer model safely.
    """
    try:
        connections.connect("default", uri=uri, token=token)
        collection = Collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"[Milvus] Failed to init collection '{collection_name}': {e}")

    try:
        model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        raise RuntimeError(f"[Model] Failed to load SentenceTransformer '{embedding_model_name}': {e}")
    
    try:
        cross_encoder = CrossEncoder(cross_encoder_name)
    except Exception as e:
        raise RuntimeError(f"[Model] Failed to load CrossEncoder '{cross_encoder_name}': {e}")

    return collection, model, cross_encoder


def build_bm25_corpus(collection, batch_size=500, cache_file=None):
    """
    Build BM25 corpus from Milvus collection.
    Use cache_file (pickle) if exists to speed up production start.
    """
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            tokenized_corpus, all_docs = pickle.load(f)
        print(f"[BM25] Loaded cached corpus from {cache_file}, docs={len(all_docs)}")
        bm25_corpus_full = BM25Okapi(tokenized_corpus)
        return all_docs, bm25_corpus_full

    offset = 0
    all_docs = []
    tokenized_corpus = []

    while True:
        batch = collection.query(
            expr="",  # no filter for global corpus
            offset=offset,
            limit=batch_size,
            output_fields=["HotelID", "Description"],
        )

        if not batch:
            break

        all_docs.extend(batch)

        # Tokenize từng batch
        batch_texts = [d["Description"] for d in batch]
        tokenized_batch = clean_and_tokenize(batch_texts)
        tokenized_corpus.extend(tokenized_batch)

        offset += batch_size
        print(f"[BM25] Loaded batch size={len(batch)}, total_docs={len(all_docs)}")
    
    bm25_corpus_full = BM25Okapi(tokenized_corpus)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((tokenized_corpus, all_docs), f)
        print(f"[BM25] Saved corpus cache to {cache_file}")

    return all_docs, bm25_corpus_full


def build_bm25_local(docs_list):
    """
    Build BM25 for a small subset of filtered docs (local BM25)
    """
    tokenized = clean_and_tokenize(docs_list)
    return BM25Okapi(tokenized)


def clean_and_tokenize(docs_list):
    """
    Clean and tokenize list of documents
    """
    tokenized_corpus = []
    for doc in docs_list:
        doc_tokens = removed_puncts(doc).split()
        tokenized_corpus.append(doc_tokens)
    return tokenized_corpus


def removed_puncts(text):
    """
    Remove punctuation,-"""
    if not isinstance(text, str):
        return ""
    return text.translate(str.maketrans('','',string.punctuation)).lower()

def clean_text_for_query(text):
    """
    Clean query text for embedding / search
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r"[^\w\s/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_for_cross_encoder(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_hybrid_results_json(reranked_docs, query, top_k, debug=False):
    """
    Chuyển danh sách reranked_docs thành JSON API-style
    reranked_docs: list of dict, mỗi dict có keys:
        - HotelID
        - Description
        - Location
        - score
        - dense_score (optional)
        - bm25_score (optional)
    query: string
    top_k: int
    debug: bool, nếu True giữ thêm dense_score và bm25_score
    """
    if not reranked_docs:
        return make_api_response(
            status=200,
            message="No results found",
            query=query,
            extra={"top_k": top_k}
        )

    results_list = []
    for doc in reranked_docs[:top_k]:
        doc_json = {
            "HotelID": str(doc.get("HotelID", "")),
            "Description": doc.get("Description", ""),
            "Location": doc.get("Location", ""),
            "score": float(doc.get("score", 0.0))
        }
        if debug:
            if "dense_score" in doc:
                doc_json["dense_score"] = float(doc["dense_score"])
            if "bm25_score" in doc:
                doc_json["bm25_score"] = float(doc["bm25_score"])

        results_list.append(doc_json)

    response = make_api_response(
        status=200,
        message="Query executed successfully",
        query=query,
        results=results_list,
        extra={"top_k": top_k}
    )

    return response