# utils/production_utils.py
import os
import pickle
import string
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from rank_bm25 import BM25Okapi

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def init_milvus_and_model(uri, token, 
                          collection_name="hotels_collection_mpnet_base_v2",
                          model_name="paraphrase-multilingual-mpnet-base-v2"):
    """
    Init Milvus collection and SentenceTransformer model safely.
    """
    try:
        connections.connect("default", uri=uri, token=token)
        collection = Collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"[Milvus] Failed to init collection '{collection_name}': {e}")

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"[Model] Failed to load SentenceTransformer '{model_name}': {e}")

    return collection, model


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
        doc = removed_puncts(doc)
        doc_tokens = doc.split()
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
