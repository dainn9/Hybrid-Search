# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # milvus_host: str = "localhost"
    # milvus_port: int = 19530
    uri_milvus: str = ""
    api_key_milvus: str = ""
    redis_url: str = ""
    alpha: float = 0.5
    default_top_k1: int = 50
    default_top_k2: int = 20
    default_top_k3: int = 10
    collection_name: str = "hotels_collection_mpnet_base_v2"
    embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    max_query_length: int = 50
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()