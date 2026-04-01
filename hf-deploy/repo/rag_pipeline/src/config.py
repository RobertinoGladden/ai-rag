from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-70b-versatile", env="GROQ_MODEL")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field("cpu", env="EMBEDDING_DEVICE")

    # Vector Store
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("rag_documents", env="CHROMA_COLLECTION_NAME")

    # RAG Settings
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(5, env="TOP_K_RETRIEVAL")
    max_tokens: int = Field(2048, env="MAX_TOKENS")
    temperature: float = Field(0.1, env="TEMPERATURE")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD")

    # MLflow
    mlflow_tracking_uri: str = Field("./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("rag_pipeline", env="MLFLOW_EXPERIMENT_NAME")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        """Buat direktori yang dibutuhkan jika belum ada."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.mlflow_tracking_uri).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings — di-cache supaya tidak re-parse tiap request."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
