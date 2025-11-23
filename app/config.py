import os
from dataclasses import dataclass
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)


@dataclass
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    inc_collection: str = os.getenv("QDRANT_INC_COLLECTION", "incidents_csv")
    kb_collection: str = os.getenv("QDRANT_KB_COLLECTION", "kb_csv")


@dataclass
class EmbeddingConfig:
    base_url: str = os.getenv("TEI_URL", "http://localhost:8080/v1/embeddings")
    model: str = os.getenv("TEI_MODEL", "BAAI/bge-small-en-v1.5")
    dim: int = int(os.getenv("TEI_EMBED_DIM", "384"))


@dataclass
class OllamaConfig:
    url: str = os.getenv("OLLAMA_URL", "")
    model: str = os.getenv("OLLAMA_MODEL", "")
    threads: int = int(os.getenv("OLLAMA_THREADS", "8"))

    @dataclass
class DataConfig:
    data_dir: str = os.getenv("DATA_DIR", "")
    incident_csv: str = os.getenv("INCIDENT_CSV", "incidents.csv")
    kb_csv: str = os.getenv("KB_CSV", "kb.csv")

    @property
    def incident_path(self):
        # Falls absolute Pfade angegeben wurden → direkt nutzen
        if self.incident_csv.startswith("/"):
            return self.incident_csv
        return os.path.join(self.data_dir, self.incident_csv)

    @property
    def kb_path(self):
        if self.kb_csv.startswith("/"):
            return self.kb_csv
        return os.path.join(self.data_dir, self.kb_csv)
