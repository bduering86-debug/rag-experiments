import os
from dataclasses import dataclass

# dotenv ist optional für Tests; falls nicht installiert, keine .env laden
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(path: str | None = None):
        return None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Versuche, die .env Datei zu laden (noop falls dotenv fehlt)
load_dotenv(ENV_FILE)

def _str_to_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

@dataclass
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    inc_collection: str = os.getenv("QDRANT_INC_COLLECTION", "incidents_csv")
    kb_collection: str = os.getenv("QDRANT_KB_COLLECTION", "kb_csv")


@dataclass
class EmbeddingConfig:
    base_url: str = os.getenv("EMBEDDING_URL", "http://localhost:8080/v1/embeddings")
    model: str = os.getenv("TEMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    dim: int = int(os.getenv("EMBEDDING_DIM", "384"))


@dataclass
class OllamaConfig:
    #URLs für verschiedene Profile
    url: str = os.getenv("OLLAMA_URL", "")
    url_low_profile: str = os.getenv("OLLAMA_URL_LOW_PROFILE", "")
    url_mid_profile: str = os.getenv("OLLAMA_URL_MID_PROFILE", "")
    url_high_profile: str = os.getenv("OLLAMA_URL_HIGH_PROFILE", "")
    url_ultra_profile: str = os.getenv("OLLAMA_URL_ULTRA_PROFILE", "")
    url_test: str = os.getenv("OLLAMA_URL_TEST", "")

    #Standardmodell
    model: str = os.getenv("OLLAMA_MODEL", "")
    threads: int = int(os.getenv("OLLAMA_THREADS", "8"))
    
@dataclass
class DataConfig:
    data_dir: str = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    # Standard auf vorhandene Dateien im `data`-Ordner setzen
    incident_csv: str = os.getenv("INCIDENT_CSV", "synthetic_incidents_with_kb.csv")
    kb_csv: str = os.getenv("KB_CSV", "kb_articles_llm.csv")
    total_tickets: int = int(os.getenv("TOTAL_TICKETS", "20"))
    tickets_per_call: int = int(os.getenv("TICKETS_PER_CALL", "5"))
    model_incidents: str = os.getenv("OLLAMA_MODEL_INCIDENTS", "llama3.1:8b-instruct-q4_K_M")

    @property
    def incident_path(self):
        # Falls absolute Pfade angegeben wurden, diese direkt nutzen
        if self.incident_csv.startswith("/"):
            return self.incident_csv
        return os.path.join(self.data_dir, self.incident_csv)

    @property
    def kb_path(self):
        if self.kb_csv.startswith("/"):
            return self.kb_csv
        return os.path.join(self.data_dir, self.kb_csv)

@dataclass
class GeneratorConfig:
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    output_csv_filename: str = os.getenv("OUTPUT_CSV_FILENAME", "generated_tickets.csv")
    total_tickets: int = int(os.getenv("TOTAL_TICKETS", "1"))
    tickets_per_call: int = int(os.getenv("TICKETS_PER_CALL", "1"))
    generator_model_incidents: str = os.getenv("GENERATOR_MODEL_INCIDENTS", "llama3.1:8b-instruct-q4_K_M")
    
    generator_temperature: float = float(os.getenv("GENERATOR_TEMPERATURE", "0.2"))
    generator_max_tokens: int = int(os.getenv("GENERATOR_MAX_TOKENS", "512"))
    generator_top_p: float = float(os.getenv("GENERATOR_TOP_P", "0.9"))
    generator_ctx_tokens: int = int(os.getenv("GENERATOR_CTX_TOKENS", "2048"))
    generator_seed: int = int(os.getenv("GENERATOR_SEED", "12345"))
    generator_repeat_penalty: float = float(os.getenv("GENERATOR_REPEAT_PENALTY", "1.1"))
    generator_num_predict: int = int(os.getenv("GENERATOR_NUM_PREDICT", "1024"))
    
    generator_model_knowledgebase: str = os.getenv("GENERATOR_MODEL_KNOWLEDGEBASE", "llama3.1:8b-instruct-q4_K_M")
    generator_model_knowledgebase_test: str = os.getenv("GENERATOR_MODEL_KNOWLEDGEBASE_TEST", "phi3:3.8b")
    generator_tickets_for_kb_context: int = int(os.getenv("GENERATOR_TICKETS_FOR_KB_CONTEXT", "10"))
    generator_kb_temperature: float = float(os.getenv("GENERATOR_KB_TEMPERATURE", "0.5"))
    generator_kb_top_p: float = float(os.getenv("GENERATOR_KB_TOP_P", "0.9"))
    generator_kb_repeat_penalty: float = float(os.getenv("GENERATOR_KB_REPEAT_PENALTY", "1.1"))
    generator_kb_ctx_tokens: int = int(os.getenv("GENERATOR_KB_CTX_TOKENS", "4096"))
    generator_kb_num_predict: int = int(os.getenv("GENERATOR_KB_NUM_PREDICT", "1500"))

@dataclass
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    to_console: bool = _str_to_bool(os.getenv("LOG_TO_CONSOLE", "true"), True)
    to_file: bool = _str_to_bool(os.getenv("LOG_TO_FILE", "true"), True)
    path: str = os.getenv("LOG_PATH", "logs")
    log_file: str = os.getenv("LOG_FILE", path+"/default.log")
