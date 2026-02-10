import os
from dataclasses import dataclass

# dotenv ist optional für Tests; falls nicht installiert, keine .env laden
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(path: str | None = None):
        return None

# BASE_DIR zeigt auf das Projekt-Root-Verzeichnis (2 Ebenen über diesem File)
# /home/user/rag_csv/src/rag_csv/config/settings.py -> /home/user/rag_csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Versuche, die .env Datei zu laden (noop falls dotenv fehlt)
load_dotenv(ENV_FILE)

def _str_to_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

@dataclass
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL") or "http://localhost:6333"
    api_key: str = os.getenv("QDRANT_API_KEY") or ""
    inc_collection: str = os.getenv("QDRANT_INC_COLLECTION") or "incidents"
    kb_collection: str = os.getenv("QDRANT_KB_COLLECTION") or "knowledgebase"


@dataclass
class EmbeddingConfig:
    # Ollama Embedding endpoint - Werte müssen in .env gesetzt sein
    base_url: str = os.getenv("EMBEDDING_URL")
    fallback_url: str = os.getenv("EMBEDDING_FALLBACK_URL") or ""  # Optional: Fallback-Server
    model: str = os.getenv("EMBEDDING_MODEL")
    dim: int = int(os.getenv("EMBEDDING_DIM") or "0")
    
    def __post_init__(self):
        if not self.base_url:
            raise ValueError("EMBEDDING_URL muss in .env gesetzt sein")
        if not self.model:
            raise ValueError("EMBEDDING_MODEL muss in .env gesetzt sein")
        if self.dim <= 0:
            raise ValueError("EMBEDDING_DIM muss in .env gesetzt sein und > 0")


@dataclass
class OllamaConfig:
    #URLs für verschiedene Profile
    url: str = os.getenv("OLLAMA_URL") or ""
    url_low_profile: str = os.getenv("OLLAMA_URL_LOW_PROFILE") or ""
    url_mid_profile: str = os.getenv("OLLAMA_URL_MID_PROFILE") or ""
    url_high_profile: str = os.getenv("OLLAMA_URL_HIGH_PROFILE") or ""
    url_ultra_profile: str = os.getenv("OLLAMA_URL_ULTRA_PROFILE") or ""
    url_test: str = os.getenv("OLLAMA_URL_TEST") or ""

    #Standardmodell
    #model: str = os.getenv("OLLAMA_MODEL") or ""
    threads: int = int(os.getenv("OLLAMA_THREADS") or "8")
    threads_low: int = int(os.getenv("OLLAMA_THREADS_LOW") or "4")
    threads_mid: int = int(os.getenv("OLLAMA_THREADS_MID") or "8")
    threads_high: int = int(os.getenv("OLLAMA_THREADS_HIGH") or "16")
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX") or "4096")


@dataclass
class EvaluationConfig:
    """Konfiguration für RAG Evaluation."""
    top_k: int = int(os.getenv("TOP_K") or "10")
    runs_per_testcase: int = int(os.getenv("RUNS_PER_TESTCASE") or "3")
    # LLM Judge Konfiguration
    use_llm_judge: bool = _str_to_bool(os.getenv("USE_LLM_JUDGE"), False)
    llm_judge_api_url: str = os.getenv("LLM_JUDGE_API_URL") or ""
    llm_judge_api_key: str = os.getenv("LLM_JUDGE_API_KEY") or ""
    llm_judge_model: str = os.getenv("LLM_JUDGE_MODEL") or "gpt-4o-mini"
    llm_judge_temperature: float = float(os.getenv("LLM_JUDGE_TEMPERATURE") or "0.1")
    llm_judge_max_tokens: int = int(os.getenv("LLM_JUDGE_MAX_TOKENS") or "1000")
    
@dataclass
class DataConfig:
    data_dir: str = os.getenv("DATA_DIR") or os.path.join(BASE_DIR, "data")
    # Standard auf vorhandene Dateien im `data`-Ordner setzen
    incident_csv: str = os.getenv("INCIDENT_CSV") or "synthetic_incidents_with_kb.csv"
    kb_csv: str = os.getenv("KB_CSV") or "kb_articles_llm.csv"
    total_tickets: int = int(os.getenv("TOTAL_TICKETS") or "20")
    tickets_per_call: int = int(os.getenv("TICKETS_PER_CALL") or "5")
    model_incidents: str = os.getenv("OLLAMA_MODEL_INCIDENTS") or "llama3.1:8b-instruct-q4_K_M"

    @property
    def incident_path(self):
        # Falls absolute Pfade angegeben wurden, diese direkt nutzen
        if self.incident_csv.startswith("/"):
            return self.incident_csv
        # Immer absoluten Pfad verwenden basierend auf BASE_DIR
        path = os.path.join(self.data_dir, self.incident_csv)
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        return path

    @property
    def kb_path(self):
        if self.kb_csv.startswith("/"):
            return self.kb_csv
        # Immer absoluten Pfad verwenden basierend auf BASE_DIR
        path = os.path.join(self.data_dir, self.kb_csv)
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        return path

@dataclass
class GeneratorConfig:
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    output_csv_path: str = os.getenv("OUTPUT_CSV_PATH") or os.path.join(os.getenv("OUTPUT_DIR", "output"), "generator")
    output_csv_filename: str = os.getenv("OUTPUT_CSV_FILENAME", "generated_tickets.csv")
    benchmarks_dir: str = os.path.join(os.getenv("OUTPUT_DIR", "output"), "benchmarks")
    metrics_dir: str = os.path.join(os.getenv("OUTPUT_DIR", "output"), "metrics")
    logs_dir: str = os.path.join(os.getenv("OUTPUT_DIR", "output"), "logs")
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
