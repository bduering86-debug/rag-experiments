# Klassendiagramm der Anwendung (src/rag_csv)

Hinweis: In diesem Workspace existiert der Pfad `src/rag` nicht. Das Diagramm basiert daher auf den Klassen unter `src/rag_csv`.

```mermaid
classDiagram
    direction LR

    class RAGIngestor
    class DataConfig
    class Embeddings
    class EmbeddingConfig
    class QdrantConfig
    class SearchHit
    class CrossEncoderReranker
    class RerankingConfig
    class RetrievalConfig

    class RAGAnswerOrchestrator
    class OllamaConfig
    class EvaluationConfig
    class nDCGTopK
    class RecallTopK
    class RetrievalScoreCalculator
    class LatencyTracker
    class TokenTracker
    class SystemMetricsLogger

    class LLMJudge
    class LLMAPIClient
    class LLMConfig
    class AnswerQualityCalculator
    class QualityScores
    class TokenScoreCalculator
    class LatencyScoreCalculator

    class KBGeneratorConfig
    class KBArticle
    class KBGenerator
    class TicketGenerator
    class GeneratorConfig
    class OllamaRunMetrics

    %% Ingestion / Retrieval
    RAGIngestor --> DataConfig : uses
    Embeddings --> EmbeddingConfig : uses
    CrossEncoderReranker ..> SearchHit : reranks
    RerankingConfig ..> CrossEncoderReranker : configures
    RetrievalConfig ..> SearchHit : retrieval params
    QdrantConfig ..> SearchHit : source collection context

    %% Evaluation Orchestration
    RAGAnswerOrchestrator --> OllamaConfig : uses
    RAGAnswerOrchestrator --> EvaluationConfig : uses
    RAGAnswerOrchestrator --> nDCGTopK : metric
    RAGAnswerOrchestrator --> RecallTopK : metric
    RAGAnswerOrchestrator --> RetrievalScoreCalculator : metric
    RAGAnswerOrchestrator --> LatencyTracker : tracks
    RAGAnswerOrchestrator --> TokenTracker : tracks
    RAGAnswerOrchestrator --> SystemMetricsLogger : logs
    RAGAnswerOrchestrator --> LLMJudge : optional quality eval
    RAGAnswerOrchestrator ..> SearchHit : consumes retrieval hits

    %% LLM Judge Subsystem
    LLMJudge --> LLMAPIClient : calls
    LLMJudge --> AnswerQualityCalculator : computes score
    LLMAPIClient --> LLMConfig : configured by
    AnswerQualityCalculator ..> QualityScores : input

    %% Generator Subsystem
    KBGenerator --> KBGeneratorConfig : uses
    KBGenerator --> KBArticle : creates
    TicketGenerator ..> GeneratorConfig : configured from

    %% Metrics helpers
    TokenScoreCalculator ..> RAGAnswerOrchestrator : post-processing
    LatencyScoreCalculator ..> RAGAnswerOrchestrator : post-processing
    OllamaRunMetrics ..> KBGenerator : runtime metrics
    OllamaRunMetrics ..> TicketGenerator : runtime metrics
```

## Lesart

- Durchgezogene Pfeile (`-->`) zeigen direkte Nutzung oder Aggregation.
- Gestrichelte Pfeile (`..>`) zeigen lose Abhängigkeiten oder indirekte Kopplung.
- `RAGAnswerOrchestrator` ist die zentrale Klasse für den Evaluations-Flow.
- `RAGIngestor` deckt den Ingestion-Flow ab, `CrossEncoderReranker` erweitert den Retrieval-Teil optional.
