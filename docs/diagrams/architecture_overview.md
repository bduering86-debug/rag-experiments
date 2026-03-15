# Vereinfachte Architekturansicht

```mermaid
flowchart LR
    subgraph Data[Data Layer]
        CSV1[Incidents CSV]
        CSV2[KB CSV]
        TC[Testcases CSV]
    end

    subgraph Ingest[Ingestion]
        ING[RAGIngestor]
        CH[Chunking]
        EMB[Embeddings]
        QD[(Qdrant Collections)]
    end

    subgraph Runtime[Retrieval and Answering]
        ORCH[RAGAnswerOrchestrator]
        RET[Retrieval API]
        RR[Cross-Encoder Reranker]
        OLL[Ollama Models]
    end

    subgraph Eval[Evaluation and Metrics]
        METRICS[SystemMetricsLogger]
        LAT[LatencyTracker]
        TOK[TokenTracker]
        JUDGE[LLMJudge]
        SCORE[Score Calculators]
    end

    CSV1 --> ING
    CSV2 --> ING
    ING --> CH --> EMB --> QD

    TC --> ORCH
    ORCH --> RET --> QD
    RET -. optional .-> RR
    ORCH --> OLL

    ORCH --> LAT
    ORCH --> TOK
    ORCH --> METRICS
    ORCH -. optional .-> JUDGE
    ORCH --> SCORE

    SCORE --> OUT[(Output CSV and Summaries)]
    METRICS --> OUT
```

Kurzbeschreibung:
- Links liegen die Datenquellen, in der Mitte die Laufzeitkomponenten, rechts die Auswertung.
- Qdrant ist das gemeinsame Bindeglied zwischen Ingestion und Retrieval.
- Der Orchestrator verbindet Retrieval, Modellaufruf und Metrikberechnung.
