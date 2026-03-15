# Sequenzdiagramm: Lauf eines Testfalls

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Test Runner
    participant Orch as RAGAnswerOrchestrator
    participant Ret as RetrievalAPI
    participant VS as QdrantVectorStore
    participant ReRank as CrossEncoderReranker
    participant Ollama as Ollama API
    participant Judge as LLMJudge
    participant Metrics as SystemMetricsLogger

    Runner->>Orch: evaluate_testcase(testcase)
    Orch->>Metrics: capture_snapshot(trace_id, pre)
    Orch->>Metrics: start_logging(trace_id)

    Orch->>Orch: latency.start_total()
    Orch->>Orch: latency.start_retrieval()

    Orch->>Ret: search(query, top_k, rerank)
    Ret->>VS: similarity_search_with_score(query)
    VS-->>Ret: raw chunk hits

    alt reranking enabled
        Ret->>ReRank: rerank(query, hits, top_k)
        ReRank-->>Ret: reranked hits
    end

    Ret-->>Orch: SearchHit list
    Orch->>Orch: latency.end_retrieval()

    Orch->>Orch: build prompt/context
    Orch->>Orch: latency.start_llm()
    Orch->>Ollama: generate(model, prompt)
    Ollama-->>Orch: answer + token stats
    Orch->>Orch: latency.end_llm()

    opt llm judge enabled
        Orch->>Judge: evaluate(ticket, context, answer)
        Judge-->>Orch: quality metrics
    end

    Orch->>Orch: compute Recall@nDCG and scores
    Orch->>Orch: latency.end_total()

    Orch->>Metrics: capture_snapshot(trace_id, post)
    Orch->>Metrics: stop_logging()

    Orch-->>Runner: result row with metrics
```

Kurzbeschreibung:
- Der Orchestrator ist der zentrale Ablaufknoten.
- Retrieval und LLM-Aufruf werden separat in der Latenz gemessen.
- Monitoring (pre/continuous/post) ist über die Trace-ID mit dem Testfall korreliert.
