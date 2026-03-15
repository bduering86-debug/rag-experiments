# Detailliertes Diagramm: Retrieval und Ingestion

```mermaid
classDiagram
    direction LR

    class RAGIngestor {
        +load_source(source, file_path) List~Document~
        +chunk(docs, kind) List~Document~
        +embed_and_store(chunks, collection, recreate) int
        +ingest_full_pipeline(source, collection, file_path, recreate) int
        +ingest_all(recreate) dict
    }

    class DataConfig {
        +data_dir str
        +incident_csv str
        +kb_csv str
        +incident_path str
        +kb_path str
    }

    class Embeddings {
        +embed_documents(texts) List~List~float~~
        +embed_query(text) List~float~
        -_embed(texts) List~List~float~~
        -_post_single(session, url, text) List~float~
    }

    class EmbeddingConfig {
        +base_url str
        +fallback_url str
        +model str
        +dim int
    }

    class QdrantConfig {
        +url str
        +inc_collection str
        +kb_collection str
    }

    class SearchHit {
        +collection str
        +score float
        +text str
        +metadata Dict
    }

    class CrossEncoderReranker {
        +enabled bool
        +rerank(query, hits, top_k) List~SearchHit~
        +batch_rerank(queries, hits_list, top_k) List~List~SearchHit~~
    }

    class RerankingConfig {
        +model_name str
        +enabled bool
        +top_k_multiplier int
    }

    class RetrievalConfig {
        +retrieve_k_chunks_multiplier int
    }

    class VectorstoreAPI {
        +get_vectorstore(kind)
        +recreate_collection(kind)
        +ingest_documents(kind, docs, batch_size)
        +count_points(kind) int
    }

    class RetrievalAPI {
        +search(query, top_k, use_kb, use_incidents, merge, rerank) List~SearchHit~
        +search_collection(query, collection, top_k) List~SearchHit~
        +dedup_by_kb_id(docs_with_scores)
    }

    class Document

    RAGIngestor --> DataConfig : uses
    RAGIngestor --> VectorstoreAPI : calls
    VectorstoreAPI --> QdrantConfig : uses
    VectorstoreAPI --> Embeddings : uses
    Embeddings --> EmbeddingConfig : uses

    RAGIngestor ..> Document : loads/chunks

    RetrievalAPI --> VectorstoreAPI : calls
    RetrievalAPI --> RetrievalConfig : uses
    RetrievalAPI ..> SearchHit : returns

    RetrievalAPI --> CrossEncoderReranker : optional rerank
    RerankingConfig ..> CrossEncoderReranker : config
    CrossEncoderReranker ..> SearchHit : reranks
```

Kurzbeschreibung:
- Der Ingestion-Flow liegt in RAGIngestor und delegiert Speicherung/Embeddings an die Vectorstore-Ebene.
- Der Retrieval-Flow nutzt SearchHit als einheitliches Ergebnisobjekt.
- Reranking ist optional und wird auf die vorselektierten Treffer angewendet.
