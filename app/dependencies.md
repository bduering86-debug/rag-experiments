┌─────────────────────────────────────────────────────────────────────────┐
│                          ARCHITEKTUR-DIAGRAMM                           │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         BIN (KONFIGURATION)                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐  │
│  │  config.py      │      │ logging_utils.py │      │ metrics_utils  │  │
│  │─────────────────│      │──────────────────│      │────────────────│  │
│  │ • QdrantConfig  │      │ • setup_logging()│      │ • start_run()  │  │
│  │ • EmbeddingCfg  │◄─────┤ • get_logger()   │      │ • log_call()   │  │
│  │ • OllamaConfig  │      │                  │      │ • end_run()    │  │
│  │ • DataConfig    │      └──────────────────┘      └────────────────┘  │
│  │ • GeneratorCfg  │              ▲                         ▲            │
│  │ • LoggingConfig │              │                         │            │
│  └─────────────────┘              │                         │            │
│         ▲                          │                         │            │
│         │                          │                         │            │
└─────────┼──────────────────────────┼─────────────────────────┼────────────┘
          │                          │                         │
          │ imports                  │ imports                 │ imports
          │                          │                         │
┌─────────┼──────────────────────────┼─────────────────────────┼────────────┐
│          │                          │                         │            │
│          ▼                          ▼                         ▼            │
│  ┌────────────────────┐    ┌─────────────────────┐  ┌──────────────────┐ │
│  │   APP MODULE       │    │  GENERATOR MODULE   │  │  BENCHMARK       │ │
│  ├────────────────────┤    ├─────────────────────┤  ├──────────────────┤ │
│  │                    │    │                     │  │ • visual_bm.py   │ │
│  │ ┌────────────────┐ │    │ ┌─────────────────┐ │  │ • benchmark.py   │ │
│  │ │ loaders.py     │ │    │ │ ticketgenerator │ │  └──────────────────┘ │
│  │ │─────────────   │ │    │ │─────────────────│ │                       │
│  │ │ load_incidents │ │    │ │ TicketGenerator │ │  ┌──────────────────┐ │
│  │ │ load_kb_csv    │ │    │ │ • _build_prompt │ │  │  METRICS         │ │
│  │ └────────────────┘ │    │ │ • _call_ollama  │ │  │ • nDCGTopK       │ │
│  │         ▲          │    │ │ • _parse_batch  │ │  │ • RecallTopK     │ │
│  │         │          │    │ └─────────────────┘ │  └──────────────────┘ │
│  │         │          │                         │                       │
│  │ ┌─────────────────┐│    │ ┌─────────────────┐ │                       │
│  │ │ chunking.py     ││    │ │ kb_generator.py │ │                       │
│  │ │─────────────────││    │ │─────────────────│ │                       │
│  │ │ chunk_documents ││    │ │ KBGeneratorCfg  │ │                       │
│  │ └─────────────────┘│    │ │ KBArticle       │ │                       │
│  │         ▲          │    │ │ KBGenerator     │ │                       │
│  │         │          │    │ └─────────────────┘ │                       │
│  │         │          │           ▲              │                       │
│  │         │          │           │              │                       │
│  │ ┌────────────────┐ │    ┌───────┴───────────┐ │                       │
│  │ │ embeddings.py  │ │    │ generator_test.py │ │                       │
│  │ │────────────────│ │    │───────────────────│ │                       │
│  │ │ Embeddings     │◄────┤ • run_tests()     │ │                       │
│  │ │ • _embed()     │ │    └───────────────────┘ │                       │
│  │ │ • embed_docs   │ │                         │                       │
│  │ │ • embed_query  │ │                         │                       │
│  │ └────────────────┘ │                         │                       │
│  │         ▲          │                         │                       │
│  │         │          │                         │                       │
│  │ ┌────────────────┐ │                         │                       │
│  │ │ vectorstore.py │ │                         │                       │
│  │ │────────────────│ │                         │                       │
│  │ │ • get_vector   │ │                         │                       │
│  │ │ • recreate_coll│ │                         │                       │
│  │ │ • ingest_docs  │ │                         │                       │
│  │ │ • count_points │ │                         │                       │
│  │ └────────────────┘ │                         │                       │
│  │         ▲          │                         │                       │
│  │         │          │                         │                       │
│  │ ┌────────────────┐ │                         │                       │
│  │ │ ingest_kb.py   │ │    ┌─────────────────┐  │                       │
│  │ │ ingest_inc.py  │ │    │ setup_coll.py   │  │                       │
│  │ │────────────────│ │    │─────────────────│  │                       │
│  │ │ • main()       │ │    │ • setup_collct()  │  │                       │
│  │ └────────────────┘ │    └─────────────────┘  │                       │
│  │         ▲          │           ▲             │                       │
│  │         │          │           │             │                       │
│  │ ┌────────────────┐ │           │             │                       │
│  │ │ query_demo.py  │ │           │             │                       │
│  │ │────────────────│ │           │             │                       │
│  │ │ • retrieve_inc │ │           │             │                       │
│  │ │   _and_kb()    │ │           │             │                       │
│  │ │ • build_prompt │ │           │             │                       │
│  │ │ • ask_ollama() │ │           │             │                       │
│  │ └────────────────┘ │           │             │                       │
│  │         ▲          │           │             │                       │
│  └─────────┼──────────┘           │             │                       │
│            │                      │             │                       │
│  ┌─────────┴──────────────────────┴─────────────┴────────────────────┐  │
│  │                      EXTERNE DEPENDENCIES                         │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                                                                 │  │
│  │  • QdrantClient ◄──┐  QdrantVectorStore, Distance, VectorParams │  │
│  │  • Qdrant (deprecated) ◄──┐ LangChain Qdrant Integration          │  │
│  │  • Embeddings (LangChainEmbeddings) ◄──┐ LangChain Core           │  │
│  │  • RecursiveCharacterTextSplitter ◄──┐ LangChain Text Splitter   │  │
│  │  • Document ◄──┐ LangChain Core Document                         │  │
│  │  • requests ◄──┐ HTTP Library für Ollama & Embedding API        │  │
│  │  • pandas ◄──┐ Optional CSV Loader                              │  │
│  │  • csv ◄──┐ Standard CSV Fallback                               │  │
│  │  • matplotlib ◄──┐ Visualisierung                                │  │
│  │                                                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         DATENFLUSS                                       │
└─────────────────────────────────────────────────────────────────────────┘

  CSV-Dateien                      API-Server
      │                                │
      ▼                                ▼
  ┌─────────┐                    ┌──────────────┐
  │ Loaders │◄────────config─────┤ Embeddings   │
  └────┬────┘                    └──────────────┘
       │                                ▲
       │ List[Document]                 │
       ▼                                │
  ┌─────────┐                          │
  │Chunking │                          │
  └────┬────┘                          │
       │ List[Document]                │
       ▼                                │
  ┌─────────────┐                      │
  │ Vectorstore │◄─────────────────────┘
  │  (Qdrant)   │
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │ Collections │
  └─────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                    ABHÄNGIGKEITSBAUM (Top-Down)                         │
└─────────────────────────────────────────────────────────────────────────┘

main() in ingest_kb.py / ingest_incidents.py
  │
  ├─► DataConfig (from bin.config)
  │    └─► Properties: incident_path, kb_path
  │
  ├─► get_logger() (from bin.logging_utils)
  │    └─► LoggingConfig (from bin.config)
  │
  ├─► load_kb_csv() / load_incidents_csv() (from app.loaders)
  │    └─► CSV-Dateien (über config Pfade)
  │
  ├─► chunk_documents() (from app.chunking)
  │    ├─► RecursiveCharacterTextSplitter
  │    └─► Document (LangChain)
  │
  ├─► recreate_collection() (from app.vectorstore)
  │    ├─► QdrantConfig (from bin.config)
  │    ├─► EmbeddingConfig (from bin.config)
  │    ├─► QdrantClient
  │    └─► VectorParams, Distance (Qdrant models)
  │
  ├─► ingest_documents() (from app.vectorstore)
  │    ├─► get_vectorstore()
  │    │    ├─► get_embeddings()
  │    │    │    └─► Embeddings (from app.embeddings)
  │    │    │         ├─► EmbeddingConfig (from bin.config)
  │    │    │         └─► requests (für API-Calls)
  │    │    ├─► QdrantConfig (from bin.config)
  │    │    ├─► QdrantClient
  │    │    └─► Qdrant (LangChain wrapper - deprecated)
  │    └─► vs.add_documents()
  │
  └─► count_points() (from app.vectorstore)
       └─► QdrantClient.get_collection()