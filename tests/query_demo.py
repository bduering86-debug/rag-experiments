import os, textwrap, requests
from langchain_core.documents import Document
from rag_csv.core.vectorstore import get_vectorstore
from rag_csv.config.settings import OllamaConfig

ollama_cfg = OllamaConfig()


def retrieve_incidents_and_kb(query: str, k_inc: int = 3, k_kb: int = 5) -> list[Document]:
    vs_inc = get_vectorstore("incidents")
    vs_kb = get_vectorstore("kb")

    inc_docs = vs_inc.similarity_search(query, k=k_inc)
    kb_docs = vs_kb.similarity_search(query, k=k_kb)

    return inc_docs + kb_docs


def build_prompt(query: str, docs: list[Document]) -> str:
    context_blocks = []
    for d in docs:
        meta = d.metadata
        header = ""
        if meta.get("source") == "incident":
            header = f"[INC {meta.get('ticket_id','')}, Status: {meta.get('status','')}]\n"
        elif meta.get("source") == "kb":
            header = f"[KB {meta.get('kb_id','')}, Kategorie: {meta.get('category','')}]\n"

        context_blocks.append(header + d.page_content)

    context = "\n\n-----\n\n".join(context_blocks)

    prompt = textwrap.dedent(f"""
    Du bist ein IT-Support-Spezialist. Analysiere das gemeldete Problem und erstelle eine sachliche, strukturierte Lösung.

    Kontext aus Wissensdatenbank und früheren Incidents:
    {context}

    Gemeldetes Problem: {query}

    Erstelle eine präzise Problemanalyse und Lösungsanleitung nach folgendem Format:

    **Problemanalyse:**
    - Beschreibe kurz das identifizierte Problem

    **Lösungsschritte:**
    1. [Erster konkreter Handlungsschritt]
    2. [Zweiter konkreter Handlungsschritt]
    3. [...]

    **Relevante KB-Artikel:** [Gib die KB-IDs an, falls vorhanden]

    Hinweis: Nutze ausschließlich die Informationen aus dem bereitgestellten Kontext. Bei fehlenden Informationen weise darauf hin.
    """).strip()

    return prompt


def ask_ollama(prompt: str, model: str = "llama3.1:8b-instruct-q4_K_M") -> str:
    if not ollama_cfg.url:
        raise RuntimeError("OLLAMA_URL ist in .env nicht gesetzt")

    resp = requests.post(
        ollama_cfg.url + "/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {"num_ctx": 4096},
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "VPN bricht nach 5 Minuten ab"
    docs = retrieve_incidents_and_kb(query)
    prompt = build_prompt(query, docs)
    answer = ask_ollama(prompt)

    print("=== Frage ===")
    print(query)
    print("\n=== Antwort ===")
    print(answer)
    print("\n=== Verwendete Kontexte (IDs) ===")
    for d in docs:
        print(d.metadata.get("source"), d.metadata.get("ticket_id") or d.metadata.get("kb_id"))


if __name__ == "__main__":
    main()
