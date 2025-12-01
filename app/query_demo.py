import os, textwrap, requests
from langchain_core.documents import Document
from .vectorstore import get_vectorstore
from bin.config import OllamaConfig

ollama_cfg = OllamaConfig()


def retrieve_incidents_and_kb(query: str, k_inc: int = 3, k_kb: int = 3) -> list[Document]:
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
    Du bist ein IT-Support-Assistent in einem Incident-Management-Kontext.
    Nutze ausschließlich den folgenden Kontext (Incidents & KB-Artikel),
    um die Frage zu beantworten. Wenn dir Informationen fehlen, sage das.

    Kontext:
    {context}

    Frage: {query}

    Antwort (präzise, auf Deutsch, mit konkreten Handlungsschritten):
    """).strip()

    return prompt


def ask_ollama(prompt: str) -> str:
    if not ollama_cfg.url:
        raise RuntimeError("OLLAMA_URL ist in .env nicht gesetzt")

    resp = requests.post(
        ollama_cfg.url,
        json={
            "model": ollama_cfg.model,
            "prompt": prompt,
            "options": {"num_gpu": 0, "num_thread": ollama_cfg.threads, "num_ctx": 4096},
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
