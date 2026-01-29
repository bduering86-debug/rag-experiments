from rag_csv.utils.metrics import OllamaRunMetrics

def call_ollama_generate(model, prompt, temperature=0.8, phase="", key=""):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9}
    }

    start = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    
    wall_s = time.time() - start
    text = data.get("response","").strip()
    text_len = len(text)

    # 1) Metrik erstellen
    metrics = OllamaRunMetrics.from_ollama_response(
        data=data,
        model=model,
        phase=phase,
        key=key,
        wall_s=wall_s,
        text_len=text_len
    )

    # 2) In Logfile schreiben (optional - entfernt wegen Refactoring)
    # log_ollama_metrics(metrics)

    # 3) In CSV schreiben (optional - entfernt wegen Refactoring)
    # append_benchmark(...)

    return text
