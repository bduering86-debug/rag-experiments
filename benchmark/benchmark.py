from bin.metrics_utils import OllamaMetrics
from bin.benchmark import append_benchmark
from bin.logging_utils import log_ollama_metrics

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
    metrics = OllamaMetrics.from_ollama_response(
        data=data,
        model=model,
        phase=phase,
        key=key,
        wall_s=wall_s,
        text_len=text_len
    )

    # 2) In Logfile schreiben
    log_ollama_metrics(metrics)

    # 3) In CSV schreiben
    append_benchmark(
        model=metrics.model,
        phase=metrics.phase,
        key=metrics.key,
        wall_s=metrics.wall_s,
        load_ms=metrics.load_ms,
        eval_ms=metrics.eval_ms,
        eval_tokens=metrics.eval_tokens,
        tokens_per_s=metrics.tokens_per_s,
        response_chars=metrics.response_chars,
    )

    return text
