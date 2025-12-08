# bin/metrics_utils.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .logging_utils import get_logger

logger = get_logger("metrics")


@dataclass
class OllamaRunMetrics:
    run_id: str
    model: str
    total_tickets: int
    tickets_per_call: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    num_calls: int = 0
    total_eval_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_time: float = 0.0  # reine Zeit für Ollama-Calls (Summe der Durations)

    temperature: float = 0.0
    top_p: float = 0.0
    ctx_tokens: int = 0
    repeat_penalty: float = 0.0
    seed: Optional[int] = None
    num_predict: Optional[int] = None


_metrics: Optional[OllamaRunMetrics] = None


def start_run(
    model: str, 
    total_tickets: int, 
    tickets_per_call: int, 
    temperature: float,
    top_p: float,
    ctx_tokens: int,
    repeat_penalty: float,
    seed: int,
    num_predict: int
) -> str:
    """
    Startet einen neuen Metrics-Run für die Ticketgenerierung.
    Gibt eine run_id zurück (kannst du später in Logs/MA referenzieren).
    """
    global _metrics
    run_id = str(uuid.uuid4())
    _metrics = OllamaRunMetrics(
        run_id=run_id,
        model=model,
        total_tickets=total_tickets,
        tickets_per_call=tickets_per_call,
        temperature=temperature,
        top_p=top_p,
        ctx_tokens=ctx_tokens,
        repeat_penalty=repeat_penalty,
        seed=seed
    )


    logger.info(
        "Starte Ollama-Metrik-Run: run_id=%s, model=%s, total_tickets=%s, tickets_per_call=%s",
        run_id,
        model,
        total_tickets,
        tickets_per_call,
    )
    return run_id


def log_ollama_call(
    batch_size: int,
    duration: float,
    eval_tokens: int,
    prompt_tokens: int,
) -> None:
    """
    Pro Ollama-Call aufrufen: protokolliert Dauer und Tokenzahlen
    und akkumuliert sie für den gesamten Run.
    """
    global _metrics
    if _metrics is None:
        # Falls jemand vergisst start_run aufzurufen, nicht crashen
        logger.warning(
            "log_ollama_call wurde ohne aktiven Metrics-Run aufgerufen. "
            "Rufe zuerst start_run(...) auf."
        )
        return

    _metrics.num_calls += 1
    _metrics.total_eval_tokens += eval_tokens
    _metrics.total_prompt_tokens += prompt_tokens
    _metrics.total_llm_time += duration

    tokens_per_second = (eval_tokens / duration) if duration > 0 and eval_tokens else 0.0

    logger.info(
        "Ollama-Call #%s: batch_size=%s, duration=%.3fs, eval_tokens=%s, prompt_tokens=%s, tokens/s=%.2f",
        _metrics.num_calls,
        batch_size,
        duration,
        eval_tokens,
        prompt_tokens,
        tokens_per_second,
    )


def end_run() -> None:
    """
    Schliesst den aktuellen Metrics-Run ab und loggt eine Gesamtauswertung.
    """
    global _metrics
    if _metrics is None:
        return

    _metrics.end_time = time.time()
    wall_time = _metrics.end_time - _metrics.start_time

    avg_eval_per_call = (
        _metrics.total_eval_tokens / _metrics.num_calls
        if _metrics.num_calls
        else 0.0
    )
    avg_tokens_per_second = (
        _metrics.total_eval_tokens / _metrics.total_llm_time
        if _metrics.total_llm_time > 0
        else 0.0
    )

    logger.info(
        (
            "Ollama-Run-Summary: run_id=%s, model=%s, calls=%s, "
            "total_eval_tokens=%s, total_prompt_tokens=%s, "
            "llm_time=%.2fs, wall_time=%.2fs, "
            "avg_eval_tokens_per_call=%.1f, avg_tokens_per_second=%.2f, "
            "temperature=%.2f, top_p=%.2f, ctx_tokens=%s, repeat_penalty=%.2f, seed=%s, num_predict=%s"
        ),
        _metrics.run_id,
        _metrics.model,
        _metrics.num_calls,
        _metrics.total_eval_tokens,
        _metrics.total_prompt_tokens,
        _metrics.total_llm_time,
        wall_time,
        avg_eval_per_call,
        avg_tokens_per_second,
        _metrics.temperature,
        _metrics.top_p,
        _metrics.ctx_tokens,
        _metrics.repeat_penalty,
        _metrics.seed,
        _metrics.num_predict
    )

    # Reset für nächsten Run
    _metrics = None
