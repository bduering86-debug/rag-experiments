#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama Model Benchmark for ITSM/Helpdesk KB Article Generation (with heuristics + auto ratings)

What it does:
- Runs multiple Ollama models with identical prompt & options
- Measures:
  - TTFT (time-to-first-token) in ms
  - Total wall time in ms
  - Tokens/s (from Ollama eval_count & eval_duration when available)
- Evaluates output with regex/heuristics and derives simple ratings
- Writes everything into ONE CSV (no shortlist file)

Requirements:
  pip install requests
"""

from __future__ import annotations

import csv
import json
import re
import time
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests


# -----------------------------
# CONFIG
# -----------------------------

OLLAMA_BASE_URL = "http://192.168.178.120:11434"  # <- set your Ollama VM host, e.g. http://10.0.0.50:11434
OUT_CSV_PATH = "ollama_kb_benchmark_results.csv"

# Put your 10 models from your table here (exact Ollama model identifiers)
MODELS: List[str] = [
    "qwen2.5:1.5b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "phi3:3.8b-mini-4k-instruct-q4_K_M",
    "llama3.2:3b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "llama3.2:1b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M"
]

MODELS_: List[str] = [
    "gemma2:9b",
    "deepseek-r1:7b",
    "mistral-nemo:12b",
    "granite3.1-dense:8b",
]

# Keep identical across models for comparability
GEN_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,   # minimize creative variance
    "top_p": 1.0,
    "num_predict": 650,   # enough room for 400–800 words including headings + lists
    "repeat_penalty": 1.1,
    "seed": 42,
}

TIMEOUT_S = 600


# -----------------------------
# PROMPT (strict, deterministic format)
# -----------------------------

STRICT_INSTRUCTIONS = """Du bist ein ITSM/IT-Helpdesk Knowledgebase-Autor.

WICHTIG (STRIKT EINHALTEN):
- Gib AUSSCHLIESSLICH den finalen Knowledgebase-Artikel aus.
- Keine Einleitung, keine Erklärung, keine Meta-Kommentare, keine Prosa außerhalb des Artikels.
- Keine Markdown-Codeblöcke und keine Inline-Code-Formatierung.
- Keine erfundenen Tools/Portale/Schritte. Nutze NUR Active Directory Users and Computers (ADUC) und die Stichpunkte unten.
- Schreibe auf Deutsch, sachlich, praxistauglich für 1st/2nd Level.

VERWENDE GENAU DIESE ABSCHNITTSÜBERSCHRIFTEN (exakt, gleiche Schreibweise, je eine eigene Zeile):
Titel
Zweck
Voraussetzungen
Schritt-für-Schritt
Typische Fehler & Troubleshooting
Sicherheits- & Compliance-Hinweise
Abschluss & Ticketdokumentation

UMFANG:
- 400 bis 800 Wörter (nur Artikeltext).

STRUKTURVORGABEN:
- Schritt-für-Schritt: genau 8 nummerierte Schritte (1. bis 8.), jeweils eine Zeile pro Schritt.
- Typische Fehler & Troubleshooting: mindestens 4 Bulletpoints (beginnend mit "- ").
- Sicherheits- & Compliance-Hinweise: mindestens 3 Bulletpoints (beginnend mit "- ").
"""

BULLET_CONTEXT = """Stichpunkte (gegeben, nicht verändern, nicht erweitern):
- Active Directory Users and Computers (ADUC) öffnen
- Passende OU wählen (z. B. OU=Users,OU=Berlin)
- Neuer Benutzer anlegen (Vorname, Nachname, Anzeigename, SamAccountName, UPN)
- Initiales Passwort setzen und “User must change password at next logon” aktivieren (oder nach Standard)
- Konto aktivieren/prüfen
- Benutzer-Eigenschaften prüfen (Telefon, Abteilung, E-Mail falls notwendig)
- Gruppen zuweisen:
  - Mitglied in Standardgruppen (z. B. VPN-Users, M365-E3, FileShare-Dept)
  - Rollen-/Applikationsgruppen nach Ticketanforderung
  - Prüfen, ob verschachtelte Gruppen korrekt sind
- Optional: Home-Laufwerk/Profilpfad (falls genutzt) setzen
- Replikation/Anmelde-Test: ggf. kurz warten oder DC prüfen
- Dokumentation im Ticket: OU, Gruppen, Besonderheiten, Zeitpunkt
"""

FULL_PROMPT = f"{STRICT_INSTRUCTIONS}\n\n{BULLET_CONTEXT}\n"


# -----------------------------
# ERROR ANALYSIS
# -----------------------------

def analyze_error(error_msg: str, status_code: Optional[int] = None) -> Dict[str, Any]:
    """
    Analysiert Fehlermeldungen und kategorisiert sie.
    
    Returns:
        {
            "error_category": str,  # "memory", "timeout", "http_error", "network", "unknown"
            "error_detail": str,     # Detaillierte Beschreibung
            "likely_cause": str,     # Wahrscheinliche Ursache
            "suggestion": str        # Lösungsvorschlag
        }
    """
    if not error_msg:
        return {
            "error_category": "none",
            "error_detail": "",
            "likely_cause": "",
            "suggestion": ""
        }
    
    error_lower = error_msg.lower()
    
    # Memory-related errors
    if any(keyword in error_lower for keyword in ["out of memory", "oom", "memory", "cuda out of memory", "allocation failed"]):
        return {
            "error_category": "memory",
            "error_detail": "Out of Memory (OOM)",
            "likely_cause": "GPU/RAM nicht ausreichend für Modell",
            "suggestion": "Kleineres Modell verwenden oder Server-RAM/VRAM erhöhen"
        }
    
    # Timeout errors
    if any(keyword in error_lower for keyword in ["timeout", "timed out", "read timeout"]):
        return {
            "error_category": "timeout",
            "error_detail": "Request Timeout",
            "likely_cause": "Server antwortet zu langsam oder ist überlastet",
            "suggestion": "Timeout erhöhen, Server-Performance prüfen, kleineres Modell testen"
        }
    
    # HTTP 500 errors (server-side)
    if status_code == 500 or "500" in error_lower or "internal server error" in error_lower:
        return {
            "error_category": "http_error",
            "error_detail": "HTTP 500 Internal Server Error",
            "likely_cause": "Ollama-Server interner Fehler (oft Memory-Problem)",
            "suggestion": "Ollama-Server Logs prüfen, Server neustarten, kleineres Modell testen"
        }
    
    # HTTP 404 errors (model not found)
    if status_code == 404 or "404" in error_lower or "not found" in error_lower:
        return {
            "error_category": "http_error",
            "error_detail": "HTTP 404 Model Not Found",
            "likely_cause": "Modell nicht auf Server verfügbar",
            "suggestion": "Modell mit 'ollama pull <model>' herunterladen"
        }
    
    # Connection errors
    if any(keyword in error_lower for keyword in ["connection", "refused", "unreachable", "network"]):
        return {
            "error_category": "network",
            "error_detail": "Network/Connection Error",
            "likely_cause": "Ollama-Server nicht erreichbar oder läuft nicht",
            "suggestion": "Server-Konnektivität prüfen, Ollama-Service Status prüfen"
        }
    
    # JSON decode errors
    if "jsondecode" in error_lower:
        return {
            "error_category": "protocol",
            "error_detail": "JSON Decode Error",
            "likely_cause": "Ungültige Antwort vom Server",
            "suggestion": "Server-Logs prüfen, Server neustarten"
        }
    
    # Unknown error
    return {
        "error_category": "unknown",
        "error_detail": error_msg[:200],
        "likely_cause": "Unbekannter Fehler",
        "suggestion": "Vollständige Fehlermeldung und Server-Logs prüfen"
    }


# -----------------------------
# OLLAMA CALL (streaming for TTFT)
# -----------------------------

def call_ollama_generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    options: Dict[str, Any],
    timeout_s: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True, "options": options}

    started = time.perf_counter()
    ttft_ms: Optional[float] = None
    response_parts: List[str] = []
    final_meta: Dict[str, Any] = {}
    error: Optional[str] = None
    status_code: Optional[int] = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            status_code = r.status_code
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - started) * 1000.0

                try:
                    evt = json.loads(raw_line)
                except json.JSONDecodeError:
                    error = f"JSONDecodeError on line: {raw_line[:200]}"
                    continue

                if evt.get("error"):
                    error = str(evt["error"])

                if evt.get("response"):
                    response_parts.append(evt["response"])

                if evt.get("done") is True:
                    final_meta = evt
                    break

    except requests.RequestException as e:
        error = f"RequestException: {e}"
        # Try to extract status code from exception
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code

    ended = time.perf_counter()
    total_ms = (ended - started) * 1000.0

    response_text = "".join(response_parts).strip()

    if ttft_ms is None:
        ttft_ms = total_ms

    # Extract metrics from final_meta (nanoseconds)
    prompt_tokens = final_meta.get("prompt_eval_count")
    eval_tokens = final_meta.get("eval_count")
    prompt_eval_dur_ns = final_meta.get("prompt_eval_duration")
    eval_dur_ns = final_meta.get("eval_duration")

    tokens_per_s: Optional[float] = None
    if isinstance(eval_tokens, int) and isinstance(eval_dur_ns, int) and eval_dur_ns > 0:
        tokens_per_s = eval_tokens / (eval_dur_ns / 1_000_000_000.0)
    
    # Analyze error if present
    error_analysis = analyze_error(error, status_code) if error else {
        "error_category": "none",
        "error_detail": "",
        "likely_cause": "",
        "suggestion": ""
    }

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "prompt_hash": str(hash(prompt)),
        "response_text": response_text,
        "error": error,
        "error_category": error_analysis["error_category"],
        "error_detail": error_analysis["error_detail"],
        "likely_cause": error_analysis["likely_cause"],
        "suggestion": error_analysis["suggestion"],
        "http_status_code": status_code,
        "ttft_ms": round(ttft_ms, 2),
        "total_ms": round(total_ms, 2),
        "tokens_per_s": round(tokens_per_s, 4) if tokens_per_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "eval_tokens": eval_tokens,
        "prompt_eval_ms": round(prompt_eval_dur_ns / 1_000_000.0, 2) if isinstance(prompt_eval_dur_ns, int) else None,
        "eval_ms": round(eval_dur_ns / 1_000_000.0, 2) if isinstance(eval_dur_ns, int) else None,
        "total_duration_ms_reported": round(final_meta.get("total_duration", 0) / 1_000_000.0, 2)
        if isinstance(final_meta.get("total_duration"), int) else None,
        "load_duration_ms_reported": round(final_meta.get("load_duration", 0) / 1_000_000.0, 2)
        if isinstance(final_meta.get("load_duration"), int) else None,
        "options": json.dumps(options, ensure_ascii=False),
    }


# -----------------------------
# HEURISTICS / REGEX CHECKS
# -----------------------------

REQUIRED_HEADINGS = [
    "Titel",
    "Zweck",
    "Voraussetzungen",
    "Schritt-für-Schritt",
    "Typische Fehler & Troubleshooting",
    "Sicherheits- & Compliance-Hinweise",
    "Abschluss & Ticketdokumentation",
]

FORBIDDEN_PATTERNS = [
    r"```",                  # markdown code fence
    r"`[^`]+`",              # inline code
    r"\bAs an AI\b",         # english meta
    r"\bI can(?:not|'t)\b",  # english meta
    r"\bIch kann nicht\b",   # refusal/meta style
    r"(?m)^\s*(Hinweis|Anmerkung|Meta)\s*:",  # meta sections
]

def word_count(text: str) -> int:
    words = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+(?:[-'][A-Za-zÄÖÜäöüß0-9]+)*", text)
    return len(words)

def has_all_headings(text: str) -> Tuple[bool, List[str]]:
    missing = []
    for h in REQUIRED_HEADINGS:
        if not re.search(rf"(?m)^\s*{re.escape(h)}\s*$", text):
            missing.append(h)
    return (len(missing) == 0, missing)

def starts_with_title_heading(text: str) -> bool:
    return bool(re.match(r"(?ms)^\s*Titel\s*$", text))

def extract_section(text: str, heading: str) -> str:
    # Extract from heading line to next heading or end
    heading_alt = "|".join(map(re.escape, REQUIRED_HEADINGS))
    pattern = rf"(?ms)^\s*{re.escape(heading)}\s*$\s*(.+?)(?=^\s*(?:{heading_alt})\s*$|\Z)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""

def count_numbered_steps(text: str) -> int:
    block = extract_section(text, "Schritt-für-Schritt")
    if not block:
        return 0
    steps = re.findall(r"(?m)^\s*\d+\.\s+\S+", block)
    return len(steps)

def steps_are_exact_1_to_8(text: str) -> bool:
    block = extract_section(text, "Schritt-für-Schritt")
    if not block:
        return False
    # Capture first number on each line "n."
    nums = re.findall(r"(?m)^\s*(\d+)\.\s+\S+", block)
    if len(nums) != 8:
        return False
    try:
        nums_i = [int(n) for n in nums]
    except ValueError:
        return False
    return nums_i == [1, 2, 3, 4, 5, 6, 7, 8]

def count_bullets_in_section(text: str, heading: str) -> int:
    block = extract_section(text, heading)
    if not block:
        return 0
    bullets = re.findall(r"(?m)^\s*-\s+\S+", block)
    return len(bullets)

def contains_forbidden(text: str) -> Tuple[bool, List[str]]:
    hits = []
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(p)
    return (len(hits) > 0, hits)

def has_only_expected_headings(text: str) -> bool:
    # Ensure there are no extra ALL-CAPS headings etc. (soft heuristic):
    # We allow arbitrary text, but we expect exactly these headings to appear as standalone lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    heading_lines = [ln for ln in lines if ln in REQUIRED_HEADINGS]
    return len(heading_lines) == len(REQUIRED_HEADINGS)

def heuristic_evaluate(text: str) -> Dict[str, Any]:
    wc = word_count(text)

    headings_ok, missing = has_all_headings(text)
    starts_ok = starts_with_title_heading(text)

    steps_count = count_numbered_steps(text)
    exact_8_steps = steps_count == 8
    exact_1_to_8 = steps_are_exact_1_to_8(text)

    trouble_bullets = count_bullets_in_section(text, "Typische Fehler & Troubleshooting")
    compliance_bullets = count_bullets_in_section(text, "Sicherheits- & Compliance-Hinweise")

    trouble_ok = trouble_bullets >= 4
    compliance_ok = compliance_bullets >= 3
    within_word_range = 400 <= wc <= 800

    forbidden_found, forbidden_hits = contains_forbidden(text)
    only_expected_headings = has_only_expected_headings(text)

    # "Hard compliance" flag: pass only if strict format constraints are met
    hard_pass = all([
        headings_ok,
        starts_ok,
        within_word_range,
        exact_8_steps,
        exact_1_to_8,
        trouble_ok,
        compliance_ok,
        not forbidden_found,
    ])

    # Simple heuristic score 0..8
    score = sum([
        1 if headings_ok else 0,
        1 if starts_ok else 0,
        1 if within_word_range else 0,
        1 if exact_8_steps else 0,
        1 if exact_1_to_8 else 0,
        1 if trouble_ok else 0,
        1 if compliance_ok else 0,
        1 if only_expected_headings else 0,
    ])
    if forbidden_found:
        score = max(0, score - 1)

    return {
        "word_count": wc,
        "within_word_range": within_word_range,
        "headings_ok": headings_ok,
        "missing_headings": ";".join(missing) if missing else "",
        "starts_with_title": starts_ok,
        "steps_count": steps_count,
        "exact_8_steps": exact_8_steps,
        "exact_steps_1_to_8": exact_1_to_8,
        "troubleshooting_bullets": trouble_bullets,
        "troubleshooting_ok": trouble_ok,
        "compliance_bullets": compliance_bullets,
        "compliance_ok": compliance_ok,
        "only_expected_headings": only_expected_headings,
        "forbidden_found": forbidden_found,
        "forbidden_hits": ";".join(forbidden_hits) if forbidden_hits else "",
        "hard_pass": hard_pass,
        "heuristic_score_0_8": score,
    }


# -----------------------------
# AUTO RATINGS (written into CSV)
# -----------------------------

def rate_speed_tokens_per_s(tokens_per_s: Optional[float]) -> str:
    if tokens_per_s is None:
        return "unknown"
    # CPU-only friendly buckets; adjust to your environment if needed
    if tokens_per_s >= 30:
        return "A (fast)"
    if tokens_per_s >= 15:
        return "B (ok)"
    if tokens_per_s >= 8:
        return "C (slow)"
    return "D (very slow)"

def rate_latency_ttft_ms(ttft_ms: Optional[float]) -> str:
    if ttft_ms is None:
        return "unknown"
    # Interactive helpdesk feeling: TTFT matters a lot
    if ttft_ms <= 800:
        return "A (snappy)"
    if ttft_ms <= 2000:
        return "B (ok)"
    if ttft_ms <= 5000:
        return "C (slow)"
    return "D (painful)"

def rate_format_quality(hard_pass: bool, heuristic_score: int) -> str:
    if hard_pass:
        return "A (strict pass)"
    if heuristic_score >= 7:
        return "B (minor issues)"
    if heuristic_score >= 5:
        return "C (format issues)"
    return "D (fails structure)"

def overall_rating(format_grade: str, speed_grade: str, latency_grade: str, error: Optional[str]) -> str:
    if error:
        return "FAIL (error)"
    # If strict format fails badly, treat as low regardless of speed
    if format_grade.startswith("D"):
        return "D (unusable format)"
    if format_grade.startswith("C"):
        return "C (needs fixes)"
    # Otherwise combine speed+latency quickly (rough, but useful)
    if format_grade.startswith("A") and (speed_grade.startswith("A") or latency_grade.startswith("A")):
        return "A (strong)"
    if format_grade.startswith("B") and (speed_grade.startswith("A") or latency_grade.startswith("A")):
        return "B (good)"
    return "B (good)" if format_grade.startswith("A") else "C (ok)"


# -----------------------------
# CSV WRITER
# -----------------------------

CSV_FIELDS = [
    # run/meta
    "timestamp",
    "model",
    "error",
    "error_category",
    "error_detail",
    "likely_cause",
    "suggestion",
    "http_status_code",

    # performance
    "ttft_ms",
    "total_ms",
    "tokens_per_s",
    "prompt_tokens",
    "eval_tokens",
    "prompt_eval_ms",
    "eval_ms",
    "total_duration_ms_reported",
    "load_duration_ms_reported",

    # heuristics
    "word_count",
    "within_word_range",
    "headings_ok",
    "missing_headings",
    "starts_with_title",
    "steps_count",
    "exact_8_steps",
    "exact_steps_1_to_8",
    "troubleshooting_bullets",
    "troubleshooting_ok",
    "compliance_bullets",
    "compliance_ok",
    "only_expected_headings",
    "forbidden_found",
    "forbidden_hits",
    "hard_pass",
    "heuristic_score_0_8",

    # auto ratings (this is the "make shortlist logic" but stored in CSV)
    "rating_format",
    "rating_speed",
    "rating_latency",
    "rating_overall",

    # manual review placeholder
    "manual_review",

    # traceability
    "prompt_hash",
    "options",

    # content
    "response_text",
]

def append_row_csv(path: str, row: Dict[str, Any]) -> None:
    file_exists = False
    try:
        with open(path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_FIELDS})


# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    print(f"Target Ollama: {OLLAMA_BASE_URL}")
    print(f"Output CSV:    {OUT_CSV_PATH}")
    print(f"Models ({len(MODELS)}): {', '.join(MODELS)}")
    print("-" * 80)

    for i, model in enumerate(MODELS, start=1):
        print(f"[{i}/{len(MODELS)}] Testing model: {model}")

        base = call_ollama_generate_stream(
            base_url=OLLAMA_BASE_URL,
            model=model,
            prompt=FULL_PROMPT,
            options=GEN_OPTIONS,
            timeout_s=TIMEOUT_S,
        )

        text = base.get("response_text") or ""
        heur = heuristic_evaluate(text)

        rating_speed = rate_speed_tokens_per_s(base.get("tokens_per_s"))
        rating_latency = rate_latency_ttft_ms(base.get("ttft_ms"))
        rating_format = rate_format_quality(heur.get("hard_pass", False), int(heur.get("heuristic_score_0_8", 0)))
        rating_overall = overall_rating(rating_format, rating_speed, rating_latency, base.get("error"))

        row = {
            **base,
            **heur,
            "rating_format": rating_format,
            "rating_speed": rating_speed,
            "rating_latency": rating_latency,
            "rating_overall": rating_overall,
            "manual_review": "",  # keep empty
        }

        append_row_csv(OUT_CSV_PATH, row)

        status = "OK" if not row["error"] else f"ERROR: {row['error']}"
        print(
            f"  -> {status}"
            f" | overall={row['rating_overall']}"
            f" | format={row['rating_format']}"
            f" | speed={row['rating_speed']}"
            f" | latency={row['rating_latency']}"
            f" | score={row['heuristic_score_0_8']}/8"
            f" | wc={row['word_count']}"
            f" | ttft_ms={row['ttft_ms']}"
            f" | tok/s={row['tokens_per_s']}"
        )
        if row.get("error"):
            print(f"     ERROR ANALYSIS:")
            print(f"       Category: {row.get('error_category', 'unknown')}")
            print(f"       Detail: {row.get('error_detail', 'N/A')}")
            print(f"       Likely Cause: {row.get('likely_cause', 'N/A')}")
            print(f"       Suggestion: {row.get('suggestion', 'N/A')}")
            if row.get('http_status_code'):
                print(f"       HTTP Status: {row.get('http_status_code')}")
        if row["forbidden_found"]:
            print(f"     forbidden_hits={row['forbidden_hits']}")
        if not row["headings_ok"]:
            print(f"     missing_headings={row['missing_headings']}")
        if not row["only_expected_headings"]:
            print("     note=unexpected/duplicate heading lines detected (heuristic)")
        print("-" * 80)

    print("Done.")


if __name__ == "__main__":
    main()
