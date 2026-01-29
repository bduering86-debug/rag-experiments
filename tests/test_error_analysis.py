#!/usr/bin/env python3
"""
Quick test script to verify error analysis functionality
Tests 3 models: 1 working small model + 2 models that trigger 500 errors
"""

from rag_csv.generator.evaluation import (
    OLLAMA_BASE_URL, 
    FULL_PROMPT, 
    GEN_OPTIONS, 
    TIMEOUT_S,
    CSV_FIELDS,
    call_ollama_generate_stream,
    heuristic_evaluate,
    rate_speed_tokens_per_s,
    rate_latency_ttft_ms,
    rate_format_quality,
    overall_rating,
    append_row_csv
)

# Test only 3 models
TEST_MODELS = [
    "qwen2.5:1.5b-instruct-q4_K_M",  # Should work
    "llama3.1:8b-instruct-q4_K_M",   # Should fail with 500
    "mistral:7b-instruct-q4_K_M",    # Should fail with 500
]

TEST_CSV = "test_error_analysis_results.csv"

def main():
    print(f"Target Ollama: {OLLAMA_BASE_URL}")
    print(f"Output CSV:    {TEST_CSV}")
    print(f"Models ({len(TEST_MODELS)}): {', '.join(TEST_MODELS)}")
    print("-" * 80)

    for i, model in enumerate(TEST_MODELS, start=1):
        print(f"[{i}/{len(TEST_MODELS)}] Testing model: {model}")

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
            "manual_review": "",
        }

        append_row_csv(TEST_CSV, row)

        status = "OK" if not row["error"] else f"ERROR"
        print(f"  -> {status} | overall={row['rating_overall']}")
        
        if row.get("error"):
            print(f"     ╔══════════════════════════════════════════════════════════")
            print(f"     ║ ERROR ANALYSIS:")
            print(f"     ║   Category:     {row.get('error_category', 'unknown')}")
            print(f"     ║   Detail:       {row.get('error_detail', 'N/A')}")
            print(f"     ║   Likely Cause: {row.get('likely_cause', 'N/A')}")
            print(f"     ║   Suggestion:   {row.get('suggestion', 'N/A')}")
            if row.get('http_status_code'):
                print(f"     ║   HTTP Status:  {row.get('http_status_code')}")
            print(f"     ║   Raw Error:    {row.get('error', 'N/A')[:70]}...")
            print(f"     ╚══════════════════════════════════════════════════════════")
        else:
            print(f"     ✓ Success: {len(text)} chars generated")
            
        print("-" * 80)

    print("\n✅ Done. Check CSV for full results:")
    print(f"   {TEST_CSV}")
    
    # Show CSV columns
    print(f"\n📊 CSV Fields ({len(CSV_FIELDS)}):")
    error_fields = [f for f in CSV_FIELDS if 'error' in f.lower()]
    print(f"   Error-related fields: {error_fields}")

if __name__ == "__main__":
    main()
