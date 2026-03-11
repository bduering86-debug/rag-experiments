# Experiment Summary

## Overview
- Runs: 1359
- Trace IDs: 720
- System metric samples: 97838
- Aggregierte Laufzeit (total_duration): 3136.6106 min
- Aggregierte Laufzeit (total_wall_time): 3268.2653 min

## Textuelle Zusammenfassung (Profil x Modell)
### Profil `gpu` / Modell `granite3.1-dense:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 12042.83 ms, Tokens/s-Mean 47.4428, Prompt-Tokens/s-Mean 1621.0138
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9566, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean 55.04, GPU-Memory-Mean 5864.83
Laufzeit: Gesamtlaufzeit 15.1037 min, Walltime-Summe 20.8608 min
Gesamtbewertung: Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987)

### Profil `gpu` / Modell `llama3.1:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 10277.60 ms, Tokens/s-Mean 48.8461, Prompt-Tokens/s-Mean 2363.2857
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9300, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean 48.68, GPU-Memory-Mean 5361.49
Laufzeit: Gesamtlaufzeit 12.7502 min, Walltime-Summe 18.2196 min
Gesamtbewertung: Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.930, Latency-Score=0.977, Token-Score=0.986)

### Profil `gpu` / Modell `qwen2.5:1.5b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 36053.47 ms, Tokens/s-Mean 109.9605, Prompt-Tokens/s-Mean 7307.5556
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.7756, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean 29.99, GPU-Memory-Mean 2750.57
Laufzeit: Gesamtlaufzeit 46.9284 min, Walltime-Summe 54.9172 min
Gesamtbewertung: Gesamtbewertung: gut (Retrieval=0.361, Judge=0.776, Latency-Score=0.915, Token-Score=0.986)

### Profil `high` / Modell `granite3.1-dense:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 285601.28 ms, Tokens/s-Mean 4.1238, Prompt-Tokens/s-Mean 8.3156
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9488, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 379.0995 min, Walltime-Summe 384.7825 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.949, Latency-Score=0.315, Token-Score=0.014)

### Profil `high` / Modell `granite3.1-dense:8b-instruct-q8_0`
Runs: 80
Performance: Latenz-Mean 314931.35 ms, Tokens/s-Mean 2.9372, Prompt-Tokens/s-Mean 10.3890
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9613, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 417.6226 min, Walltime-Summe 424.0013 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.961, Latency-Score=0.244, Token-Score=0.473)

### Profil `high` / Modell `llama3.1:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 214013.62 ms, Tokens/s-Mean 4.6554, Prompt-Tokens/s-Mean 9.4043
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9256, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 284.3963 min, Walltime-Summe 289.4642 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.926, Latency-Score=0.487, Token-Score=0.020)

### Profil `high` / Modell `llama3.1:8b-instruct-q8_0`
Runs: 80
Performance: Latenz-Mean 226509.48 ms, Tokens/s-Mean 3.2895, Prompt-Tokens/s-Mean 11.8205
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9144, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 301.0546 min, Walltime-Summe 306.4689 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.914, Latency-Score=0.457, Token-Score=0.405)

### Profil `high` / Modell `qwen2.5:1.5b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 68484.98 ms, Tokens/s-Mean 18.5089, Prompt-Tokens/s-Mean 46.3566
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.8043, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 90.3528 min, Walltime-Summe 96.2336 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.804, Latency-Score=0.837, Token-Score=0.051)

### Profil `high` / Modell `qwen2.5:1.5b-instruct-q8_0`
Runs: 80
Performance: Latenz-Mean 66313.77 ms, Tokens/s-Mean 14.1356, Prompt-Tokens/s-Mean 58.1004
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.7917, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 67.3534 min, Walltime-Summe 92.7248 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.792, Latency-Score=0.843, Token-Score=0.545)

### Profil `low` / Modell `granite3.1-dense:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 354323.75 ms, Tokens/s-Mean 3.5321, Prompt-Tokens/s-Mean 6.7091
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9603, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 469.7545 min, Walltime-Summe 476.3419 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.960, Latency-Score=0.150, Token-Score=0.001)

### Profil `low` / Modell `llama3.1:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 259461.90 ms, Tokens/s-Mean 3.8001, Prompt-Tokens/s-Mean 7.5857
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9216, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 344.9822 min, Walltime-Summe 350.4559 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.922, Latency-Score=0.378, Token-Score=0.001)

### Profil `low` / Modell `qwen2.5:1.5b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 92074.23 ms, Tokens/s-Mean 13.8373, Prompt-Tokens/s-Mean 37.4076
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.8030, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 101.6918 min, Walltime-Summe 126.8211 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.803, Latency-Score=0.781, Token-Score=0.003)

### Profil `mid` / Modell `granite3.1-dense:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 206430.17 ms, Tokens/s-Mean 4.6803, Prompt-Tokens/s-Mean 13.2145
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9613, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 274.2870 min, Walltime-Summe 279.0953 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.961, Latency-Score=0.506, Token-Score=0.027)

### Profil `mid` / Modell `granite3.1-dense:8b-instruct-q6_K`
Runs: 40
Performance: Latenz-Mean 933.42 ms, Tokens/s-Mean n/a, Prompt-Tokens/s-Mean n/a
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean n/a, Bewertung: Sehr niedrige Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit n/a min, Walltime-Summe 0.6225 min
Gesamtbewertung: Gesamtbewertung: sehr schwach (Retrieval=0.361, Judge=nan, Latency-Score=1.000, Token-Score=nan)

### Profil `mid` / Modell `llama3.1:8b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 159124.85 ms, Tokens/s-Mean 5.2996, Prompt-Tokens/s-Mean 14.9722
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9213, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 211.2089 min, Walltime-Summe 216.1196 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.921, Latency-Score=0.619, Token-Score=0.034)

### Profil `mid` / Modell `llama3.1:8b-instruct-q6_K`
Runs: 40
Performance: Latenz-Mean 1547.25 ms, Tokens/s-Mean n/a, Prompt-Tokens/s-Mean n/a
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean n/a, Bewertung: Sehr niedrige Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit n/a min, Walltime-Summe 1.0315 min
Gesamtbewertung: Gesamtbewertung: sehr schwach (Retrieval=0.361, Judge=nan, Latency-Score=0.998, Token-Score=nan)

### Profil `mid` / Modell `qwen2.5:1.5b-instruct-q4_K_M`
Runs: 80
Performance: Latenz-Mean 47221.10 ms, Tokens/s-Mean 21.8718, Prompt-Tokens/s-Mean 73.0614
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.8119, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 62.0036 min, Walltime-Summe 67.2515 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.812, Latency-Score=0.889, Token-Score=0.086)

### Profil `mid` / Modell `qwen2.5:1.5b-instruct-q6_k`
Runs: 79
Performance: Latenz-Mean 44781.20 ms, Tokens/s-Mean 18.3200, Prompt-Tokens/s-Mean 58.3028
Retrieval: Recall-Mean 0.4177 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.8034, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 58.0211 min, Walltime-Summe 62.8529 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.803, Latency-Score=0.894, Token-Score=0.485)


## Model x Profile
| experiment_id | profile | model | run_count | latency_ms_mean | latency_ms_p95 | tokens_per_s_mean | tokens_per_s_p95 | prompt_tokens_per_s_mean | prompt_tokens_per_s_p95 | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | total_wall_time_ms_mean | total_wall_time_min_mean | total_wall_time_min_sum | total_duration_s_mean | total_duration_min_mean | total_duration_min_sum | sys_cpu_usage_mean_mean | sys_memory_usage_mean_mean | sys_ollama_proc_rss_mb_mean_mean | sys_gpu_usage_mean_mean | sys_gpu_memory_mean_mean | latency_score_mean | retrieval_score_cfg_mean | answer_score_cfg_mean | recall@k_mean | retrieval_score_mean | llm_judge_score_mean | retrieval_interpretation | recall_interpretation | retrieval_score_interpretation | llm_judge_interpretation | overall_assessment_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | 80 | 12042.825 | 15847.15 | 47.4428 | 47.9969 | 1621.0138 | 2133.5444 | 1284.05 | 489.25 | 1773.3 | 15645.575 | 0.2608 | 20.8608 | 11.3277 | 0.1888 | 15.1037 | 4.8628 | 2.6981 | 197.4577 | 55.045 | 5864.8269 | 0.9732 | 0.3609 | 0.9566 | 0.425 | 0.3609 | 0.9566 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | llama3.1:8b-instruct-q4_K_M | 80 | 10277.6 | 12073.05 | 48.8461 | 49.4586 | 2363.2857 | 3037.6102 | 1109.0 | 425.45 | 1534.45 | 13664.675 | 0.2277 | 18.2196 | 9.5627 | 0.1594 | 12.7502 | 4.5329 | 2.6496 | 156.3574 | 48.6833 | 5361.4876 | 0.9775 | 0.3609 | 0.93 | 0.425 | 0.3609 | 0.93 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.930, Latency-Score=0.977, Token-Score=0.986) |
| 20260223_161322 | gpu | qwen2.5:1.5b-instruct-q4_K_M | 80 | 36053.475 | 387476.0 | 109.9605 | 111.2291 | 7307.5556 | 10307.1664 | 1146.05 | 3711.75 | 4857.8 | 41187.925 | 0.6865 | 54.9172 | 35.1963 | 0.5866 | 46.9284 | 4.0455 | 3.5114 | 269.528 | 29.986 | 2750.5732 | 0.9155 | 0.3609 | 0.7756 | 0.425 | 0.3609 | 0.7756 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: gut (Retrieval=0.361, Judge=0.776, Latency-Score=0.915, Token-Score=0.986) |
| 20260223_161322 | high | granite3.1-dense:8b-instruct-q4_K_M | 80 | 285601.275 | 330490.0 | 4.1238 | 4.1865 | 8.3156 | 8.4863 | 1284.05 | 530.6 | 1814.65 | 288586.9 | 4.8098 | 384.7825 | 284.3246 | 4.7387 | 379.0995 | 30.8541 | 23.3879 | 5756.8966 |  |  | 0.315 | 0.3609 | 0.9488 | 0.425 | 0.3609 | 0.9488 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.949, Latency-Score=0.315, Token-Score=0.014) |
| 20260223_161322 | high | granite3.1-dense:8b-instruct-q8_0 | 80 | 314931.35 | 395018.4 | 2.9372 | 2.9708 | 10.389 | 10.6097 | 1284.05 | 553.075 | 1837.125 | 318000.975 | 5.3 | 424.0013 | 313.217 | 5.2203 | 417.6226 | 30.8973 | 38.177 | 9390.7097 |  |  | 0.2445 | 0.3609 | 0.9612 | 0.425 | 0.3609 | 0.9612 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.961, Latency-Score=0.244, Token-Score=0.473) |
| 20260223_161322 | high | llama3.1:8b-instruct-q4_K_M | 80 | 214013.625 | 247230.85 | 4.6554 | 4.7264 | 9.4043 | 9.5595 | 1109.0 | 435.975 | 1544.975 | 217098.175 | 3.6183 | 289.4642 | 213.2972 | 3.555 | 284.3963 | 30.5706 | 22.6858 | 5584.8791 |  |  | 0.4873 | 0.3609 | 0.9256 | 0.425 | 0.3609 | 0.9256 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.926, Latency-Score=0.487, Token-Score=0.020) |
| 20260223_161322 | high | llama3.1:8b-instruct-q8_0 | 80 | 226509.475 | 264679.95 | 3.2895 | 3.3207 | 11.8205 | 12.0121 | 1109.0 | 427.875 | 1536.875 | 229851.7 | 3.8309 | 306.4689 | 225.791 | 3.7632 | 301.0546 | 30.6449 | 37.2096 | 9148.0145 |  |  | 0.4572 | 0.3609 | 0.9144 | 0.425 | 0.3609 | 0.9144 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.914, Latency-Score=0.457, Token-Score=0.405) |
| 20260223_161322 | high | qwen2.5:1.5b-instruct-q4_K_M | 80 | 68484.975 | 104422.1 | 18.5089 | 18.8151 | 46.3566 | 47.3265 | 1146.05 | 746.775 | 1892.825 | 72175.225 | 1.2029 | 96.2336 | 67.7646 | 1.1294 | 90.3528 | 27.8567 | 5.8025 | 1441.4274 |  |  | 0.8374 | 0.3609 | 0.8043 | 0.425 | 0.3609 | 0.8043 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.804, Latency-Score=0.837, Token-Score=0.051) |
| 20260223_161322 | high | qwen2.5:1.5b-instruct-q8_0 | 80 | 66313.775 | 79339.75 | 14.1356 | 14.2921 | 58.1004 | 59.5396 | 1148.7179 | 429.7179 | 1578.4359 | 69543.625 | 1.1591 | 92.7248 | 51.8103 | 0.8635 | 67.3534 | 27.5497 | 8.8646 | 2194.0962 |  |  | 0.8426 | 0.3609 | 0.7917 | 0.425 | 0.3609 | 0.7917 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.792, Latency-Score=0.843, Token-Score=0.545) |
| 20260223_161322 | low | granite3.1-dense:8b-instruct-q4_K_M | 80 | 354323.75 | 432431.7 | 3.5321 | 3.5885 | 6.7091 | 6.8228 | 1284.05 | 563.55 | 1847.6 | 357256.425 | 5.9543 | 476.3419 | 352.3159 | 5.8719 | 469.7545 | 97.7554 | 70.2097 | 5766.2769 |  |  | 0.1497 | 0.3609 | 0.9603 | 0.425 | 0.3609 | 0.9603 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.960, Latency-Score=0.150, Token-Score=0.001) |
| 20260223_161322 | low | llama3.1:8b-instruct-q4_K_M | 80 | 259461.9 | 291148.0 | 3.8001 | 3.8396 | 7.5857 | 7.6916 | 1109.0 | 422.3 | 1531.3 | 262841.95 | 4.3807 | 350.4559 | 258.7367 | 4.3123 | 344.9822 | 97.079 | 68.1805 | 5599.5488 |  |  | 0.3779 | 0.3609 | 0.9216 | 0.425 | 0.3609 | 0.9216 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.922, Latency-Score=0.378, Token-Score=0.001) |
| 20260223_161322 | low | qwen2.5:1.5b-instruct-q4_K_M | 80 | 92074.225 | 121945.05 | 13.8373 | 14.1278 | 37.4076 | 38.2595 | 1144.8974 | 629.3333 | 1774.2308 | 95115.825 | 1.5853 | 126.8211 | 78.2245 | 1.3037 | 101.6918 | 89.2863 | 17.4319 | 1449.5027 |  |  | 0.7807 | 0.3609 | 0.803 | 0.425 | 0.3609 | 0.803 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.803, Latency-Score=0.781, Token-Score=0.003) |
| 20260223_161322 | mid | granite3.1-dense:8b-instruct-q4_K_M | 80 | 206430.175 | 255520.0 | 4.6803 | 4.7347 | 13.2145 | 13.4718 | 1284.05 | 503.0 | 1787.05 | 209321.475 | 3.4887 | 279.0953 | 205.7152 | 3.4286 | 274.287 | 96.7129 | 35.0773 | 5753.1425 |  |  | 0.5055 | 0.3609 | 0.9612 | 0.425 | 0.3609 | 0.9612 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.961, Latency-Score=0.506, Token-Score=0.027) |
| 20260223_161322 | mid | granite3.1-dense:8b-instruct-q6_K | 40 | 933.425 | 969.6 |  |  |  |  |  |  |  | 933.7 | 0.0156 | 0.6225 |  |  |  | 6.6583 | 4.4125 | 712.2911 |  |  | 1.0 | 0.3609 |  | 0.425 | 0.3609 |  | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr niedrige Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr schwach (Retrieval=0.361, Judge=nan, Latency-Score=1.000, Token-Score=nan) |
| 20260223_161322 | mid | llama3.1:8b-instruct-q4_K_M | 80 | 159124.85 | 177567.75 | 5.2996 | 5.3577 | 14.9722 | 15.2204 | 1109.0 | 438.15 | 1547.15 | 162089.725 | 2.7015 | 216.1196 | 158.4067 | 2.6401 | 211.2089 | 95.3092 | 34.0332 | 5584.0616 |  |  | 0.6193 | 0.3609 | 0.9212 | 0.425 | 0.3609 | 0.9212 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.921, Latency-Score=0.619, Token-Score=0.034) |
| 20260223_161322 | mid | llama3.1:8b-instruct-q6_K | 40 | 1547.25 | 1593.3 |  |  |  |  |  |  |  | 1547.325 | 0.0258 | 1.0316 |  |  |  | 8.2794 | 3.6225 | 586.0559 |  |  | 0.9985 | 0.3609 |  | 0.425 | 0.3609 |  | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr niedrige Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr schwach (Retrieval=0.361, Judge=nan, Latency-Score=0.998, Token-Score=nan) |
| 20260223_161322 | mid | qwen2.5:1.5b-instruct-q4_K_M | 80 | 47221.1 | 67920.9 | 21.8718 | 22.1399 | 73.0614 | 74.3787 | 1146.05 | 625.55 | 1771.6 | 50438.65 | 0.8406 | 67.2515 | 46.5027 | 0.775 | 62.0036 | 83.6793 | 8.7585 | 1446.9794 |  |  | 0.8886 | 0.3609 | 0.8119 | 0.425 | 0.3609 | 0.8119 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.812, Latency-Score=0.889, Token-Score=0.086) |
| 20260223_161322 | mid | qwen2.5:1.5b-instruct-q6_k | 79 | 44781.2025 | 67189.7 | 18.32 | 18.5501 | 58.3028 | 59.4509 | 1145.0253 | 419.2532 | 1564.2785 | 47736.3797 | 0.7956 | 62.8529 | 44.0666 | 0.7344 | 58.0211 | 84.5984 | 13.7658 | 2266.9037 |  |  | 0.8942 | 0.3609 | 0.8034 | 0.4177 | 0.3528 | 0.8017 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.803, Latency-Score=0.894, Token-Score=0.485) |

## Test Case x Model x Profile
| experiment_id | profile | model | test_case_id | run_count | latency_ms_mean | tokens_per_s_mean | prompt_tokens_per_s_mean | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | total_duration_s_mean | total_duration_min_mean | total_duration_min_sum | total_wall_time_min_sum | recall@k_mean | ndcg@k_mean | retrieval_score_mean | retrieval_score_cfg_mean | llm_judge_score_mean | answer_score_cfg_mean | retrieval_interpretation | recall_interpretation | retrieval_score_interpretation | llm_judge_interpretation | overall_assessment_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-31 | 2 | 9570.0 | 47.9309 | 1518.2669 | 1192.0 | 384.0 | 1576.0 | 8.8425 | 0.1474 | 0.2947 | 0.4114 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.9375 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-32 | 2 | 14349.0 | 47.2252 | 1634.5903 | 1348.0 | 602.0 | 1950.0 | 13.6139 | 0.2269 | 0.4538 | 0.6386 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-33 | 2 | 9152.0 | 47.4382 | 1591.3004 | 1289.0 | 361.0 | 1650.0 | 8.4653 | 0.1411 | 0.2822 | 0.4508 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.9375 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-34 | 2 | 7778.0 | 47.6872 | 1575.0914 | 1258.0 | 296.0 | 1554.0 | 7.0465 | 0.1174 | 0.2349 | 0.3604 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.9375 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-35 | 2 | 10068.0 | 48.1302 | 1490.8125 | 1148.0 | 413.0 | 1561.0 | 9.3952 | 0.1566 | 0.3132 | 0.4375 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-36 | 2 | 15778.0 | 47.5117 | 1581.3227 | 1258.0 | 675.0 | 1933.0 | 15.0443 | 0.2507 | 0.5015 | 0.6533 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-37 | 2 | 9971.0 | 47.7022 | 1572.8874 | 1242.0 | 400.0 | 1642.0 | 9.2192 | 0.1537 | 0.3073 | 0.4456 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-38 | 2 | 15448.0 | 47.355 | 1583.8315 | 1298.0 | 658.0 | 1956.0 | 14.7565 | 0.2459 | 0.4919 | 0.6153 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-39 | 2 | 13017.0 | 47.7074 | 1547.6765 | 1217.0 | 548.0 | 1765.0 | 12.3181 | 0.2053 | 0.4106 | 0.5862 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-40 | 2 | 14464.0 | 47.6255 | 1558.3254 | 1243.0 | 614.0 | 1857.0 | 13.7315 | 0.2289 | 0.4577 | 0.6124 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.85 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-01 | 2 | 17161.0 | 47.6808 | 1144.8208 | 1402.0 | 405.0 | 1807.0 | 16.4249 | 0.2737 | 0.5475 | 0.6811 | 1.0 | 1.0 | 1.0 | 0.3609 | 1.0 | 0.9566 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-02 | 2 | 11330.0 | 47.7217 | 1548.4939 | 1233.0 | 465.0 | 1698.0 | 10.5816 | 0.1764 | 0.3527 | 0.4668 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-03 | 2 | 13072.0 | 47.2987 | 1611.0063 | 1334.0 | 541.0 | 1875.0 | 12.3091 | 0.2052 | 0.4103 | 0.5294 | 1.0 | 0.6309 | 0.8155 | 0.3609 | 1.0 | 0.9566 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-04 | 2 | 18986.0 | 47.3977 | 1557.3981 | 1264.0 | 826.0 | 2090.0 | 18.2813 | 0.3047 | 0.6094 | 0.7769 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.9125 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-05 | 2 | 10699.0 | 47.7844 | 1535.1823 | 1217.0 | 436.0 | 1653.0 | 9.9615 | 0.166 | 0.3321 | 0.4468 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-06 | 2 | 9881.0 | 47.0976 | 1634.1806 | 1358.0 | 392.0 | 1750.0 | 9.1947 | 0.1532 | 0.3065 | 0.4062 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-07 | 2 | 11141.0 | 47.2139 | 1620.7389 | 1359.0 | 450.0 | 1809.0 | 10.412 | 0.1735 | 0.3471 | 0.524 | 1.0 | 0.4307 | 0.7153 | 0.3609 | 1.0 | 0.9566 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-08 | 2 | 9320.0 | 46.7984 | 1669.4055 | 1433.0 | 361.0 | 1794.0 | 8.6155 | 0.1436 | 0.2872 | 0.4295 | 1.0 | 0.6309 | 0.8155 | 0.3609 | 0.85 | 0.9566 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-09 | 2 | 10583.0 | 47.3106 | 1639.7566 | 1356.0 | 425.0 | 1781.0 | 9.8556 | 0.1643 | 0.3285 | 0.5008 | 1.0 | 0.5 | 0.75 | 0.3609 | 0.9375 | 0.9566 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-10 | 2 | 12550.0 | 47.9949 | 1493.3468 | 1164.0 | 527.0 | 1691.0 | 11.8051 | 0.1968 | 0.3935 | 0.5443 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.85 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-11 | 2 | 10325.0 | 47.0808 | 2165.5612 | 1122.0 | 427.0 | 1549.0 | 9.6317 | 0.1605 | 0.3211 | 0.5207 | 0.0 | 0.0 | 0.0 | 0.3609 | 0.9375 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-12 | 2 | 10892.0 | 47.3479 | 1608.7367 | 1331.0 | 441.0 | 1772.0 | 10.1839 | 0.1697 | 0.3395 | 0.5254 | 1.0 | 1.0 | 1.0 | 0.3609 | 1.0 | 0.9566 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-13 | 2 | 13655.0 | 47.4698 | 1590.0247 | 1292.0 | 574.0 | 1866.0 | 12.9506 | 0.2158 | 0.4317 | 0.5329 | 1.0 | 0.3869 | 0.6934 | 0.3609 | 1.0 | 0.9566 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-14 | 2 | 10969.0 | 47.1083 | 1657.7629 | 1389.0 | 441.0 | 1830.0 | 10.2437 | 0.1707 | 0.3415 | 0.4794 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-15 | 2 | 10768.0 | 47.754 | 1559.7438 | 1241.0 | 441.0 | 1682.0 | 10.0736 | 0.1679 | 0.3358 | 0.4407 | 1.0 | 1.0 | 1.0 | 0.3609 | 1.0 | 0.9566 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-16 | 2 | 13692.0 | 47.2367 | 1637.1326 | 1351.0 | 571.0 | 1922.0 | 12.9581 | 0.216 | 0.4319 | 0.568 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-17 | 2 | 13140.0 | 46.6038 | 1716.6822 | 1521.0 | 535.0 | 2056.0 | 12.4079 | 0.2068 | 0.4136 | 0.5279 | 1.0 | 1.0 | 1.0 | 0.3609 | 0.9375 | 0.9566 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-18 | 2 | 11285.0 | 47.1267 | 2132.9298 | 1114.0 | 471.0 | 1585.0 | 10.5628 | 0.176 | 0.3521 | 0.5443 | 0.0 | 0.0 | 0.0 | 0.3609 | 1.0 | 0.9566 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-19 | 2 | 9782.0 | 47.1304 | 2145.2231 | 1105.0 | 401.0 | 1506.0 | 9.0635 | 0.1511 | 0.3021 | 0.4329 | 1.0 | 0.3562 | 0.6781 | 0.3609 | 0.9375 | 0.9566 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
| 20260223_161322 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-20 | 2 | 8341.0 | 47.7417 | 1562.4034 | 1241.0 | 323.0 | 1564.0 | 7.6029 | 0.1267 | 0.2534 | 0.418 | 1.0 | 0.6309 | 0.8155 | 0.3609 | 0.9375 | 0.9566 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=0.361, Judge=0.957, Latency-Score=0.973, Token-Score=0.987) |
