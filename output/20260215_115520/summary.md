# Experiment Summary

## Overview
- Runs: 12
- Trace IDs: 12
- System metric samples: 128
- Aggregierte Laufzeit (total_duration): 2.0054 min
- Aggregierte Laufzeit (total_wall_time): 2.6110 min

## Textuelle Zusammenfassung (Profil x Modell)
### Profil `gpu` / Modell `granite3.1-dense:8b-instruct-q4_K_M`
Runs: 4
Performance: Latenz-Mean 12788.75 ms, Tokens/s-Mean 47.3779, Prompt-Tokens/s-Mean 1625.8183
Retrieval: Recall-Mean 0.5000 (Mittlerer Recall), Retrieval-Score-Mean 0.4167 (Mittlere Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9844, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean 67.90, GPU-Memory-Mean 6449.28
Laufzeit: Gesamtlaufzeit 0.8147 min, Walltime-Summe 1.0017 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.417, Judge=0.984, Latency-Score=0.260, Token-Score=0.011)

### Profil `gpu` / Modell `llama3.1:8b-instruct-q4_K_M`
Runs: 4
Performance: Latenz-Mean 10933.50 ms, Tokens/s-Mean 48.9765, Prompt-Tokens/s-Mean 2248.8445
Retrieval: Recall-Mean 0.5000 (Mittlerer Recall), Retrieval-Score-Mean 0.4167 (Mittlere Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9313, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean 58.25, GPU-Memory-Mean 5860.12
Laufzeit: Gesamtlaufzeit 0.6915 min, Walltime-Summe 0.9028 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.417, Judge=0.931, Latency-Score=0.376, Token-Score=0.036)

### Profil `gpu` / Modell `qwen2.5:1.5b-instruct-q4_K_M`
Runs: 4
Performance: Latenz-Mean 8058.75 ms, Tokens/s-Mean 110.0768, Prompt-Tokens/s-Mean 6092.6457
Retrieval: Recall-Mean 0.5000 (Mittlerer Recall), Retrieval-Score-Mean 0.4167 (Mittlere Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.6062, Bewertung: Mittlere Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean 38.18, GPU-Memory-Mean 8012.69
Laufzeit: Gesamtlaufzeit 0.4992 min, Walltime-Summe 0.7065 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.417, Judge=0.606, Latency-Score=0.555, Token-Score=0.975)


## Model x Profile
| experiment_id | profile | model | run_count | latency_ms_mean | latency_ms_p95 | tokens_per_s_mean | tokens_per_s_p95 | prompt_tokens_per_s_mean | prompt_tokens_per_s_p95 | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | total_wall_time_ms_mean | total_wall_time_min_mean | total_wall_time_min_sum | total_duration_s_mean | total_duration_min_mean | total_duration_min_sum | sys_cpu_usage_mean_mean | sys_memory_usage_mean_mean | sys_ollama_proc_rss_mb_mean_mean | sys_gpu_usage_mean_mean | sys_gpu_memory_mean_mean | token_score_mean | latency_score_mean | recall@k_mean | retrieval_score_mean | llm_judge_score_mean | retrieval_interpretation | recall_interpretation | retrieval_score_interpretation | llm_judge_interpretation | overall_assessment_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260215_115520 | gpu | granite3.1-dense:8b-instruct-q4_K_M | 4 | 12788.75 | 15253.55 | 47.3779 | 47.7246 | 1625.8183 | 2085.9231 | 1273.5 | 501.5 | 1775.0 | 15024.75 | 0.2504 | 1.0017 | 12.2209 | 0.2037 | 0.8147 | 6.4033 | 11.7639 | 1968.7673 | 67.9041 | 6449.2788 | 0.0111 | 0.2603 | 0.5 | 0.4167 | 0.9844 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Mittlere Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.417, Judge=0.984, Latency-Score=0.260, Token-Score=0.011) |
| 20260215_115520 | gpu | llama3.1:8b-instruct-q4_K_M | 4 | 10933.5 | 12627.25 | 48.9765 | 49.5285 | 2248.8445 | 2995.3804 | 1103.25 | 439.5 | 1542.75 | 13542.25 | 0.2257 | 0.9028 | 10.3722 | 0.1729 | 0.6915 | 6.7539 | 11.8979 | 1976.1797 | 58.2468 | 5860.125 | 0.0357 | 0.3758 | 0.5 | 0.4167 | 0.9312 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Mittlere Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.417, Judge=0.931, Latency-Score=0.376, Token-Score=0.036) |
| 20260215_115520 | gpu | qwen2.5:1.5b-instruct-q4_K_M | 4 | 8058.75 | 11212.0 | 110.0768 | 110.8588 | 6092.6457 | 9129.1688 | 1140.75 | 734.0 | 1874.75 | 10597.75 | 0.1766 | 0.7065 | 7.4887 | 0.1248 | 0.4992 | 6.259 | 13.4062 | 1975.2464 | 38.1833 | 8012.6944 | 0.9747 | 0.5548 | 0.5 | 0.4167 | 0.6062 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Mittlere Retrieval-Qualitaet | Mittlere Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.417, Judge=0.606, Latency-Score=0.555, Token-Score=0.975) |

## Test Case x Model x Profile
| experiment_id | profile | model | test_case_id | run_count | latency_ms_mean | tokens_per_s_mean | prompt_tokens_per_s_mean | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | total_duration_s_mean | total_duration_min_mean | total_duration_min_sum | total_wall_time_min_sum | recall@k_mean | ndcg@k_mean | retrieval_score_mean | llm_judge_score_mean | retrieval_interpretation | recall_interpretation | retrieval_score_interpretation | llm_judge_interpretation | overall_assessment_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260215_115520 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-31 | 1 | 9359.0 | 46.916 | 2166.9307 | 1127.0 | 385.0 | 1512.0 | 8.7706 | 0.1462 | 0.1462 | 0.1995 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.260, Token-Score=0.011) |
| 20260215_115520 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-N-32 | 1 | 15506.0 | 47.214 | 1626.8798 | 1350.0 | 665.0 | 2015.0 | 14.9564 | 0.2493 | 0.2493 | 0.2987 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.260, Token-Score=0.011) |
| 20260215_115520 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-01 | 1 | 13823.0 | 47.7391 | 1148.9416 | 1379.0 | 429.0 | 1808.0 | 13.2604 | 0.221 | 0.221 | 0.2642 | 1.0 | 1.0 | 1.0 | 0.9375 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=1.000, Judge=0.938, Latency-Score=0.260, Token-Score=0.011) |
| 20260215_115520 | gpu | granite3.1-dense:8b-instruct-q4_K_M | TC-P-02 | 1 | 12467.0 | 47.6423 | 1560.5212 | 1238.0 | 527.0 | 1765.0 | 11.8959 | 0.1983 | 0.1983 | 0.2392 | 1.0 | 0.3333 | 0.6667 | 1.0 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.667, Judge=1.000, Latency-Score=0.260, Token-Score=0.011) |
| 20260215_115520 | gpu | llama3.1:8b-instruct-q4_K_M | TC-N-31 | 1 | 10053.0 | 48.7981 | 2813.6828 | 973.0 | 440.0 | 1413.0 | 9.4993 | 0.1583 | 0.1583 | 0.2219 | 0.0 | 0.0 | 0.0 | 0.85 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.850, Latency-Score=0.376, Token-Score=0.036) |
| 20260215_115520 | gpu | llama3.1:8b-instruct-q4_K_M | TC-N-32 | 1 | 11671.0 | 49.1703 | 1859.0315 | 1194.0 | 507.0 | 1701.0 | 11.0976 | 0.185 | 0.185 | 0.239 | 0.0 | 0.0 | 0.0 | 0.9375 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.938, Latency-Score=0.376, Token-Score=0.036) |
| 20260215_115520 | gpu | llama3.1:8b-instruct-q4_K_M | TC-P-01 | 1 | 12796.0 | 49.5918 | 1295.2189 | 1164.0 | 417.0 | 1581.0 | 12.2419 | 0.204 | 0.204 | 0.2576 | 1.0 | 1.0 | 1.0 | 0.9375 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=1.000, Judge=0.938, Latency-Score=0.376, Token-Score=0.036) |
| 20260215_115520 | gpu | llama3.1:8b-instruct-q4_K_M | TC-P-02 | 1 | 9214.0 | 48.3459 | 3027.4447 | 1082.0 | 394.0 | 1476.0 | 8.65 | 0.1442 | 0.1442 | 0.1842 | 1.0 | 0.3333 | 0.6667 | 1.0 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.667, Judge=1.000, Latency-Score=0.376, Token-Score=0.036) |
| 20260215_115520 | gpu | qwen2.5:1.5b-instruct-q4_K_M | TC-N-31 | 1 | 4037.0 | 109.4476 | 9606.5078 | 1004.0 | 354.0 | 1358.0 | 3.4526 | 0.0575 | 0.0575 | 0.1039 | 0.0 | 0.0 | 0.0 | 0.075 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr niedrige Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.075, Latency-Score=0.555, Token-Score=0.975) |
| 20260215_115520 | gpu | qwen2.5:1.5b-instruct-q4_K_M | TC-N-32 | 1 | 11611.0 | 109.7369 | 6424.2478 | 1235.0 | 1179.0 | 2414.0 | 11.0506 | 0.1842 | 0.1842 | 0.2423 | 0.0 | 0.0 | 0.0 | 0.9625 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.000, Judge=0.963, Latency-Score=0.555, Token-Score=0.975) |
| 20260215_115520 | gpu | qwen2.5:1.5b-instruct-q4_K_M | TC-P-01 | 1 | 8951.0 | 110.1361 | 2875.8973 | 1206.0 | 653.0 | 1859.0 | 8.3725 | 0.1395 | 0.1395 | 0.1864 | 1.0 | 1.0 | 1.0 | 0.9375 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: sehr stark (Retrieval=1.000, Judge=0.938, Latency-Score=0.555, Token-Score=0.975) |
| 20260215_115520 | gpu | qwen2.5:1.5b-instruct-q4_K_M | TC-P-02 | 1 | 7636.0 | 110.9864 | 5463.9301 | 1118.0 | 750.0 | 1868.0 | 7.079 | 0.118 | 0.118 | 0.1739 | 1.0 | 0.3333 | 0.6667 | 0.45 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Niedrige Antwortqualitaet laut LLM Judge | Gesamtbewertung: gut (Retrieval=0.667, Judge=0.450, Latency-Score=0.555, Token-Score=0.975) |
