# Experiment Summary

## Overview
- Runs: 120
- Trace IDs: 120
- System metric samples: 29443
- Aggregierte Laufzeit (total_duration): 456.2032 min
- Aggregierte Laufzeit (total_wall_time): 494.1266 min

## Textuelle Zusammenfassung (Profil x Modell)
### Profil `low` / Modell `granite3.1-dense:8b-instruct-q4_K_M`
Runs: 40
Performance: Latenz-Mean 345322.45 ms, Tokens/s-Mean 3.5285, Prompt-Tokens/s-Mean 6.6213
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9597, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 228.8426 min, Walltime-Summe 232.0232 min
Gesamtbewertung: Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.960, Latency-Score=0.464, Token-Score=0.006)

### Profil `low` / Modell `llama3.1:8b-instruct-q4_K_M`
Runs: 40
Performance: Latenz-Mean 265869.92 ms, Tokens/s-Mean 3.8045, Prompt-Tokens/s-Mean 7.4775
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.9319, Bewertung: Sehr hohe Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 176.5325 min, Walltime-Summe 179.0286 min
Gesamtbewertung: Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.932, Latency-Score=0.608, Token-Score=0.031)

### Profil `low` / Modell `qwen2.5:1.5b-instruct-q4_K_M`
Runs: 40
Performance: Latenz-Mean 121876.60 ms, Tokens/s-Mean 13.9472, Prompt-Tokens/s-Mean 36.9600
Retrieval: Recall-Mean 0.4250 (Mittlerer Recall), Retrieval-Score-Mean 0.3609 (Schwache Retrieval-Qualitaet), Interpretation: Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking
LLM Judge: Score-Mean 0.8230, Bewertung: Gute Antwortqualitaet laut LLM Judge
System: GPU-Usage-Mean n/a, GPU-Memory-Mean n/a
Laufzeit: Gesamtlaufzeit 50.8281 min, Walltime-Summe 83.0747 min
Gesamtbewertung: Gesamtbewertung: gut (Retrieval=0.361, Judge=0.823, Latency-Score=0.869, Token-Score=0.970)


## Model x Profile
| experiment_id | profile | model | run_count | latency_ms_mean | latency_ms_p95 | tokens_per_s_mean | tokens_per_s_p95 | prompt_tokens_per_s_mean | prompt_tokens_per_s_p95 | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | total_wall_time_ms_mean | total_wall_time_min_mean | total_wall_time_min_sum | total_duration_s_mean | total_duration_min_mean | total_duration_min_sum | sys_cpu_usage_mean_mean | sys_memory_usage_mean_mean | sys_ollama_proc_rss_mb_mean_mean | sys_gpu_usage_mean_mean | sys_gpu_memory_mean_mean | token_score_mean | latency_score_mean | recall@k_mean | retrieval_score_mean | llm_judge_score_mean | retrieval_interpretation | recall_interpretation | retrieval_score_interpretation | llm_judge_interpretation | overall_assessment_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | 40 | 345322.45 | 408581.0 | 3.5285 | 3.5796 | 6.6213 | 6.7538 | 1284.05 | 522.35 | 1806.4 | 348034.85 | 5.8006 | 232.0232 | 343.2639 | 5.7211 | 228.8426 | 97.7775 | 70.8675 | 5840.4369 |  |  | 0.0058 | 0.4636 | 0.425 | 0.3609 | 0.9597 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.361, Judge=0.960, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | llama3.1:8b-instruct-q4_K_M | 40 | 265869.925 | 318763.35 | 3.8045 | 3.8458 | 7.4775 | 7.6022 | 1109.0 | 436.75 | 1545.75 | 268542.975 | 4.4757 | 179.0286 | 264.7988 | 4.4133 | 176.5325 | 97.2487 | 68.7835 | 5670.6373 |  |  | 0.0313 | 0.6078 | 0.425 | 0.3609 | 0.9319 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.361, Judge=0.932, Latency-Score=0.608, Token-Score=0.031) |
| 20260215_141255 | low | qwen2.5:1.5b-instruct-q4_K_M | 40 | 121876.6 | 600589.85 | 13.9472 | 14.2023 | 36.96 | 37.8451 | 1148.1892 | 684.3514 | 1832.5405 | 124612.1 | 2.0769 | 83.0747 | 82.4239 | 1.3737 | 50.8281 | 90.1908 | 18.0359 | 1521.1063 |  |  | 0.9697 | 0.8693 | 0.425 | 0.3609 | 0.823 | Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking | Mittlerer Recall | Schwache Retrieval-Qualitaet | Gute Antwortqualitaet laut LLM Judge | Gesamtbewertung: gut (Retrieval=0.361, Judge=0.823, Latency-Score=0.869, Token-Score=0.970) |

## Test Case x Model x Profile
| experiment_id | profile | model | test_case_id | run_count | latency_ms_mean | tokens_per_s_mean | prompt_tokens_per_s_mean | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | total_duration_s_mean | total_duration_min_mean | total_duration_min_sum | total_wall_time_min_sum | recall@k_mean | ndcg@k_mean | retrieval_score_mean | llm_judge_score_mean | retrieval_interpretation | recall_interpretation | retrieval_score_interpretation | llm_judge_interpretation | overall_assessment_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-31 | 1 | 345801.0 | 3.5276 | 6.6733 | 1192.0 | 580.0 | 1772.0 | 343.693 | 5.7282 | 5.7282 | 5.8096 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-32 | 1 | 411716.0 | 3.4798 | 6.5919 | 1348.0 | 711.0 | 2059.0 | 409.5994 | 6.8267 | 6.8267 | 6.9142 | 0.0 | 0.0 | 0.0 | 0.9375 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-33 | 1 | 335065.0 | 3.5194 | 6.6329 | 1289.0 | 486.0 | 1775.0 | 332.9874 | 5.5498 | 5.5498 | 5.636 | 0.0 | 0.0 | 0.0 | 0.9375 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-34 | 1 | 315769.0 | 3.5376 | 6.6149 | 1258.0 | 435.0 | 1693.0 | 313.6516 | 5.2275 | 5.2275 | 5.3139 | 0.0 | 0.0 | 0.0 | 0.9375 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-35 | 1 | 336231.0 | 3.5475 | 6.7405 | 1148.0 | 579.0 | 1727.0 | 334.1775 | 5.5696 | 5.5696 | 5.6486 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-36 | 1 | 330408.0 | 3.5645 | 6.6462 | 1258.0 | 492.0 | 1750.0 | 327.8764 | 5.4646 | 5.4646 | 5.5505 | 0.0 | 0.0 | 0.0 | 0.85 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.850, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-37 | 1 | 320918.0 | 3.5611 | 6.6556 | 1242.0 | 469.0 | 1711.0 | 318.8518 | 5.3142 | 5.3142 | 5.3844 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-38 | 1 | 352486.0 | 3.5272 | 6.6277 | 1298.0 | 543.0 | 1841.0 | 350.4037 | 5.8401 | 5.8401 | 5.921 | 0.0 | 0.0 | 0.0 | 0.9375 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-39 | 1 | 403610.0 | 3.511 | 6.6455 | 1217.0 | 763.0 | 1980.0 | 401.2781 | 6.688 | 6.688 | 6.7652 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-N-40 | 1 | 390891.0 | 3.497 | 6.6193 | 1243.0 | 700.0 | 1943.0 | 388.7313 | 6.4789 | 6.4789 | 6.5746 | 0.0 | 0.0 | 0.0 | 0.9125 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.912, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-01 | 1 | 354007.0 | 3.5065 | 6.0978 | 1402.0 | 394.0 | 1796.0 | 353.2433 | 5.8874 | 5.8874 | 5.9312 | 1.0 | 1.0 | 1.0 | 1.0 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=1.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-02 | 1 | 297855.0 | 3.5281 | 6.6609 | 1233.0 | 383.0 | 1616.0 | 294.1271 | 4.9021 | 4.9021 | 5.011 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-03 | 1 | 319611.0 | 3.5456 | 6.5914 | 1334.0 | 411.0 | 1745.0 | 318.7803 | 5.313 | 5.313 | 5.3813 | 1.0 | 0.6309 | 0.8155 | 1.0 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.815, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-04 | 1 | 394077.0 | 3.5054 | 6.6469 | 1264.0 | 703.0 | 1967.0 | 391.4819 | 6.5247 | 6.5247 | 6.6127 | 0.0 | 0.0 | 0.0 | 0.85 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.850, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-05 | 1 | 321560.0 | 3.5396 | 6.6572 | 1217.0 | 479.0 | 1696.0 | 318.6821 | 5.3114 | 5.3114 | 5.4051 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-06 | 1 | 359378.0 | 3.4959 | 6.5896 | 1358.0 | 525.0 | 1883.0 | 356.8583 | 5.9476 | 5.9476 | 6.031 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-07 | 1 | 351631.0 | 3.4991 | 6.5652 | 1359.0 | 496.0 | 1855.0 | 349.3158 | 5.8219 | 5.8219 | 5.9045 | 1.0 | 0.4307 | 0.7153 | 1.0 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.715, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-08 | 1 | 354929.0 | 3.5148 | 6.516 | 1433.0 | 465.0 | 1898.0 | 352.7509 | 5.8792 | 5.8792 | 5.9621 | 1.0 | 0.6309 | 0.8155 | 0.85 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.815, Judge=0.850, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-09 | 1 | 328737.0 | 3.5343 | 6.5773 | 1356.0 | 424.0 | 1780.0 | 326.6196 | 5.4437 | 5.4437 | 5.5339 | 1.0 | 0.5 | 0.75 | 0.9375 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.750, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-10 | 1 | 344601.0 | 3.5443 | 6.7256 | 1164.0 | 598.0 | 1762.0 | 342.4636 | 5.7077 | 5.7077 | 5.7883 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-11 | 1 | 398742.0 | 3.5477 | 6.7533 | 1122.0 | 813.0 | 1935.0 | 396.1787 | 6.603 | 6.603 | 6.6932 | 0.0 | 0.0 | 0.0 | 0.85 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=0.850, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-12 | 1 | 327171.0 | 3.5544 | 6.5911 | 1331.0 | 435.0 | 1766.0 | 324.8274 | 5.4138 | 5.4138 | 5.5096 | 1.0 | 1.0 | 1.0 | 0.9375 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=1.000, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-13 | 1 | 330784.0 | 3.542 | 6.6104 | 1292.0 | 470.0 | 1762.0 | 328.6826 | 5.478 | 5.478 | 5.5553 | 1.0 | 0.3869 | 0.6934 | 1.0 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.693, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-14 | 1 | 375727.0 | 3.4782 | 6.5971 | 1389.0 | 565.0 | 1954.0 | 373.6278 | 6.2271 | 6.2271 | 6.3023 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-15 | 1 | 431234.0 | 3.5001 | 6.6739 | 1241.0 | 848.0 | 2089.0 | 429.1377 | 7.1523 | 7.1523 | 7.2276 | 1.0 | 1.0 | 1.0 | 1.0 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=1.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-16 | 1 | 347472.0 | 3.525 | 6.6014 | 1351.0 | 494.0 | 1845.0 | 345.3577 | 5.756 | 5.756 | 5.8376 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-17 | 1 | 338926.0 | 3.4981 | 6.5036 | 1521.0 | 355.0 | 1876.0 | 335.7821 | 5.5964 | 5.5964 | 5.7032 | 1.0 | 1.0 | 1.0 | 0.9375 | Sehr gut - Relevante Dokumente gefunden und gut gerankt | Sehr hoher Recall | Sehr starke Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=1.000, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-18 | 1 | 303235.0 | 3.5991 | 6.763 | 1114.0 | 488.0 | 1602.0 | 300.8698 | 5.0145 | 5.0145 | 5.096 | 0.0 | 0.0 | 0.0 | 1.0 | Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt | Sehr niedriger Recall | Sehr schwache Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: schwach (Retrieval=0.000, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-19 | 1 | 277995.0 | 3.6227 | 6.7644 | 1105.0 | 406.0 | 1511.0 | 275.898 | 4.5983 | 4.5983 | 4.6769 | 1.0 | 0.3562 | 0.6781 | 1.0 | Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen) | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.678, Judge=1.000, Latency-Score=0.464, Token-Score=0.006) |
| 20260215_141255 | low | granite3.1-dense:8b-instruct-q4_K_M | TC-P-20 | 1 | 270324.0 | 3.5769 | 6.6984 | 1241.0 | 301.0 | 1542.0 | 269.7906 | 4.4965 | 4.4965 | 4.5501 | 1.0 | 0.6309 | 0.8155 | 0.9375 | Gut - Relevante Dokumente gefunden, Ranking könnte besser sein | Sehr hoher Recall | Gute Retrieval-Qualitaet | Sehr hohe Antwortqualitaet laut LLM Judge | Gesamtbewertung: mittel (Retrieval=0.815, Judge=0.938, Latency-Score=0.464, Token-Score=0.006) |

<!-- BEGIN CODEx EXTENDED ANALYSIS -->

## Manuell ergänzte Auswertung (Codex)

Diese Sektion ergänzt die automatisch erzeugte `summary.md` um konsolidierte Ergebnisse aus `summary_extended_runs_260215_014.md` sowie 5 zusätzliche Analysen (Korrelationen, Fehleranalyse, Tail-Latency, Warmup/Coldstart, Retrieval-vs-Judge-Muster).

### A. Wichtigste Ergebnisse (konsolidiert)

| Metrik | Wert |
| --- | --- |
| Runs / Testfälle / Modelle | 120 / 40 / 3 |
| Fehlerquote | 2.5 % (3 Runs) |
| Retrieval-Score Ø | 0.361 |
| LLM-Judge Ø | 0.907 |
| Latenz Ø [s] / P95 [s] | 244.4 / 405.8 |
| Tokens/s Ø | 6.92 |
| System raw CPU Ø / P95 [%] | 96.66 / 100.00 |
| System raw RAM Ø / P95 [%] | 61.13 / 71.00 |
| Ollama Proc CPU raw Ø / P95 [%] | 385.76 / 400.30 |
| Ollama Proc RSS raw Ø / P95 [MB] | 5044.7 / 5847.5 |
| Embedding raw Coverage | 100.0 % (120/120) |
| Embedding Dauer raw Ø / P95 [ms] | 1092.8 / 3462.9 |

### B. Modellvergleich (kompakt, inkl. Embedding-Aggregate)

| Modell | Runs | Fehler | Recall | nDCG | Judge | Latenz Ø [s] | tok/s Ø | Emb Dauer Ø [ms] | Emb CPU Ø |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| granite3.1-dense:8b-instruct-q4_K_M | 40 | 0 | 0.425 | 0.297 | 0.960 | 345.3 | 3.53 | 2091.2 | 3.33 |
| llama3.1:8b-instruct-q4_K_M | 40 | 0 | 0.425 | 0.297 | 0.932 | 265.9 | 3.80 | 943.1 | 4.78 |
| qwen2.5:1.5b-instruct-q4_K_M | 40 | 3 | 0.425 | 0.297 | 0.823 | 121.9 | 13.95 | n/a | n/a |

### C. Vollständigkeit der Embedding-Aggregation (`case_agg`/`model_agg`)

| Modell | Emb Samples | Emb Dauer | Emb CPU | Emb RAM | Emb Proc RSS |
| --- | --- | --- | --- | --- | --- |
| granite3.1-dense:8b-instruct-q4_K_M | 40/40 | 6/40 | 6/40 | 6/40 | 6/40 |
| llama3.1:8b-instruct-q4_K_M | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| qwen2.5:1.5b-instruct-q4_K_M | 40/40 | 0/40 | 0/40 | 0/40 | 0/40 |

Interpretation: `embedding_metrics_*.csv` ist vollständig vorhanden, aber die Aggregation in `case_agg`/`model_agg` ist für `granite` teilweise und für `qwen` (Dauer/CPU/RAM) leer. Embedding-Vergleiche auf Aggregationsebene deshalb nur eingeschränkt belastbar.

### 1) Korrelationen (Pearson)

| Ebene | n | Retrieval vs Judge | Retrieval vs Latenz | Total Tokens vs Latenz |
| --- | --- | --- | --- | --- |
| Gesamt | 120 | 0.196 | -0.006 | 0.041 |
| granite3.1-dense:8b-instruct-q4_K_M | 40 | 0.088 | -0.019 | 0.970 |
| llama3.1:8b-instruct-q4_K_M | 40 | 0.232 | 0.193 | 0.977 |
| qwen2.5:1.5b-instruct-q4_K_M | 40 | 0.345 | -0.046 | 0.979 |

Interpretation: `Retrieval vs Judge` zeigt, ob bessere Retrieval-Qualität tatsächlich zu besseren Antworten führt; `Total Tokens vs Latenz` trennt Antwortlänge von Modellgeschwindigkeit.

### 2) Fehleranalyse (`qwen`-Fehlerfälle)

| Testfall | Trace ID | Latenz [s] | Wall [s] | Load [s] | CPU % Ø | RAM % Ø | Ollama CPU % Ø | Ollama RSS MB Ø | Fehlende Felder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TC-N-37 | 13cdeec4-f437-471b-aa34-e763ed01c794 | 600.6 | 600.6 | n/a | 96.22 | 17.91 | 383.52 | 1510.9 | llm_judge_score, tokens_per_s, completion_tokens, eval_duration_s |
| TC-N-40 | b5d6ddf3-709f-4c8c-b010-c9cf8753f69a | 600.6 | 600.6 | n/a | 96.41 | 17.94 | 384.24 | 1512.1 | llm_judge_score, tokens_per_s, completion_tokens, eval_duration_s |
| TC-P-20 | 40b37912-979d-41e4-8274-526cc08d51c3 | 600.7 | 600.7 | n/a | 96.19 | 17.86 | 383.34 | 1509.9 | llm_judge_score, tokens_per_s, completion_tokens, eval_duration_s |

- Fehlerbild: 3 Fehler-Runs, davon 3 mit Timeout-artiger Latenz nahe 600 s (harte Obergrenze wahrscheinlich).
- Typisches Muster: `error_flag=1` plus fehlende `llm_judge_score`/`tokens_per_s`/`completion_tokens` deutet auf Abbruch vor vollständiger Antwort/Judge-Pipeline hin.

### 3) Tail-Latency nach Modell und Falltyp (`TC-N` / `TC-P`)

| Modell | Falltyp | Runs | Mean [s] | P95 [s] | P99 [s] | Max [s] | Fehlerquote % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| granite3.1-dense:8b-instruct-q4_K_M | TC-N | 10 | 354.3 | 408.1 | 411.0 | 411.7 | 0.0 |
| granite3.1-dense:8b-instruct-q4_K_M | TC-P | 30 | 342.3 | 407.2 | 424.6 | 431.2 | 0.0 |
| llama3.1:8b-instruct-q4_K_M | TC-N | 10 | 262.2 | 305.5 | 316.0 | 318.6 | 0.0 |
| llama3.1:8b-instruct-q4_K_M | TC-P | 30 | 267.1 | 319.7 | 322.4 | 322.6 | 0.0 |
| qwen2.5:1.5b-instruct-q4_K_M | TC-N | 10 | 180.3 | 600.6 | 600.6 | 600.6 | 20.0 |
| qwen2.5:1.5b-instruct-q4_K_M | TC-P | 30 | 102.4 | 126.8 | 463.8 | 600.7 | 3.3 |

Interpretation: Besonders bei `qwen` treiben einzelne Timeout-/Ausreißerfälle die Tail-Latenz stark nach oben, obwohl der Mittelwert niedrig ist.

### 4) Warmup / Coldstart (erste 5 vs letzte 5 Runs je Modell)

| Modell | Segment | Runs | Load Ø [s] | Latenz Ø [s] | Ollama RSS Ø [MB] | Ollama CPU Ø [%] |
| --- | --- | --- | --- | --- | --- | --- |
| granite3.1-dense:8b-instruct-q4_K_M | erste 5 | 5 | 2.160 | 337.4 | 5817.9 | 388.5 |
| granite3.1-dense:8b-instruct-q4_K_M | letzte 5 | 5 | 0.062 | 359.7 | 5845.4 | 390.7 |
| llama3.1:8b-instruct-q4_K_M | erste 5 | 5 | 1.695 | 259.9 | 5639.0 | 387.8 |
| llama3.1:8b-instruct-q4_K_M | letzte 5 | 5 | 0.213 | 261.8 | 5677.5 | 388.4 |
| qwen2.5:1.5b-instruct-q4_K_M | erste 5 | 5 | 0.851 | 87.4 | 1535.5 | 355.8 |
| qwen2.5:1.5b-instruct-q4_K_M | letzte 5 | 5 | 0.174 | 283.8 | 1517.6 | 367.4 |

Delta (`letzte 5 - erste 5`):

| Modell | Δ Load [s] | Δ Latenz [s] | Δ RSS [MB] | Δ Ollama CPU [%] |
| --- | --- | --- | --- | --- |
| granite3.1-dense:8b-instruct-q4_K_M | -2.098 | 22.2 | 27.5 | 2.2 |
| llama3.1:8b-instruct-q4_K_M | -1.482 | 1.9 | 38.5 | 0.6 |
| qwen2.5:1.5b-instruct-q4_K_M | -0.677 | 196.5 | -17.9 | 11.6 |

Interpretation: Zeigt Warmup-Effekte (sinkende Load-Duration) oder Drift/Memory-Buildup (steigendes RSS) über die Laufzeit des Experiments.

### 5) Retrieval-vs-Judge-Muster (Cluster)

Definitionen: `RH_JH` = Retrieval hoch (>=0.7) & Judge hoch (>=0.9), `RL_JH` = Retrieval niedrig (<0.2) & Judge hoch, `RH_JL` = Retrieval hoch & Judge niedrig (<0.8), `RL_JL` = beide niedrig.

| Ebene | n | RH_JH | RH_JL | RL_JH | RL_JL | MID | MISSING |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gesamt | 120 | 29 | 3 | 38 | 6 | 41 | 3 |
| granite3.1-dense:8b-instruct-q4_K_M | 40 | 12 | 0 | 19 | 0 | 9 | 0 |
| llama3.1:8b-instruct-q4_K_M | 40 | 10 | 0 | 16 | 0 | 14 | 0 |
| qwen2.5:1.5b-instruct-q4_K_M | 40 | 7 | 3 | 3 | 6 | 18 | 3 |

Beispiel-Fälle (zur manuellen Nachprüfung):

| Cluster | Testfall | Modell | Retr.Score | Judge | Latenz [s] | Err |
| --- | --- | --- | --- | --- | --- | --- |
| RL_JH | TC-N-31 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 1.00 | 345.8 | 0 |
| RL_JH | TC-N-32 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 0.94 | 411.7 | 0 |
| RL_JH | TC-N-32 | llama3.1:8b-instruct-q4_K_M | 0.00 | 1.00 | 318.6 | 0 |
| RL_JH | TC-N-33 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 0.94 | 335.1 | 0 |
| RL_JH | TC-N-34 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 0.94 | 315.8 | 0 |
| RL_JH | TC-N-35 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 1.00 | 336.2 | 0 |
| RL_JH | TC-N-35 | llama3.1:8b-instruct-q4_K_M | 0.00 | 0.94 | 257.3 | 0 |
| RL_JH | TC-N-37 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 1.00 | 320.9 | 0 |
| RL_JH | TC-N-37 | llama3.1:8b-instruct-q4_K_M | 0.00 | 0.94 | 226.7 | 0 |
| RL_JH | TC-N-38 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 0.94 | 352.5 | 0 |
| RL_JH | TC-N-38 | llama3.1:8b-instruct-q4_K_M | 0.00 | 0.94 | 289.4 | 0 |
| RL_JH | TC-N-39 | granite3.1-dense:8b-instruct-q4_K_M | 0.00 | 1.00 | 403.6 | 0 |
| RH_JL | TC-P-08 | qwen2.5:1.5b-instruct-q4_K_M | 0.82 | 0.79 | 82.7 | 0 |
| RH_JL | TC-P-15 | qwen2.5:1.5b-instruct-q4_K_M | 1.00 | 0.77 | 78.0 | 0 |
| RH_JL | TC-P-22 | qwen2.5:1.5b-instruct-q4_K_M | 1.00 | 0.75 | 85.5 | 0 |

Interpretation: Hohe Anzahl `RL_JH` wäre ein Warnsignal für Judge-/Prompt-Robustheit oder Halluzinationsrisiko trotz schwachem Retrieval.

<!-- END CODEx EXTENDED ANALYSIS -->
