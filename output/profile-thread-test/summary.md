# Profile Thread Test Summary

- Timestamp: 2026-02-19T14:38:49.998374
- Prompt: `Warum ist der Himmel Blau?`
- Models: `llama3.1:8b-instruct-q4_K_M, granite3.1-dense:8b-instruct-q4_K_M, qwen2.5:1.5b-instruct-q4_K_M`
- Runs per config: 3

## Resolved Endpoints

| Profile | Ollama URL | Metrics Endpoint |
|---|---|---|
| low | `http://192.168.178.120:11434` | `http://192.168.178.120:8080/metrics` |
| mid | `http://192.168.178.126:11434` | `http://192.168.178.126:8080/metrics` |
| high | `http://192.168.178.122:11434` | `http://192.168.178.122:8080/metrics` |

## Means per Profile/Threads

| Model | Profile | Threads | Runs | Success Rate | Latency Mean (s) | Tokens/s Mean | Ollama CPU% Mean | RAM% Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| granite3.1-dense:8b-instruct-q4_K_M | high | 4 | 3 | 1.00 | 125.215 | 4.08 | 393.26 | 24.47 |
| granite3.1-dense:8b-instruct-q4_K_M | high | 8 | 3 | 1.00 | 92.192 | 5.53 | 776.19 | 24.52 |
| granite3.1-dense:8b-instruct-q4_K_M | high | 12 | 3 | 1.00 | 101.979 | 4.98 | 1157.15 | 24.54 |
| granite3.1-dense:8b-instruct-q4_K_M | high | 14 | 3 | 1.00 | 225.226 | 2.22 | 1312.32 | 24.73 |
| granite3.1-dense:8b-instruct-q4_K_M | high | 16 | 3 | 0.33 | 291.212 | 1.80 | 1486.06 | 24.74 |
| granite3.1-dense:8b-instruct-q4_K_M | low | 4 | 3 | 1.00 | 123.877 | 4.19 | 387.56 | 69.62 |
| granite3.1-dense:8b-instruct-q4_K_M | mid | 6 | 3 | 1.00 | 97.091 | 5.27 | 584.61 | 35.79 |
| granite3.1-dense:8b-instruct-q4_K_M | mid | 7 | 3 | 1.00 | 93.159 | 5.48 | 681.41 | 35.74 |
| granite3.1-dense:8b-instruct-q4_K_M | mid | 8 | 3 | 1.00 | 90.574 | 5.63 | 775.67 | 35.74 |
| llama3.1:8b-instruct-q4_K_M | high | 4 | 3 | 1.00 | 87.249 | 4.19 | 388.64 | 23.24 |
| llama3.1:8b-instruct-q4_K_M | high | 8 | 3 | 1.00 | 60.997 | 6.04 | 761.50 | 23.27 |
| llama3.1:8b-instruct-q4_K_M | high | 12 | 3 | 1.00 | 65.599 | 5.58 | 1133.42 | 23.29 |
| llama3.1:8b-instruct-q4_K_M | high | 14 | 3 | 1.00 | 151.898 | 2.38 | 1301.98 | 23.63 |
| llama3.1:8b-instruct-q4_K_M | high | 16 | 3 | 1.00 | 212.003 | 1.69 | 1463.93 | 23.87 |
| llama3.1:8b-instruct-q4_K_M | low | 4 | 3 | 1.00 | 84.013 | 4.43 | 381.46 | 66.97 |
| llama3.1:8b-instruct-q4_K_M | mid | 6 | 3 | 1.00 | 67.475 | 5.45 | 576.53 | 34.52 |
| llama3.1:8b-instruct-q4_K_M | mid | 7 | 3 | 1.00 | 64.712 | 5.69 | 669.04 | 34.63 |
| llama3.1:8b-instruct-q4_K_M | mid | 8 | 3 | 1.00 | 64.337 | 5.72 | 763.09 | 34.62 |
| qwen2.5:1.5b-instruct-q4_K_M | high | 4 | 3 | 1.00 | 8.394 | 18.39 | 316.37 | 7.69 |
| qwen2.5:1.5b-instruct-q4_K_M | high | 8 | 3 | 1.00 | 6.168 | 25.12 | 559.61 | 7.20 |
| qwen2.5:1.5b-instruct-q4_K_M | high | 12 | 3 | 1.00 | 6.728 | 23.99 | 815.12 | 7.36 |
| qwen2.5:1.5b-instruct-q4_K_M | high | 14 | 3 | 1.00 | 36.842 | 3.62 | 1223.33 | 7.58 |
| qwen2.5:1.5b-instruct-q4_K_M | high | 16 | 3 | 1.00 | 73.499 | 1.80 | 1423.72 | 7.68 |
| qwen2.5:1.5b-instruct-q4_K_M | low | 4 | 3 | 1.00 | 8.881 | 17.61 | 307.11 | 18.87 |
| qwen2.5:1.5b-instruct-q4_K_M | mid | 6 | 3 | 1.00 | 6.973 | 23.71 | 407.98 | 10.01 |
| qwen2.5:1.5b-instruct-q4_K_M | mid | 7 | 3 | 1.00 | 6.145 | 25.37 | 501.47 | 9.47 |
| qwen2.5:1.5b-instruct-q4_K_M | mid | 8 | 3 | 1.00 | 5.977 | 26.18 | 570.02 | 9.43 |

## Ranking (Throughput + Latency)

Composite = 60% Throughput-Score + 40% Latency-Score (beide 0..1 normalisiert).

| Rank | Model | Profile | Threads | Composite | TPS Score | Latency Score |
|---:|---|---|---:|---:|---:|---:|
| 1 | qwen2.5:1.5b-instruct-q4_K_M | mid | 8 | 1.0000 | 1.0000 | 1.0000 |
| 2 | qwen2.5:1.5b-instruct-q4_K_M | mid | 7 | 0.9800 | 0.9671 | 0.9994 |
| 3 | qwen2.5:1.5b-instruct-q4_K_M | high | 8 | 0.9737 | 0.9566 | 0.9993 |
| 4 | qwen2.5:1.5b-instruct-q4_K_M | high | 12 | 0.9453 | 0.9106 | 0.9974 |
| 5 | qwen2.5:1.5b-instruct-q4_K_M | mid | 6 | 0.9380 | 0.8990 | 0.9965 |
| 6 | qwen2.5:1.5b-instruct-q4_K_M | high | 4 | 0.8059 | 0.6821 | 0.9915 |
| 7 | qwen2.5:1.5b-instruct-q4_K_M | low | 4 | 0.7860 | 0.6501 | 0.9898 |
| 8 | llama3.1:8b-instruct-q4_K_M | high | 8 | 0.4295 | 0.1778 | 0.8071 |
| 9 | llama3.1:8b-instruct-q4_K_M | mid | 8 | 0.4169 | 0.1646 | 0.7954 |
| 10 | llama3.1:8b-instruct-q4_K_M | mid | 7 | 0.4156 | 0.1633 | 0.7941 |
| 11 | llama3.1:8b-instruct-q4_K_M | high | 12 | 0.4118 | 0.1590 | 0.7910 |
| 12 | llama3.1:8b-instruct-q4_K_M | mid | 6 | 0.4059 | 0.1535 | 0.7844 |
| 13 | qwen2.5:1.5b-instruct-q4_K_M | high | 14 | 0.4041 | 0.0789 | 0.8918 |
| 14 | granite3.1-dense:8b-instruct-q4_K_M | mid | 8 | 0.3781 | 0.1612 | 0.7034 |
| 15 | granite3.1-dense:8b-instruct-q4_K_M | high | 8 | 0.3733 | 0.1571 | 0.6977 |
| 16 | granite3.1-dense:8b-instruct-q4_K_M | mid | 7 | 0.3708 | 0.1550 | 0.6943 |
| 17 | granite3.1-dense:8b-instruct-q4_K_M | mid | 6 | 0.3600 | 0.1463 | 0.6806 |
| 18 | llama3.1:8b-instruct-q4_K_M | low | 4 | 0.3578 | 0.1120 | 0.7264 |
| 19 | llama3.1:8b-instruct-q4_K_M | high | 4 | 0.3474 | 0.1022 | 0.7151 |
| 20 | granite3.1-dense:8b-instruct-q4_K_M | high | 12 | 0.3460 | 0.1343 | 0.6634 |
| 21 | qwen2.5:1.5b-instruct-q4_K_M | high | 16 | 0.3081 | 0.0046 | 0.7633 |
| 22 | granite3.1-dense:8b-instruct-q4_K_M | low | 4 | 0.2959 | 0.1020 | 0.5867 |
| 23 | granite3.1-dense:8b-instruct-q4_K_M | high | 4 | 0.2915 | 0.0978 | 0.5820 |
| 24 | llama3.1:8b-instruct-q4_K_M | high | 14 | 0.2122 | 0.0281 | 0.4884 |
| 25 | llama3.1:8b-instruct-q4_K_M | high | 16 | 0.1111 | 0.0000 | 0.2777 |
| 26 | granite3.1-dense:8b-instruct-q4_K_M | high | 14 | 0.1055 | 0.0217 | 0.2313 |
| 27 | granite3.1-dense:8b-instruct-q4_K_M | high | 16 | 0.0028 | 0.0047 | 0.0000 |