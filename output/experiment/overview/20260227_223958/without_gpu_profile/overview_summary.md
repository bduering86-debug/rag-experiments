# Run Overview - without_gpu_profile

- Generated (UTC): 2026-03-06 22:45:39
- Runs file: `output/experiment/runs_260227_004.csv`

## Global
- Runs: 1800
- Traces: 1800
- Profiles: 3 (high, low, mid)
- Models: 9 (granite3.1-dense:8b-instruct-q4_K_M, granite3.1-dense:8b-instruct-q6_K, granite3.1-dense:8b-instruct-q8_0, llama3.1:8b-instruct-q4_K_M, llama3.1:8b-instruct-q6_K, llama3.1:8b-instruct-q8_0, qwen2.5:1.5b-instruct-q4_K_M, qwen2.5:1.5b-instruct-q6_k, qwen2.5:1.5b-instruct-q8_0)
- Test cases: 40
- Errors: 249 (13.83%)
- Latency mean/p95 (s): 116.085 / 326.194
- Tokens/s mean: 8.60
- LLM judge mean: 0.8843

## Top-5 by LLM Judge
```text
profile                               model  llm_judge_mean
   high   granite3.1-dense:8b-instruct-q8_0        0.966458
    mid granite3.1-dense:8b-instruct-q4_K_M        0.956356
   high granite3.1-dense:8b-instruct-q4_K_M        0.955252
    low granite3.1-dense:8b-instruct-q4_K_M        0.952648
    mid         llama3.1:8b-instruct-q4_K_M        0.929555
```

## Top-5 Fastest (Median Latency in Sekunden)
```text
profile                             model  latency_median_s  fail_rate_pct
    mid granite3.1-dense:8b-instruct-q6_K            0.9855     100.000000
    mid         llama3.1:8b-instruct-q6_K            1.6125     100.000000
    mid        qwen2.5:1.5b-instruct-q6_k           30.0090       0.833333
    mid      qwen2.5:1.5b-instruct-q4_K_M           38.3445       3.333333
   high        qwen2.5:1.5b-instruct-q8_0           39.2930       0.000000
```

## Inference System Metrics
```text
profile                               model  utilization_mean  memory_mean_mb  inference_proc_cpu_mean metric_basis
   high granite3.1-dense:8b-instruct-q4_K_M         24.756070     5755.058824               389.049538      cpu/ram
   high   granite3.1-dense:8b-instruct-q8_0         24.955434     9378.965428               391.311767      cpu/ram
   high         llama3.1:8b-instruct-q4_K_M         24.601428     5602.157405               386.460468      cpu/ram
   high           llama3.1:8b-instruct-q8_0         24.456344     9177.112784               383.420182      cpu/ram
   high        qwen2.5:1.5b-instruct-q4_K_M         22.551910     1433.854143               352.882774      cpu/ram
   high          qwen2.5:1.5b-instruct-q8_0         22.238542     2165.755492               347.936571      cpu/ram
    low granite3.1-dense:8b-instruct-q4_K_M         97.257633     5714.034540               387.885347      cpu/ram
    low         llama3.1:8b-instruct-q4_K_M         96.236615     5555.110552               383.485018      cpu/ram
    low        qwen2.5:1.5b-instruct-q4_K_M         87.888544     1388.010955               350.007622      cpu/ram
    mid granite3.1-dense:8b-instruct-q4_K_M         96.277133     5833.074395               767.834798      cpu/ram
    mid   granite3.1-dense:8b-instruct-q6_K          7.755417      690.058854                50.103333      cpu/ram
    mid         llama3.1:8b-instruct-q4_K_M         94.666182     5707.897902               754.885601      cpu/ram
    mid           llama3.1:8b-instruct-q6_K          5.782917      673.091738                40.608333      cpu/ram
    mid        qwen2.5:1.5b-instruct-q4_K_M         83.948911     1514.668500               667.777438      cpu/ram
    mid          qwen2.5:1.5b-instruct-q6_k         82.943378     2193.473163               660.389212      cpu/ram
```

## Normalized Scores (latest scores file)
```text
profile                               model  token_score_norm  latency_score  retrieval_score  answer_score
   high granite3.1-dense:8b-instruct-q4_K_M          0.001591       0.463798         0.360917      0.955252
   high   granite3.1-dense:8b-instruct-q8_0          0.480092       0.336832         0.360917      0.966458
   high         llama3.1:8b-instruct-q4_K_M          0.004095       0.600888         0.360917      0.917396
   high           llama3.1:8b-instruct-q8_0          0.258459       0.538955         0.360917      0.891843
   high        qwen2.5:1.5b-instruct-q4_K_M          0.020410       0.829439         0.360917      0.812723
   high          qwen2.5:1.5b-instruct-q8_0          0.600522       0.889431         0.360917      0.797604
    low granite3.1-dense:8b-instruct-q4_K_M          0.002493       0.467579         0.360917      0.952648
    low         llama3.1:8b-instruct-q4_K_M          0.001634       0.589527         0.360917      0.923438
    low        qwen2.5:1.5b-instruct-q4_K_M          0.002714       0.846444         0.360917      0.789916
    mid granite3.1-dense:8b-instruct-q4_K_M          0.027610       0.634074         0.360917      0.956356
    mid   granite3.1-dense:8b-instruct-q6_K               NaN       0.999917         0.360917           NaN
    mid         llama3.1:8b-instruct-q4_K_M          0.033466       0.728211         0.360917      0.929555
    mid           llama3.1:8b-instruct-q6_K               NaN       0.998322         0.360917           NaN
    mid        qwen2.5:1.5b-instruct-q4_K_M          0.082935       0.856726         0.360917      0.800108
    mid          qwen2.5:1.5b-instruct-q6_k          0.557592       0.908447         0.360917      0.795513
```

## Files
- `overview_global.csv`
- `overview_by_profile_model.csv`
- `overview_by_testcase.csv`
- `overview_failures.csv`
- `overview_scores_latest.csv`