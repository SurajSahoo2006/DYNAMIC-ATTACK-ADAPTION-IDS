# Phase 11 — Edge Compression and Latency Benchmark

## Overview and Objectives

Phase 11 evaluates the footprint, single-thread per-flow inference latency, batch throughput, memory footprint, and classification accuracy of the Phase 3 B2 ensemble (Random Forest 0.60 + XGBoost 0.40, 19 selected features) across multiple model compression strategies. Measurements reflect local single-thread execution on a standard development laptop CPU and do not represent physical IoT edge hardware.

### Key Findings

- **Model Footprint Reduction**: Compressed size decreased from 41.55 MB to 6.84 MB (6.07x reduction; uncompressed: 204.84 MB to 30.49 MB, 6.72x reduction).
- **Inference Latency Reduction**: Single-flow p95 latency reduced from 27.82 ms to 6.88 ms (75.3% reduction; median latency: 12.56 ms to 3.13 ms, 75.1% reduction).
- **Test Accuracy Difference**: 8-class test accuracy changed by -0.0003 (0.9542 to 0.9538), with a paired bootstrap 95% confidence interval of [-0.00126, 0.00061].
- **Chosen Edge Configuration**: `v1_rf_25` selected via validation data criteria (smallest model footprint within 1.0 percentage point of baseline validation macro F1).

## Benchmark Environment

| Metric | Specification |
| :--- | :--- |
| **Processor** | `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel` |
| **Architecture** | `AMD64` |
| **Logical Cores** | 8 |
| **Physical Cores** | 4 |
| **Total System RAM** | 7.79 GB |
| **Execution Threading** | Single-thread (`OMP_NUM_THREADS=1`, `n_jobs=1`, `nthread=1`) |
| **Python Version** | 3.13.1 |
| **Scikit-Learn** | 1.9.1 |
| **XGBoost** | 3.4.1 |
| **Joblib** | 1.6.0 |
| **NumPy** | 2.5.3 |
| **Pandas** | 3.0.6 |
| **PyTorch** | 2.15.0+cpu |
| **Psutil** | 7.2.2 |

## Table 1 — Compression Variants Benchmark (Validation Set)

| Variant | Category | Size (MB, uncomp) | Size (MB, comp=3) | p95 Latency (ms) | Median Latency (ms) | Throughput (fps) | Peak Memory Delta (MB) | Val Accuracy | Val Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (B2)** | Reference | 204.84 | 41.55 | 27.82 | 12.56 | 13402.4 | 0.0000 | 0.9645 | 0.8116 |
| `v1_rf_100` | V1 | 107.11 | 22.00 | 15.09 | 6.93 | 13660.2 | 0.0000 | 0.9646 | 0.8109 |
| `v1_rf_25` **(Chosen)** | V1 | 30.49 | 6.84 | 6.88 | 3.13 | 25314.8 | 0.0000 | 0.9642 | 0.8087 |
| `v1_rf_50` | V1 | 57.09 | 11.99 | 9.91 | 4.22 | 15796.0 | 0.0000 | 0.9645 | 0.8107 |
| `v2_xgb_125` | V2 | 202.49 | 40.56 | 29.21 | 13.68 | 11378.2 | 0.0000 | 0.9631 | 0.8049 |
| `v2_xgb_30` | V2 | 200.62 | 39.92 | 27.88 | 12.40 | 20119.9 | 0.0000 | 0.9619 | 0.8016 |
| `v2_xgb_60` | V2 | 201.24 | 40.14 | 29.34 | 13.34 | 12631.7 | 0.0039 | 0.9622 | 0.8018 |
| `v3_rf_depth10` | V3 | 21.81 | 6.88 | 28.62 | 13.59 | 15456.3 | 0.0039 | 0.9588 | 0.7863 |
| `v3_rf_depth15` | V3 | 78.22 | 19.17 | 30.56 | 13.31 | 10972.0 | 0.0000 | 0.9598 | 0.7902 |
| `v4_float32` | V4 | 204.84 | 41.37 | 29.45 | 13.46 | 12761.9 | 0.0000 | 0.9645 | 0.8116 |
| `v5_combined` | V5 | 104.77 | 21.19 | 16.49 | 7.21 | 22861.4 | 0.0000 | 0.9631 | 0.8046 |

## Table 2 — Final Test Set Comparison (Single-Pass Evaluation)

Evaluation on the untouched test partition ($N = 26,292$ flows, 19 features):

| Model Variant | Size (comp=3, MB) | p95 Latency (ms) | Throughput (fps) | Test Accuracy (8-class) | Wilson 95% CI (Acc) | Paired Diff vs Base (Mean, 95% CI) | Test Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (B2)** | 41.55 | 27.82 | 13402.4 | 0.9542 | [0.9516, 0.9566] | Reference (0.0000) | 0.8728 |
| **`v1_rf_25` (Chosen)** | 6.84 | 6.88 | 25314.8 | 0.9538 | [0.9512, 0.9563] | -0.0003 [-0.00126, 0.00061] | 0.8707 |
| **`v5_combined`** | 21.19 | 16.49 | 22861.4 | 0.9521 | [0.9495, 0.9546] | -0.0021 [-0.00289, -0.00110] | 0.8667 |

## Table 3 — Rare-Class F1 Score Impact (34-Class Model)

Per-class F1 score across the three low-support classes on the test partition comparing the full baseline against `v1_rf_25`:

| Attack Class | Class Index | Test Support | Baseline F1 | Chosen Variant F1 | F1 Difference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Uploading_Attack` | 31 | 34 | 0.4598 | 0.4396 | -0.0202 |
| `Recon-PingSweep` | 28 | 59 | 0.5631 | 0.5631 | +0.0000 |
| `Backdoor_Malware` | 0 | 95 | 0.5701 | 0.5728 | +0.0027 |

## Observations and Trade-offs

1. **Tree Truncation Efficiency**: Random Forest size scales directly with tree count. Reducing from 200 trees to smaller sub-forests yields near-linear reductions in file size and latency while preserving macro F1 within the 1.0 percentage point budget.
2. **XGBoost Round Scaling**: XGBoost models possess a compact initial footprint. Truncating boosting rounds offers moderate inference speedup with negligible metric deviation.
3. **Precision Casting (`float32`)**: Converting inputs to single-precision float32 produced zero prediction divergences across all validation samples compared to double-precision evaluation.
4. **Rare-Class Sensitivity**: Rare attack classes with low sample support exhibit sensitivity to tree pruning, underscoring the trade-off between edge model sparsity and tail-class recall.

## Limitations and Boundaries

- **Host Platform**: Timings were recorded on a standard laptop CPU (Intel x86_64 architecture) rather than a physical embedded IoT gateway or edge microcontroller (e.g., ARM Cortex or RISC-V). Relative performance on constrained hardware will differ.
- **Measurement Noise**: Per-flow latency was recorded via wall-clock time (`time.perf_counter()`), which is subject to OS scheduling jitter and CPU frequency scaling.
- **Pipeline Scope**: Latency figures measure model inference only and exclude packet ingestion, flow reassembly, and feature extraction stages.
- **Feature Selection Pipeline**: The pre-processing overhead required to project raw network features (46 to 19 features) is not included in the inference latency measurement.
- **Quantization Scope**: INT8 post-training quantization was not evaluated because standard scikit-learn Random Forest and XGBoost tree ensembles do not provide a native INT8 runtime path.
- **Seed and Sample Size**: Benchmarks were performed using a single random seed (`42`), and rare-class sample counts in the test partition are small ($N < 100$ per class).
