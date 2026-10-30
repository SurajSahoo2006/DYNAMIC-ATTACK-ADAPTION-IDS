# Phase 11 — Edge Compression and Latency Benchmark

## Overview and Objectives

Phase 11 evaluates the footprint, single-thread per-flow inference latency, batch throughput, true resident memory (psutil RSS), and classification accuracy of the Phase 3 B2 ensemble (Random Forest 0.60 + XGBoost 0.40, 19 selected features) across multiple model compression strategies. Measurements reflect local single-thread execution on a standard development laptop CPU and do not represent physical IoT edge hardware.

### Key Findings

- **Model Footprint Reduction**: Compressed size decreased from 41.47 MB to 6.70 MB (6.19x reduction; uncompressed: 204.84 MB to 30.49 MB, 6.72x reduction).
- **Inference Latency Reduction**: Single-flow p95 latency reduced from 37.76 ms to 7.70 ms (79.6% reduction; median latency: 32.59 ms to 6.61 ms, 79.7% reduction).
- **Resident Memory**: Model loading RSS was 397.32 MB for baseline vs 211.47 MB for `v1_rf_25`; peak batch RSS was 397.32 MB vs 215.32 MB.
- **Test Accuracy Difference**: 8-class test accuracy changed by -0.0003 (0.9542 to 0.9538), with a paired bootstrap 95% confidence interval of [-0.00126, 0.00061].
- **Chosen Edge Configuration**: `v1_rf_25` selected via validation data criteria (smallest model footprint within 1.0 percentage point of baseline validation macro F1).

## Baseline Reconciliation

To ensure consistency across development phases, baseline metrics are reconciled against historical milestones recorded in `artifacts/model_manifest.json` and `federated/results/fedavg_results.json`:

| Milestone / Experiment | Architecture / Model | Test File | Test Rows | 8-Class Test Accuracy | Macro F1 | Status / Reconciliation Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 3 Manifest** | `Phase 3 Weighted Ensemble (rf_8class_baseline 0.20 + xgb_8class_baseline 0.80)` | `data/processed/test.parquet` | 26,292 | 0.9553 | 0.8766 | Reproduced exactly (0.9553 acc, diff: 0.0000) using unchanged Phase 3 pipeline |
| **Phase 9 Centralized** | `Centralized Random Forest Only (rf_8class_b2.joblib, 200 trees, no XGBoost)` | `data/processed/test.parquet` | 26,292 | 0.9523 | 0.8697 | Evaluates single centralized RF model without XGBoost component |
| **Phase 11 Baseline (B2)** | `Phase 3 B2 Ensemble (rf_8class_b2.joblib 0.60 + xgb_8class_b2.joblib 0.40)` | `data/processed/test.parquet` | 26,292 | 0.9542 | 0.8728 | Diff vs Phase 3 manifest: -0.0011; verified cause: different model version (B2 unsmoted balanced weights) and ensemble weighting (0.60/0.40 vs 0.20/0.80) |

## Benchmark Environment

| Metric | Specification |
| :--- | :--- |
| **Processor** | `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel` |
| **Architecture** | `AMD64` |
| **Logical Cores** | 8 |
| **Physical Cores** | 4 |
| **Total System RAM** | 7.79 GB |
| **Execution Threading** | Single-thread (`OMP_NUM_THREADS=1`, `n_jobs=1`, `nthread=1`) |
| **Process Isolation** | Fresh subprocess per variant for resident memory isolation |
| **Python Version** | 3.13.1 |
| **Scikit-Learn** | 1.9.1 |
| **XGBoost** | 3.4.1 |
| **Joblib** | 1.6.0 |
| **NumPy** | 2.5.3 |
| **Pandas** | 3.0.6 |
| **PyTorch** | 2.15.0+cpu |
| **Psutil** | 7.2.2 |

## Table 1 — Compression Variants Benchmark (Validation Set)

| Variant | Category | Size (MB, uncomp) | Size (MB, comp=3) | p95 Latency (ms) | Median Latency (ms) | Throughput (fps) | Load RSS (MB) | Peak RSS (MB) | Val Accuracy | Val Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (B2)** | Reference | 204.84 | 41.47 | 37.76 | 32.59 | 5328.5 | 397.32 | 397.32 | 0.9645 | 0.8116 |
| `v1_rf_100` | V1 | 107.11 | 22.05 | 21.14 | 17.86 | 7181.9 | 294.32 | 294.32 | 0.9646 | 0.8109 |
| `v1_rf_25` **(Chosen)** | V1 | 30.49 | 6.70 | 7.70 | 6.61 | 9667.7 | 211.47 | 215.32 | 0.9642 | 0.8087 |
| `v1_rf_50` | V1 | 57.09 | 11.98 | 12.07 | 10.35 | 8732.2 | 236.91 | 239.67 | 0.9645 | 0.8107 |
| `v2_xgb_125` | V2 | 202.49 | 40.64 | 34.23 | 32.57 | 6699.4 | 388.93 | 388.93 | 0.9631 | 0.8049 |
| `v2_xgb_30` | V2 | 200.62 | 39.93 | 33.62 | 31.99 | 8473.1 | 384.62 | 387.89 | 0.9619 | 0.8016 |
| `v2_xgb_60` | V2 | 201.24 | 40.20 | 37.19 | 32.53 | 7840.9 | 384.81 | 395.17 | 0.9622 | 0.8018 |
| `v3_rf_depth10` | V3 | 21.81 | 6.92 | 34.17 | 32.94 | 6120.9 | 202.91 | 204.65 | 0.9588 | 0.7863 |
| `v3_rf_depth15` | V3 | 78.23 | 19.18 | 34.44 | 32.90 | 5544.4 | 257.35 | 269.01 | 0.9598 | 0.7902 |
| `v4_float32` | V4 | 204.84 | 41.48 | 36.39 | 33.14 | 5119.6 | 398.10 | 398.10 | 0.9645 | 0.8116 |
| `v5_combined` | V5 | 104.77 | 21.19 | 19.07 | 17.65 | 10315.6 | 287.83 | 296.71 | 0.9631 | 0.8046 |

## Table 2 — Final Test Set Comparison (Single-Pass Evaluation)

Evaluation on the untouched test partition ($N = 26,292$ flows, 19 features):

| Model Variant | Size (comp=3, MB) | p95 Latency (ms) | Throughput (fps) | Load RSS (MB) | Peak RSS (MB) | Test Accuracy (8-class) | Wilson 95% CI (Acc) | Paired Diff vs Base (Mean, 95% CI) | Test Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (B2)** | 41.47 | 37.76 | 5328.5 | 397.32 | 397.32 | 0.9542 | [0.9516, 0.9566] | Reference (0.0000) | 0.8728 |
| **`v1_rf_25` (Chosen)** | 6.70 | 7.70 | 9667.7 | 211.47 | 215.32 | 0.9538 | [0.9512, 0.9563] | -0.0003 [-0.00126, 0.00061] | 0.8707 |
| **`v5_combined`** | 21.19 | 19.07 | 10315.6 | 287.83 | 296.71 | 0.9521 | [0.9495, 0.9546] | -0.0021 [-0.00289, -0.00110] | 0.8667 |

## Table 3 — Rare-Class F1 Score Impact (34-Class Model)

Per-class F1 score across the three low-support classes on the test partition comparing the full baseline against `v1_rf_25`:

| Attack Class | Class Index | Test Support | Baseline F1 | Chosen Variant F1 | F1 Difference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Uploading_Attack` | 31 | 34 | 0.4598 | 0.4396 | -0.0202 |
| `Recon-PingSweep` | 28 | 59 | 0.5631 | 0.5631 | +0.0000 |
| `Backdoor_Malware` | 0 | 95 | 0.5701 | 0.5728 | +0.0027 |

## Observations and Trade-offs

1. **Tree Truncation Proportionality**: Random Forest tree truncation reduces model size and inference latency roughly in proportion to tree count: from 200 trees (baseline: 204.84 MB uncompressed, 41.47 MB compressed, 37.76 ms p95 latency) to 100 trees (107.11 MB uncompressed, 22.05 MB compressed, 21.14 ms latency), 50 trees (57.09 MB, 11.98 MB, 12.07 ms), and 25 trees (30.49 MB, 6.70 MB, 7.70 ms), while validation macro F1 changes minimally from 0.8116 to 0.8087.
2. **RF Depth Capping Trade-off**: RF depth capping reduces file size (max_depth=15: 78.23 MB uncompressed; max_depth=10: 21.81 MB vs baseline 204.84 MB) but did not reduce latency (p95 latency was 34.44 ms and 34.17 ms vs baseline 37.76 ms) while reducing validation macro F1 to 0.7902 and 0.7863.
3. **XGBoost Booster Truncation**: Reloaded truncated XGBoost models reduced the standalone XGBoost component file size from 4.86 MB uncompressed to 2.51 MB (125 rounds), 1.26 MB (60 rounds), and 0.64 MB (30 rounds), with validation macro F1 remaining at 0.8049, 0.8018, and 0.8016.
4. **Input Precision Casting**: Single-precision float32 input casting resulted in 0 prediction differences out of 106,766 validation flows relative to double-precision evaluation.
5. **Rare-Class Sensitivity**: Rare attack classes exhibited F1 changes of -0.0202, +0.0000, +0.0027 on supports of 34, 59, 95; the differences are within sampling noise and show no consistent direction.

## Limitations and Boundaries

- **Host Platform**: Timings were recorded on a standard laptop CPU (Intel x86_64 architecture) rather than a physical embedded IoT gateway or edge microcontroller (e.g., ARM Cortex or RISC-V). Relative performance on constrained hardware will differ.
- **Measurement Noise**: Per-flow latency was recorded via wall-clock time (`time.perf_counter()`), which is subject to OS scheduling jitter and CPU frequency scaling.
- **Pipeline Scope**: Latency figures measure model inference only and exclude packet ingestion, flow reassembly, and feature extraction stages.
- **Feature Selection Pipeline**: The pre-processing overhead required to project raw network features (46 to 19 features) is not included in the inference latency measurement.
- **Quantization Scope**: INT8 post-training quantization was not evaluated because standard scikit-learn Random Forest and XGBoost tree ensembles do not provide a native INT8 runtime path.
- **Seed and Sample Size**: Benchmarks were performed using a single random seed (`42`), and rare-class sample counts in the test partition are small ($N < 100$ per class).
