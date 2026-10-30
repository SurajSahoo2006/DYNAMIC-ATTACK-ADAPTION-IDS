# Phase 13 Summary: Consolidated IDS Pipeline Integration

## 1. System Architecture

The consolidated `IDSPipeline` unifies all telemetry processing, inference, calibration,
anomaly detection, explanation, response, and logging components developed across Phases 1 through 12.
Execution is single-threaded CPU (`torch.set_num_threads(1)`), lazy-loaded, with zero duplicated configuration.

| Stage # | Subsystem | Supplying Phase | Primary Component / Model | Operational Function |
|---|---|---|---|---|
| 1 | Feature Scaling | Phase 1 & 2 | `StandardScaler` (19 slice) | Standardizes raw physical telemetry to float32 |
| 2 | B2 Ensemble Classifier | Phase 3 & 11 | RF (0.60) + XGBoost (0.40) | Computes raw 8-class probabilities |
| 3 | Probability Calibration | Phase 12 | One-vs-Rest Isotonic | Computes calibrated class probabilities and confidence |
| 4 | Anomaly Scoring | Phase 10 | Benign PyTorch Autoencoder | Reconstruction MSE score against 97.5th percentile cutoff |
| 5 | Fusion Decision Rule | Phase 10 & 12 | Decision Engine | Predicts known class, flags UNKNOWN, assigns severity |
| 6 | Explainability & Briefing | Phase 3 & 4 | XGBoost TreeExplainer + LLM | Top-5 signed SHAP feature attributions + incident brief |
| 7 | Tactical Defense | Phase 5 | MAPPO Team (3 Agents) | Greedy balanced Triage, Containment, and Escalation |
| 8 | Forensic Alert Logging | Phase 4 & 5 | `ForensicAlertLogger` | Cryptographic SHA-256 hash chaining + tail verification |

## 2. Decision Fusion Rule Specification

The decision logic executes a deterministic multi-stage fusion rule:

```text
1. predicted_class = argmax(P_raw)
2. ae_flag = (ae_score > 0.534869)               # Phase 10 97.5th percentile threshold
3. low_conf_flag = (calibrated_confidence < 0.521507) # Phase 12 10.0% validation Benign FPR cutoff
4. unknown_flag = ae_flag OR low_conf_flag
5. severity = get_flow_severity(predicted_class, raw_confidence)
```

The predicted class retains the raw ensemble argmax to preserve continuity with the Phase 3-6
evaluations and MARL cache observations. The UNKNOWN flag designates flows either displaying high
autoencoder reconstruction deviation or failing the calibrated confidence threshold.

## 3. Subsystem Parity Verification (500 Test Rows)

Parity tests compare unified pipeline outputs against the original standalone phase code on 500 held-out rows.

| Test ID | Subsystem Property | Expected Source Output | Metric / Difference | Tolerance | Status |
|---|---|---|---|---|---|
| P1 | Pipeline Raw Probabilities == predict.py Output | Independent Phase Implementation | Max Diff: 0.00e+00 | 1.00e-06 | PASSED |
| P2 | Pipeline Calibrated Confidence == Calibration Code Output | Independent Phase Implementation | Max Diff: 0.00e+00 | 1.00e-06 | PASSED |
| P3 | Pipeline AE Score == Phase 10 AE Scoring Output | Independent Phase Implementation | Max Diff: 0.00e+00 | 1.00e-06 | PASSED |
| P4 | Severity == Phase 3 Deterministic Severity Table | Independent Phase Implementation | 500 of 500 exact matches | Exact match | PASSED |
| P5 | Pipeline SHAP Top-5 Feature Names == xai/shap_explain.py Output (20 rows) | Independent Phase Implementation | 0 mismatches on 20 rows | 0 mismatches | PASSED |
| P6 | Forensic Alert Logger Pass on Chain & Rejection on Tampered Copy | Independent Phase Implementation | Clean log PASS; Tampered copy FAIL | Boolean match | PASSED |
| P7 | analyze_batch Rejects Missing Feature and Names Missing Column | Independent Phase Implementation | Rejected missing column 'flow_duration' | ValueError raised | PASSED |

## 4. End-to-End Evaluation on 2,000 Stratified Test Flows

A stratified sample of 2,000 flows from `data/processed/test.parquet` was evaluated through the pipeline.

### Classification Metrics & Baseline Comparison

| Metric | Measured Pipeline Value | 95% Wilson Confidence Interval | Phase 11 B2 Baseline | Difference |
|---|---|---|---|---|
| 8-Class Accuracy | 95.50% | [94.50%, 96.32%] | 95.42% | +0.08% |
| 8-Class Macro F1 | 0.8701 | N/A | 0.8728 | -0.0027 |

*Note on difference*: Evaluated on 2,000 stratified samples (test.parquet) versus full 26,292 test split in Phase 11.

### System Runtime Performance

| Performance Characteristic | Measured Value | Unit | Operating Conditions |
|---|---|---|---|
| Inference Throughput | 4631.6 | flows / sec | Single process, batch mode |
| Median Latency (p50) | 107.439 | ms / flow | Single thread sequential |
| 95th Percentile Latency (p95) | 124.594 | ms / flow | Single thread sequential |
| Memory RSS Before Inference | 367.96 | MB | Loaded system baseline |
| Peak Process RSS | 724.86 | MB | Subprocess peak memory |

## 5. UNKNOWN Flagging Rate by Attack Class

The table below reports the proportion of test flows flagged as UNKNOWN under the fusion decision rule,
alongside 95% Wilson confidence intervals and the Phase 12 known-class selective prediction control.

| Ground Truth Class | Evaluated Flows | Flagged UNKNOWN | UNKNOWN Rate (%) | 95% Wilson CI | Phase 12 Control (%) | Rate Difference (%) |
|---|---|---|---|---|---|---|
| Benign | 76 | 10 | 13.16% | [7.31%, 22.55%] | 12.20% | +0.96% |
| BruteForce | 27 | 5 | 18.52% | [8.18%, 36.70%] | 22.07% | -3.55% |
| DDoS | 863 | 391 | 45.31% | [42.01%, 48.64%] | 0.02% | +45.29% |
| DoS | 302 | 104 | 34.44% | [29.30%, 39.96%] | 0.10% | +34.34% |
| Mirai | 225 | 162 | 72.00% | [65.80%, 77.46%] | 0.00% | +72.00% |
| Recon | 304 | 77 | 25.33% | [20.77%, 30.50%] | 7.35% | +17.98% |
| Spoofing | 152 | 43 | 28.29% | [21.73%, 35.92%] | 8.59% | +19.70% |
| Web-based | 51 | 10 | 19.61% | [11.02%, 32.46%] | 23.81% | -4.20% |

## 6. Multi-Agent Tactical Defense & Forensic Verification

- **MARL Tactical Response (500 flows)**: Executed greedy balanced MAPPO team policy.
  - Triage Action Distribution: `{"accept_verdict": 1.0}`
  - Containment Action Distribution: `{"isolate": 0.224, "none": 0.722, "rate_limit": 0.054}`
  - Escalation Action Distribution: `{"notify_analyst": 0.144, "none": 0.856}`
- **Forensic Alert Log Verification (200 flows)**: 200 lines verified
  with verification status `PASS`.
  External tail hash anchor: `21e852c4e852fc56bdc753c0f734bca68084f4772fd444c282b8d163583b9594`.

## 7. Limitations

1. **Simulated Telemetry**: The pipeline operates offline on recorded network flows from the CICIoT2023 dataset rather than real-time hardware packet captures.
2. **Simulated Response Dynamics**: The MARL tactical response engine executes an offline simulation on sequential flows; actions do not alter live host firewall rules or network interface hardware.
3. **Asymmetric Decision Pathways**: The predicted attack class is determined by the argmax of raw B2 ensemble probabilities (preserving Phase 3-6 alignment), whereas the UNKNOWN decision incorporates calibrated confidence and autoencoder reconstruction error.
4. **Benign False-Positive Rate**: The fusion rule exhibits a Benign false-positive rate of 13.16% (95% Wilson CI: [7.31%, 22.55%]), reflecting the trade-off required for zero-day attack identification.
5. **Model-Generated Language Outputs**: Incident summaries and remediation recommendations generated by the local LLM require human analyst oversight before implementing critical network isolation actions.
6. **Model Weight Storage**: Trained neural network checkpoints and ensemble weights are generated locally by phase training scripts and are excluded from git version control.
