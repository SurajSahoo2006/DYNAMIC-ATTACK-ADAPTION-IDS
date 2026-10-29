# Phase 4 - Explainable AI (SHAP), Adversarial Robustness & Forensic Alert Logging

> **Generated**: 2026-10-29  
> **Repository Branch**: `phase4-xai-adversarial`  
> **Primary Classifier**: Method B2 Ensemble (Random Forest 0.60 + XGBoost 0.40 + PyTorch DNN 0.00)  
> **Target Dataset**: CICIoT2023 (19 RFE-selected features)  
> **Hardware Environment**: Local Windows Laptop, 8 GB RAM, CPU-only (No GPU)  
> **Measured Peak RAM**: **987.3 MB (0.96 GB)** during SHAP TreeExplainer; **709.3 MB (0.69 GB)** during Adversarial Suite. *(Strict hardware ceiling: < 6.0 GB)*

---

## 1. Executive Summary & Flowchart Alignment

Phase 4 completes three key components of the intrusion detection pipeline:
1. **Explainable AI (SHAP TreeExplainer)**:
   - Provides mathematically grounded Shapley feature attributions for tree components of the B2 ensemble.
   - Generates global feature importance rankings, per-class top feature distributions, and local sample explanations.
   - Verifies architectural feature stability between Random Forest (weight 0.60) and XGBoost (weight 0.40).
   - Integrates with the LLM Semantic Orchestrator via an opt-in `--use-shap` flag.
2. **Adversarial Robustness Assessment (Adversarial Robustness Toolbox - ART)**:
   - Implements **Zero-Attack Pipeline Invariance Control ($\epsilon = 0$)**: proves bit-for-bit mathematical identity ($0.000000$ drift).
   - Implements **Random-Noise Baseline Control**: compares isotropic noise of identical $L_2$ magnitude to evaluate whether attacks beat random perturbations.
   - Corrects **PyTorch DNN Surrogate**: properly loads trained weights, establishing a **77.60%** clean baseline (76.80% on full test set) for gradient calculation.
   - Evaluates three attack families: **FGSM transfer** (gradient-based), **HopSkipJump** (decision-based black-box across 2 budgets), and **ZOO** (score-based black-box across 2 budgets).
   - Executes a **4-Arm Defense Comparison**: evaluates Arm A (Original B2), Arm B (Retrained Clean 1x), Arm C (Retrained Clean 2x volume control), and Arm D (Retrained Clean + Adv) across Clean, FGSM, Random Noise, and HopSkipJump conditions.
3. **Cryptographic Tamper-Evident Forensic Alert Logging**:
   - Persists detection records to `logs/alerts.jsonl` in a flat, ELK/Kibana-ready schema.
   - Eliminates vague "risk" labels by storing TreeExplainer `shap_value` and physical `raw_value` in distinct, explicitly typed fields.
   - Implements cryptographic SHA-256 hash chaining (`prev_hash` -> `entry_hash`) guaranteeing tamper evidence.
   - Validates log integrity and proves tamper detection through automated audit tests.

---

## 2. Part A: SHAP Explainability Evaluation

### Architectural Note on Ensemble Explainability
The B2 Ensemble is a weighted soft-voting mixture:
$$P_{\text{ens}}(y = c \mid x) = 0.60 \cdot P_{\text{RF}}(y = c \mid x) + 0.40 \cdot P_{\text{XGB}}(y = c \mid x) + 0.00 \cdot P_{\text{DNN}}(y = c \mid x)$$
Because `shap.TreeExplainer` operates in the log-odds margin space of individual decision trees, a convex mixture of disparate probability estimators cannot be represented as a single tree. Therefore:
- **Global & Local Explanations**: Computed on **XGBoost** (weight = 0.40), which evaluates in milliseconds per sample.
- **Stability Cross-Check**: Evaluated against **Random Forest** (weight = 0.60) on a stratified sample to verify whether both tree models rely on identical physical features.

### Empirical Execution Telemetry
- **Sample Size**: 1,998 stratified test flows across all 8 classes (`seed=42`).
- **XGBoost SHAP Runtime**: **5.59 s** (2.80 ms/flow).
- **Random Forest Stability Check Runtime**: 87.27 s on 200 stratified flows.
- **Peak Process RAM**: **987.3 MB (0.96 GB)**.

### Global Feature Importance (Top 10 Features)

| Rank | Feature Name | Mean \|SHAP Value\| | Physical Interpretation |
|:---:|:---|:---:|:---|
| 1 | `IAT` | **2.6829** | Inter-Arrival Time between packets (dominant timing metric) |
| 2 | `Protocol Type` | **0.4732** | Transport protocol code (TCP vs UDP vs ICMP) |
| 3 | `Header_Length` | **0.4397** | Accumulated packet header byte length |
| 4 | `rst_count` | **0.4213** | Rate of connection reset flags |
| 5 | `urg_count` | **0.3675** | Rate of urgent pointer flags |
| 6 | `flow_duration` | **0.3237** | Total elapsed duration of the flow window |
| 7 | `Min` | **0.3034** | Minimum packet size observed |
| 8 | `Variance` | **0.2932** | Packet size variance ratio |
| 9 | `syn_count` | **0.2765** | Rate of connection initiation packets |
| 10 | `Tot size` | **0.2344** | Total transmitted packet byte volume |

> **Artifact**: Plot saved to `reports/shap_global_importance.png`.

### Per-Class Top-5 Features (8 Attack Families)

| Attack Family | Top Feature 1 | Top Feature 2 | Top Feature 3 | Top Feature 4 | Top Feature 5 |
|:---|:---|:---|:---|:---|:---|
| **`Benign`** | `IAT` (2.90) | `rst_count` (0.74) | `urg_count` (0.67) | `Tot size` (0.42) | `Header_Length` (0.41) |
| **`BruteForce`** | `IAT` (2.56) | `Header_Length` (0.53) | `syn_count` (0.53) | `urg_count` (0.44) | `flow_duration` (0.44) |
| **`DDoS`** | `IAT` (4.73) | `Variance` (0.53) | `flow_duration` (0.39) | `urg_count` (0.38) | `Tot sum` (0.37) |
| **`DoS`** | `IAT` (4.37) | `urg_count` (0.72) | `Magnitue` (0.56) | `flow_duration` (0.54) | `Variance` (0.49) |
| **`Mirai`** | `Protocol Type` (2.51) | `IAT` (1.53) | `Min` (1.45) | `rst_count` (0.69) | `Variance` (0.46) |
| **`Recon`** | `IAT` (1.03) | `syn_count` (0.69) | `rst_count` (0.51) | `Header_Length` (0.38) | `flow_duration` (0.25) |
| **`Spoofing`** | `IAT` (2.73) | `Header_Length` (0.97) | `rst_count` (0.24) | `AVG` (0.20) | `flow_duration` (0.16) |
| **`Web-based`** | `IAT` (1.61) | `Header_Length` (0.57) | `Tot size` (0.31) | `flow_duration` (0.31) | `urg_count` (0.26) |

> **Artifact**: Heatmap saved to `reports/shap_per_class_importance.png`.

### Stability Cross-Check (XGBoost vs. Random Forest)
Evaluated on a shared stratified test split (200 flows):
- **Spearman Rank Correlation**: **$\rho = 0.9088$** ($p = 0.000$)
- **Top-5 XGBoost Features**: `['IAT', 'Protocol Type', 'Header_Length', 'rst_count', 'urg_count']`
- **Top-5 Random Forest Features**: `['IAT', 'rst_count', 'Variance', 'Header_Length', 'urg_count']`
- **Top-5 Feature Overlap**: **4 / 5 (80.0%)** (`IAT`, `rst_count`, `Header_Length`, `urg_count`)
- **Conclusion**: Both models agree on the dominant physical telemetry dimensions, confirming high ensemble stability.

### LLM Telemetry Grounding & Feature Phrase Auditing
Evaluated via `xai/compare_telemetry_grounding.py` across all 40 demo flows for both Phase 3 (static telemetry prompt) and Phase 4 (dynamic SHAP top-5 telemetry prompt):
- **Ungrounded Numbers**: **0 / 40** in Phase 4; **1 / 40** in Phase 3 (Flow 23888 quoted confidence percentage verbatim).
- **Unprovided Feature Phrases**: **0 / 40** in both phases. Neither run hallucinated unauthorized features.
- **Generic Duration Fixation**: In Phase 3, the prompt always passed `flow_duration`, leading the LLM to invent generic justification phrases ("due to its low duration") in **11 / 40 (27.5%)** explanations. In Phase 4 with SHAP-driven telemetry and uncertainty guidance, generic duration claims dropped to **0 / 40 (0.0%)**, focusing explanations on real physical drivers and calibrated confidence bands.
- **Artifact**: Saved to `reports/telemetry_grounding_comparison.json`.

---

## 3. Part B: Adversarial Robustness Assessment

### Realistic Domain Constraints
Adversarial perturbations in network intrusion detection must respect network protocol invariants. Attackers cannot arbitrarily forge TCP control flags during an active session without breaking state.

| Feature Name | Feature Type | Empirical Range | Attacker Modifiable | Rationale / Defense Insight |
|:---|:---:|:---:|:---:|:---|
| `flow_duration` | Continuous | [0.00, 99435.77] | **YES** | Attacker can delay packet transmission or hold sockets open. |
| `Header_Length` | Continuous | [0.00, 9865998.00] | **YES** | Attacker can insert IP/TCP options or padding bytes. |
| `Protocol Type` | Protocol Code | [0.00, 47.00] | **NO (Fixed)** | Fixed by transport stack; forging invalidates packet delivery. |
| `fin_flag_number` | Binary Flag | [0.00, 1.00] | **NO (Fixed)** | Setting FIN prematurely closes connection state machine. |
| `syn_count` | Count | [0.00, 10.40] | **NO (Fixed)** | Rate counter; altering changes the attack identity itself. |
| `ack_count` | Count | [0.00, 5.70] | **NO (Fixed)** | Governed by TCP acknowledgment handshake mechanics. |
| `syn_flag_number` | Binary Flag | [0.00, 1.00] | **NO (Fixed)** | Connection initiation flag; defined by protocol handshake. |
| `urg_count` | Count | [0.00, 3600.50] | **NO (Fixed)** | Urgent pointer counter; restricted by OS TCP/IP stack. |
| `rst_count` | Count | [0.00, 9126.50] | **NO (Fixed)** | Reset flag abruptly terminates connection. |
| `ICMP` | Binary Flag | [0.00, 1.00] | **NO (Fixed)** | Protocol header flag; cannot be flipped for TCP/UDP flows. |
| `Tot size` | Continuous | [42.00, 5145.00] | **YES** | Attacker can append arbitrary padding bytes to payloads. |
| `IAT` | Continuous | [0.00, 167639424.00] | **YES** | Inter-Arrival Time is controlled via pacing and jitter. |
| `AVG` | Continuous | [42.00, 8490.74] | **YES** | Average packet size is shifted directly by dummy padding. |
| `Max` | Continuous | [42.00, 21786.00] | **YES** | Maximum packet size is shifted by sending padded packets. |
| `Min` | Continuous | [42.00, 4410.00] | **YES** | Minimum packet size increased by padding all smaller packets. |
| `Tot sum` | Continuous | [81.00, 61882.60] | **YES** | Total volume transmitted altered by dummy traffic. |
| `Magnitue` | Continuous | [9.17, 124.95] | **YES** | Geometric magnitude scales directly with packet padding. |
| `Covariance` | Continuous | [0.00, 43075336.00] | **YES** | Shaped by variable packet sizing and arrival bursts. |
| `Variance` | Continuous | [0.00, 1.00] | **YES** | Shaped by mixing uniform and variable-length frames. |

### Zero-Attack Pipeline Invariance Control ($\epsilon = 0$)
To guarantee that the constraint enforcement pipeline (`unscale -> clip -> round -> rescale`) does not introduce numerical artifacts or change predictions on unperturbed traffic, the pipeline was evaluated on original clean flows:
- **Maximum Numerical Drift ($|X_{\text{pipe}} - X_{\text{clean}}|$)**: **0.000000** (exact bit-for-bit identity across all coordinates).
- **PyTorch DNN Accuracy**: Raw clean = **77.60%** | Pipeline-passed = **77.60%** (0.00% difference).
- **B2 Ensemble Accuracy**: Raw clean = **95.60%** | Pipeline-passed = **95.60%** (0.00% difference).

### FGSM Evasion Transfer vs. Random-Noise Control (500 Test Flows)
White-box gradients were computed on the PyTorch DNN surrogate and evaluated for black-box evasion transfer against the B2 Tree Ensemble (RF + XGBoost). For each $\epsilon$, an isotropic Gaussian random noise baseline matching the exact sample-wise $L_2$ perturbation magnitude was applied under the exact same domain constraints:

| Epsilon ($\epsilon$) | Mean Perturbation ($L_2$) | PyTorch DNN Acc (White-Box) | B2 Ensemble Acc (FGSM Transfer) | Transfer Attack Success Rate (ASR) | Ensemble Acc (Random Noise Control) | Random Noise ASR | Attack vs Noise Advantage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\epsilon = 0.00$** | 0.000 | 77.60% | 95.60% | 0.00% | 95.60% | 0.00% | +0.00% |
| **$\epsilon = 0.01$** | 0.031 | 77.40% | **57.00%** | **41.00%** | 70.60% | 26.99% | **+14.01%** |
| **$\epsilon = 0.05$** | 0.152 | 76.00% | **53.20%** | **45.82%** | 61.00% | 37.24% | **+8.58%** |
| **$\epsilon = 0.10$** | 0.294 | 73.60% | **51.00%** | **47.91%** | 59.20% | 39.12% | **+8.79%** |
| **$\epsilon = 0.20$** | 0.569 | 62.40% | **47.80%** | **51.67%** | 56.60% | 42.05% | **+9.62%** |
| **$\epsilon = 0.30$** | 0.841 | 57.00% | **39.00%** | **60.88%** | 53.00% | 45.19% | **+15.69%** |

> **Scientific Finding**: FGSM transfer evasion clearly beats random noise across all perturbation levels by **+8.6% to +15.7% ASR**. Even small gradient-guided shifts ($\epsilon = 0.01$, mean $L_2 = 0.031$) drop tree ensemble accuracy by 38.6%, proving that gradient directions learned by the DNN surrogate expose common tabular decision vulnerabilities.

### HopSkipJump Black-Box Attack (Multi-Budget Comparison)
Evaluated directly on the B2 Tree Ensemble across decision boundaries under domain constraints:

| Query Budget Setting | Test Flows | Max Iterations | Max Evaluations | Clean Accuracy | Perturbed Accuracy | Attack Success Rate (ASR) | Mean $L_2$ | Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Budget 1 (Low)** | 20 | 3 | 10 | **100.00%** | **30.00%** | **70.00%** | 0.7166 | 18.39 s |
| **Budget 2 (High)** | 10 | 6 | 20 | **80.00%** | **10.00%** | **100.00%** | 0.7153 | 16.13 s |

### Zeroth-Order Optimization (ZOO) Attack (Multi-Budget Comparison)
Score-based coordinate descent attack against B2 Ensemble probabilities (`nb_parallel = 10`):

| Query Budget Setting | Test Flows | Max Iterations | Coordinate Batches | Clean Accuracy | Perturbed Accuracy | Attack Success Rate (ASR) | Mean $L_2$ | Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Budget 1 (Low)** | 10 | 4 | 10 | **90.00%** | **50.00%** | **44.44%** | 0.0979 | 18.37 s |
| **Budget 2 (High)** | 10 | 8 | 10 | **90.00%** | **30.00%** | **66.67%** | 0.1339 | 35.70 s |

> **Artifact**: Comparison plot saved to `reports/adversarial_accuracy_vs_epsilon.png`.

---

## 4. Defense Experiment: 4-Arm Multi-Condition Matrix

To rigorously evaluate whether adversarial retraining provides genuine robustness or merely reflects sample size variation, four model arms were evaluated on **the same fresh test split (400 flows)** under four distinct conditions:
- **Arm A**: Original B2 Ensemble (trained on full dataset in Phase 1/2).
- **Arm B**: Retrained Clean 1x (3,000 clean training flows).
- **Arm C**: Retrained Clean 2x (6,000 clean training flows, data volume control).
- **Arm D**: Retrained Clean + Adversarial (3,000 clean + 3,000 FGSM $\epsilon=0.10$ training flows).

### 4x4 Evaluation Matrix

| Model Arm | Clean Test Accuracy | FGSM Transfer ($\epsilon = 0.10$) | Random Noise Control ($L_2 \approx 0.29$) | HopSkipJump (Black-Box) |
|:---|:---:|:---:|:---:|:---:|
| **Arm A: Original B2 (Full Data)** | **96.73%** | 51.26% | 57.79% | 40.00% |
| **Arm B: Retrained Clean 1x (3k)** | 93.97% | 40.70% | 50.25% | 40.00% |
| **Arm C: Retrained Clean 2x (6k)** | 94.72% | 46.98% | 54.77% | 53.33% |
| **Arm D: Retrained Clean + Adv (3k+3k)** | 91.46% | **89.95%** | **65.08%** | **73.33%** |

### Defense Insights:
1. **Adversarial Hardening Under FGSM**: Arm D raised accuracy under FGSM evasion from 46.98% (Arm C) up to **89.95%** (+42.97% gain).
2. **Generalization Beyond FGSM**: Arm D also resisted non-gradient attacks better than clean models, raising HopSkipJump accuracy from 53.33% to **73.33%** (+20.00% gain) and Random Noise accuracy from 54.77% to **65.08%** (+10.31% gain).
3. **The "Robustness Tax"**: Arm D incurred a modest **-3.26%** drop in clean traffic accuracy (94.72% -> 91.46%), a well-documented phenomenon in adversarial defense literature where robust decision boundaries sacrifice slight precision on standard boundary margins.

---

## 5. Part C: Cryptographic Forensic Alert Logging

### Schema Specification (`logs/alerts.jsonl`)
Alerts are written as JSON Lines in a flattened, schema-uniform structure:
```json
{
  "alert_id": "ALT-DC985E1C077A",
  "timestamp_utc": "2026-10-29T08:39:53.460800+00:00",
  "timestamp_note": "Processing time (UTC). CICIoT2023 raw benchmark lacks ground-truth packet timestamps.",
  "flow_index": 20802,
  "predicted_class": "BruteForce",
  "true_class": "Benign",
  "confidence": 0.2898,
  "confidence_band": "low",
  "is_low_confidence": true,
  "severity": "Low",
  "model_version": "B2-Ensemble-8class",
  "adversarial": false,
  "source": "ollama",
  "top_shap_summary": "Tot size [SHAP: -0.6210, Raw: 645.2], syn_count [SHAP: +0.4345, Raw: 1.4], flow_duration [SHAP: +0.3978, Raw: 146.495]",
  "top_shap_features": [
    {
      "feature": "Tot size",
      "shap_value": -0.621,
      "raw_value": 645.2,
      "impact": "decreases_prediction"
    },
    {
      "feature": "syn_count",
      "shap_value": 0.4345,
      "raw_value": 1.4,
      "impact": "increases_prediction"
    }
  ],
  "top_shap_values": {
    "Tot size": -0.621,
    "syn_count": 0.4345
  },
  "top_raw_feature_values": {
    "Tot size": 645.2,
    "syn_count": 1.4
  },
  "explanation": "The network flow is classified as a BruteForce attack, but the model's confidence is low, suggesting uncertainty in this classification.",
  "recommended_action": "Network operators should investigate this event further to confirm or rule out the classification, as the low confidence level may indicate a false positive.",
  "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "entry_hash": "26ac9ea323962465bb1343d02e23148a8ea06e5f631a63db810fbb1f10af2ead"
}
```

### Verification & Tamper Evidence Test Results
- **Pristine Log Verification**: Evaluated 45 records (40 standard test flows + 5 adversarial flows).
  - Result: **100% PASS** (`[SUCCESS] Verified 45 alerts. Cryptographic hash chain is 100% intact.`).
- **Tampered Copy Verification**: Altered line 15's severity field in an exact duplicate log file.
  - Result: **CORRECTLY DETECTED TAMPER**
  ```
  [TAMPER DETECTED] Content modification at Line 15 (Alert ALT-55BDF1481737):
    Recorded entry_hash:   fb021f008035ef463ff71ee13eba0f634c5e5a5e96046ed65bd428197d3acc43
    Recomputed entry_hash: 2162037c1e26a21fe40b34e028c9e9aa77de4f9709ee5c0baed407738d42c039
    Payload was altered or forged after hashing.
  ```

---

## 6. Hardware & Resource Telemetry (8 GB Laptop Constraint)

All experiments were executed on local Windows hardware without GPU acceleration:
- **Phase 4 SHAP Analysis Peak RAM**: **987.3 MB (0.96 GB)** (runtime: 94.29 s)
- **Phase 4 Adversarial Suite Peak RAM**: **709.3 MB (0.69 GB)** (runtime: 113.27 s)
- **Phase 4 LLM Demo (`--use-shap`) Peak RAM**: **315.4 MB (0.31 GB)** (runtime: 481.6 s)
- **Ceiling Compliance**: Peak RAM across Phase 4 never exceeded **0.99 GB**, leaving > 5.0 GB free memory.

---

## 7. Honest Limitations & Threat Validity

1. **Surrogate Gradient Transferability**: Because tree ensembles are non-differentiable step functions, white-box FGSM attacks rely on the PyTorch DNN surrogate. Although transfer drops tree accuracy by up to 56.6%, the attack is not computing the exact subgradient of the ensemble's tree leaves.
2. **Computational Query Budgets in Black-Box Attacks**: Due to single-threaded CPU constraints, HopSkipJump and ZOO were evaluated with small query budgets (3-8 iterations) on modest stratified samples (10-20 flows). Higher budgets would likely achieve higher ASR but exceed interactive execution time limits.
3. **Simulated Domain Constraints**: While our constraints freeze protocol flags and bound timing/size metrics, real operational networks impose stateful TCP window dynamics, retransmissions, and Middlebox packet normalization that are not captured in static CSV flow records.
4. **No Native Timestamps in Benchmark**: CICIoT2023 lacks ground-truth packet capture timestamps. All alert timestamps represent local engine processing time (UTC).
5. **Robustness Trade-Off**: While adversarial retraining significantly hardens models against evasion (+42.97% under FGSM, +20.00% under HopSkipJump), it introduces a standard 3.26% reduction in clean classification accuracy. In production, this trade-off must be calibrated to organizational false-positive tolerances.
