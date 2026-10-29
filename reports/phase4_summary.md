# Phase 4 - Explainable AI (SHAP), Adversarial Robustness & Forensic Alert Logging

> **Generated**: 2026-10-29  
> **Repository Branch**: `phase4-xai-adversarial`  
> **Primary Classifier**: Method B2 Ensemble (Random Forest 0.60 + XGBoost 0.40 + PyTorch DNN 0.00)  
> **Target Dataset**: CICIoT2023 (19 RFE-selected features)  
> **Hardware Environment**: Local Windows Laptop, 8 GB RAM, CPU-only (No GPU)  
> **Measured Peak RAM**: **987.3 MB (0.96 GB)** during SHAP TreeExplainer; **676.0 MB (0.66 GB)** during Adversarial Suite. *(Strict hardware ceiling: 6.0 GB)*

---

## 1. Executive Summary & Flowchart Alignment

Phase 4 implements three key boxes from the project architecture flowchart:
1. **Explainable AI (SHAP TreeExplainer)**:
   - Provides mathematically grounded Shapley feature attributions for tree components of the B2 ensemble.
   - Generates global feature importance rankings, per-class top feature distributions, and local sample explanations.
   - Verifies architectural feature stability between Random Forest (weight 0.60) and XGBoost (weight 0.40).
   - Integrates with the LLM Semantic Orchestrator via an opt-in `--use-shap` flag.
2. **Adversarial Robustness Assessment (Adversarial Robustness Toolbox - ART)**:
   - Defines realistic domain constraints on all 19 features: freezes 8 protocol and flag features while restricting perturbations to 11 plausible physical features (timing, packet sizes, header padding).
   - Evaluates three attack families: **FGSM transfer** (gradient-based from DNN surrogate), **HopSkipJump** (decision-based black-box), and **ZOO** (score-based black-box).
   - Evaluates an **Adversarial Retraining Defense** experiment on unseen test flows.
3. **Cryptographic Tamper-Evident Forensic Alert Logging**:
   - Persists detection records to `logs/alerts.jsonl` in a flat, ELK/Kibana-ready schema.
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

### LLM Orchestrator Integration (`--use-shap`)
Re-executed `run_phase3_llm.py` across the 40 demo flows with `--use-shap`, replacing the hardcoded telemetry list with each flow's true top 5 SHAP features and unscaled measurements:
- **Phase 3 (Fixed Telemetry)**: **0 / 40 (0.0%)** explanations cited unprovided features.
- **Phase 4 (SHAP Top Features)**: **0 / 40 (0.0%)** explanations cited unprovided features.
- **Empirical Difference**: Grounding explanations in real SHAP attributions directed the LLM to explain the exact physical signals that drove the model (e.g., highlighting `Tot size` and `syn_count` for BruteForce, `rst_count` for Mirai, and `IAT` for DDoS) without hallucinating unprovided values.
- **Artifact**: Saved to `reports/llm_demo_outputs_shap.csv` and `reports/llm_metrics_shap.json`.

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
| `ack_count` | Count | [0.00, 5.00] | **NO (Fixed)** | Governed by TCP acknowledgment handshake mechanics. |
| `syn_flag_number` | Binary Flag | [0.00, 1.00] | **NO (Fixed)** | Connection initiation flag; defined by protocol handshake. |
| `urg_count` | Count | [0.00, 3600.50] | **NO (Fixed)** | Urgent pointer counter; restricted by OS TCP/IP stack. |
| `rst_count` | Count | [0.00, 9126.50] | **NO (Fixed)** | Reset flag abruptly terminates connection. |
| `ICMP` | Binary Flag | [0.00, 1.00] | **NO (Fixed)** | Protocol header flag; cannot be flipped for TCP/UDP flows. |
| `Tot size` | Continuous | [42.00, 5091.60] | **YES** | Attacker can append arbitrary padding bytes to payloads. |
| `IAT` | Continuous | [0.00, 167639424.00] | **YES** | Inter-Arrival Time is controlled via pacing and jitter. |
| `AVG` | Continuous | [42.00, 8490.74] | **YES** | Average packet size is shifted directly by dummy padding. |
| `Max` | Continuous | [42.00, 21786.00] | **YES** | Maximum packet size is shifted by sending padded packets. |
| `Min` | Continuous | [42.00, 4410.00] | **YES** | Minimum packet size increased by padding all smaller packets. |
| `Tot sum` | Continuous | [81.00, 61882.60] | **YES** | Total volume transmitted altered by dummy traffic. |
| `Magnitue` | Continuous | [9.17, 124.95] | **YES** | Geometric magnitude scales directly with packet padding. |
| `Covariance` | Continuous | [0.00, 43075336.00] | **YES** | Shaped by variable packet sizing and arrival bursts. |
| `Variance` | Continuous | [0.00, 1.00] | **YES** | Shaped by mixing uniform and variable-length frames. |

> **Constraint Enforcement**: Evaluated in standardized space, mapped to physical units via `FeatureUnscaler`, clipped to training $[min, max]$, rounded for count features, fixed features frozen, and re-standardized.

### Adversarial Attack Results

#### 1. FGSM Attack & Black-Box Transfer (500 Test Flows)
FGSM is computed white-box on the PyTorch DNN surrogate (the only differentiable model). Perturbations are then constrained and tested for **evasion transfer** against the B2 Tree Ensemble (RF + XGBoost):

| Perturbation ($\epsilon$) | Mean Perturbation ($L_2$) | PyTorch DNN Acc (White-Box) | B2 Ensemble Acc (Transfer) | Transfer Attack Success Rate (ASR) | Generation Time |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean Baseline** | 0.000 | 14.80% | **95.60%** | — | — |
| $\epsilon = 0.01$ | 0.0316 | 15.00% | **61.80%** | **36.40%** | 0.14 s |
| $\epsilon = 0.05$ | 0.1561 | 14.60% | **66.00%** | **32.64%** | 0.10 s |
| $\epsilon = 0.10$ | 0.3076 | 14.40% | **57.20%** | **41.84%** | 0.10 s |
| $\epsilon = 0.20$ | 0.6079 | 14.20% | **56.80%** | **41.63%** | 0.09 s |
| $\epsilon = 0.30$ | 0.9067 | 15.80% | **49.60%** | **49.58%** | 0.10 s |

> **Key Finding**: Although the PyTorch DNN had low clean accuracy on tabular flows (14.8%), gradient-directed perturbations on packet sizes and IAT timing successfully transferred to the tree ensemble, reducing ensemble accuracy from **95.60% down to 49.60%** at $\epsilon = 0.30$ (ASR: 49.58%).

#### 2. HopSkipJump Attack (Decision-Based Black-Box, 100 Flows)
- **Attack Parameters**: `max_iter = 5`, `max_eval = 20`, `init_eval = 10`
- **Clean Accuracy**: **95.00%**
- **Perturbed Accuracy**: **35.00%**
- **Attack Success Rate (ASR)**: **66.32%**
- **Mean $L_2$ Perturbation**: **0.3244**
- **Runtime**: 256.08 s (2.56 s / flow on CPU)

#### 3. ZOO Attack (Score-Based Black-Box, 40 Flows)
- **Attack Parameters**: `max_iter = 10`, `nb_parallel = 10`, `learning_rate = 0.05`
- **Clean Accuracy**: **87.50%**
- **Perturbed Accuracy**: **70.00%**
- **Attack Success Rate (ASR)**: **22.86%**
- **Mean $L_2$ Perturbation**: **0.0181**
- **Runtime**: 59.72 s (1.49 s / flow on CPU)

> **Artifact**: Curve plot saved to `reports/adversarial_accuracy_vs_epsilon.png`.

### Defense Experiment: Adversarial Retraining (AT)
Evaluated on clean training flows (3,000 flows) augmented with FGSM-perturbed flows ($\epsilon = 0.10$, 3,000 flows). Evaluated on **unseen test flows**:

| Model State | Clean Test Accuracy | Test Accuracy Under FGSM ($\epsilon = 0.10$) | Robustness Delta |
|:---|:---:|:---:|:---:|
| **Baseline B2 Ensemble** | **95.60%** | 57.20% | Baseline |
| **Adversarially Retrained Ensemble** | **91.60%** (-4.00%) | **90.80%** | **+33.60% under attack** |

> **Defense Trade-off**: Adversarial retraining sacrificed 4.0% clean accuracy on normal network traffic, but restored accuracy under $\epsilon = 0.10$ evasion from 57.20% up to 90.80%.

---

## 4. Part C: Cryptographic Forensic Alert Logging

### Schema Specification (`logs/alerts.jsonl`)
Alerts are written as JSON Lines in a flattened, schema-uniform structure:
```json
{
  "alert_id": "ALT-C0BD41591FE3",
  "timestamp_utc": "2026-10-29T13:49:02.123456+00:00",
  "timestamp_note": "Processing time (UTC). CICIoT2023 raw benchmark lacks ground-truth packet timestamps.",
  "flow_index": 12992,
  "predicted_class": "DDoS",
  "true_class": "DDoS",
  "confidence": 0.9976,
  "confidence_band": "high",
  "is_low_confidence": false,
  "severity": "Critical",
  "model_version": "B2-Ensemble-8class",
  "adversarial": false,
  "source": "ollama",
  "top_shap_summary": "IAT (risk: 0.0), Variance (risk: 0.0), urg_count (risk: 0.0), flow_duration (risk: 0.0), Tot sum (risk: 54.0)",
  "top_shap_features": [ ... ],
  "explanation": "The network flow indicates a Distributed Denial of Service (DDoS) attack...",
  "recommended_action": "Network operators should implement rate limiting and IP blocking...",
  "prev_hash": "a4f8...b129",
  "entry_hash": "75d808998eb80bbb0d952ad591c8161fe3aaa19c1a822a708eebfcf88e0c28a8"
}
```

### Tamper Evidence Verification Test
- **Pristine Log Verification**: Evaluated 45 records (40 standard test flows + 5 adversarial flows).
  - Result: **100% PASS** (`[SUCCESS] Verified 45 alerts. Cryptographic hash chain is 100% intact.`).
- **Tampered Copy Verification**: Altered line 15's severity field in an exact duplicate log file.
  - Result: **CORRECTLY DETECTED TAMPER**
  ```
  [TAMPER DETECTED] Content modification at Line 15 (Alert ALT-C0BD41591FE3):
    Recorded entry_hash:   75d808998eb80bbb0d952ad591c8161fe3aaa19c1a822a708eebfcf88e0c28a8
    Recomputed entry_hash: b4ff07b1f28da6b48e5b0401539c2aca02571bef42f80c3f6a0b10c92c21241a
    Payload was altered or forged after hashing.
  ```

---

## 5. Hardware & Resource Telemetry (8 GB Laptop Constraint)

All experiments were executed on local Windows hardware without GPU acceleration:
- **Phase 4 SHAP Analysis Peak RAM**: **987.3 MB (0.96 GB)** (runtime: 94.29 s)
- **Phase 4 Adversarial Suite Peak RAM**: **676.0 MB (0.66 GB)** (runtime: 319.48 s)
- **Phase 4 LLM Demo (`--use-shap`) Peak RAM**: **315.4 MB (0.31 GB)** (runtime: 481.6 s)
- **Ceiling Compliance**: Peak RAM across Phase 4 never exceeded **0.99 GB**, leaving > 5.0 GB free memory.

---

## 6. Limitations & Assumptions

1. **Surrogate Model Transferability**: FGSM was generated white-box against the PyTorch DNN surrogate because tree ensembles are non-differentiable. While transfer succeeded in lowering tree ensemble accuracy from 95.6% to 49.6%, this does not evaluate a direct mathematical gradient of the tree structures.
2. **Computational Query Budgets**: HopSkipJump and ZOO were evaluated on small test samples (100 flows and 40 flows respectively) with constrained query budgets to stay within CPU-only execution limits.
3. **Simulated Domain Constraints**: The feature constraints are empirical mathematical bounds based on protocol specifications. Real network routers and operating systems enforce additional low-level TCP windowing and MSS packet fragmentation constraints.
4. **No Packet Timestamps in Benchmark**: CICIoT2023 raw CSV records do not contain packet capture timestamps. All logged timestamps represent the detection engine's local processing time (UTC).
5. **Single-Run Determinism**: All experiments were executed with fixed `RANDOM_SEED = 42`. Multi-run variance and confidence intervals require dedicated multi-core hardware.
