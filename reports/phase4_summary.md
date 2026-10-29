# Phase 4 - Explainable AI (SHAP), Adversarial Robustness & Forensic Alert Logging

> **Generated**: 2026-10-29  
> **Repository Branch**: `phase4-xai-adversarial`  
> **Primary Classifier**: Method B2 Ensemble (Random Forest 0.60 + XGBoost 0.40 + PyTorch DNN 0.00)  
> **Target Dataset**: CICIoT2023 (19 RFE-selected features)  
> **Hardware Environment**: Local Windows Laptop, 8 GB RAM, CPU-only (No GPU)  
> **Measured Peak RAM**: **987.3 MB (0.96 GB)** during SHAP TreeExplainer; **680.9 MB (0.67 GB)** during Staged Adversarial Suite. *(Strict hardware ceiling: < 6.0 GB)*

---

## 1. Executive Summary & Flowchart Alignment

Phase 4 covers three key boxes of the intrusion detection flowchart:
1. **Explainable AI (SHAP TreeExplainer)**:
   - Provides Shapley feature attributions for the tree components of the B2 ensemble.
   - Generates global feature importance rankings, per-class top feature distributions, and local sample explanations.
   - Measures architectural feature stability between Random Forest (weight 0.60) and XGBoost (weight 0.40).
   - Integrates with the LLM Semantic Orchestrator via an opt-in `--use-shap` flag.
2. **Adversarial Robustness Assessment (Adversarial Robustness Toolbox - ART)**:
   - **Zero-Attack Pipeline Control ($\epsilon = 0$)**: Confirms bit-for-bit numerical identity ($0.000000$ drift).
   - **Random-Noise Control & Fine-Grained Noise Sweep**: Measures accuracy drops at small perturbations ($L_2 \in [0.0001, 0.3000]$) and explains step-function decision boundary concentration.
   - **PyTorch DNN Surrogate**: Re-established clean baseline at **77.60%** (388/500 flows) for black-box evasion transfer gradient generation.
   - **Black-Box Attacks**: Measures **HopSkipJump** (Budget 1 on 100 flows, Budget 2 on 50 flows) and **ZOO** (Budget 1 & 2 on 40 flows) with exact counts and Wilson 95% confidence intervals.
   - **4-Arm Defense Matrix**: Evaluates Arm A (Original B2), Arm B (Retrained Clean 1x), Arm C (Retrained Clean 2x volume control), and Arm D (Retrained Clean + Adv) across Clean, FGSM, Random Noise, HopSkipJump (50 flows/arm), and held-out attack conditions ($\epsilon=0.20$).
3. **Cryptographic Tamper-Evident Forensic Alert Logging**:
   - Persists detection records to `logs/alerts.jsonl` in a flat, ELK/Kibana-ready schema.
   - Stores TreeExplainer `top_shap_values` and physical `top_raw_feature_values` in separate, explicitly typed fields.
   - Implements cryptographic SHA-256 hash chaining (`prev_hash` -> `entry_hash`) and an external tail hash anchor (`logs/alerts_tail_hash.txt`) to detect file truncation.
   - Validates log integrity across automated audit tests (pristine, mid-file modification, tail deletion).

---

## 2. Part A: SHAP Explainability Evaluation

### Architectural Note on Ensemble Explainability
The B2 Ensemble is a weighted soft-voting mixture:
$$P_{\text{ens}}(y = c \mid x) = 0.60 \cdot P_{\text{RF}}(y = c \mid x) + 0.40 \cdot P_{\text{XGB}}(y = c \mid x) + 0.00 \cdot P_{\text{DNN}}(y = c \mid x)$$
Because `shap.TreeExplainer` operates in the log-odds margin space of individual decision trees, a convex mixture of disparate probability estimators cannot be represented as a single tree. Therefore:
- **Global & Local Explanations**: Computed on **XGBoost** (weight = 0.40), which evaluates in milliseconds per sample.
- **Stability Cross-Check**: Evaluated against **Random Forest** (weight = 0.60) on a stratified sample to evaluate whether both tree models rely on similar physical features.

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
- **Conclusion**: Both models place high importance on the same underlying physical telemetry dimensions.

### LLM Telemetry Grounding & Feature Phrase Auditing
Audited via `xai/compare_telemetry_grounding.py` across all 40 demo flows for both Phase 3 (static telemetry prompt) and Phase 4 (dynamic SHAP top-5 telemetry prompt):
- **Unmatched Numbers**: **0 / 40** in Phase 4; **0 / 40** in Phase 3. Both runs adhered strictly to provided numerical values.
- **Unprovided Feature Phrases**:
  - **Phase 3**: **0 / 40** unprovided features. `flow_duration` was provided in every prompt; Phase 3 explanations frequently cited duration ("due to its low duration" in 11/40 flows) because it was statically present in every prompt.
  - **Phase 4**: **3 / 40** flows (Flows 11497, 25398, 20091) were flagged for using the colloquial term "volume" when `Tot size` or `Tot sum` were not in the dynamic top-5 SHAP features.
- **Duration Fixation**: In Phase 4 with SHAP-driven dynamic feature selection, generic duration claims dropped to **0 / 40 (0.0%)**, grounding explanations in the real top contributors for that specific flow.
- **Artifact**: Saved to `reports/telemetry_grounding_comparison.json`.

---

## 3. Part B: Adversarial Robustness Assessment

### Realistic Domain Constraints
Adversarial perturbations in network intrusion detection must respect network protocol invariants:

| Feature Name | Feature Type | Empirical Range | Attacker Modifiable | Rationale / Physical Constraint |
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
The constraint enforcement pipeline (`unscale -> clip -> round -> rescale`) was applied to unperturbed test flows:
- **Maximum Numerical Drift ($|X_{\text{pipe}} - X_{\text{clean}}|$)**: **0.000000** (exact bit-for-bit identity).
- **PyTorch DNN Accuracy**: Raw clean = **77.60% (388/500)** | Pipeline-passed = **77.60% (388/500)**.
- **B2 Ensemble Accuracy**: Raw clean = **95.60% (478/500)** | Pipeline-passed = **95.60% (478/500)**.

### Random Noise Sweep & Decision Boundary Fragility Audit
To investigate why tree ensemble accuracy degrades even under small random perturbations, a fine-grained Gaussian noise sweep was executed on 500 stratified test flows, comparing domain-constrained perturbations against an unconstrained isotropic Gaussian control:

| Noise Target $ | Measured Mean $ | Constrained Correct Flows | Constrained Accuracy (95% CI) | Unconstrained Correct Flows | Unconstrained Accuracy (95% CI) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean Baseline** | 0.0000 | 478 / 500 | **95.6%** [93.4%, 97.1%] | 478 / 500 | **95.6%** [93.4%, 97.1%] |
| ** = 0.0001$** | 0.0001 | 471 / 500 | **94.2%** [91.8%, 95.9%] | 475 / 500 | **95.0%** [92.7%, 96.6%] |
| ** = 0.0010$** | 0.0009 | 452 / 500 | **90.4%** [87.5%, 92.7%] | 459 / 500 | **91.8%** [89.1%, 93.9%] |
| ** = 0.0100$** | 0.0093 | 382 / 500 | **76.4%** [72.5%, 79.9%] | 388 / 500 | **77.6%** [73.7%, 81.0%] |
| ** = 0.0300$** | 0.0278 | 330 / 500 | **66.0%** [61.7%, 70.0%] | 337 / 500 | **67.4%** [63.2%, 71.4%] |
| ** = 0.1000$** | 0.0902 | 329 / 500 | **65.8%** [61.5%, 69.8%] | 327 / 500 | **65.4%** [61.1%, 69.4%] |
| ** = 0.3000$** | 0.2568 | 305 / 500 | **61.0%** [56.7%, 65.2%] | 260 / 500 | **52.0%** [47.6%, 56.4%] |

#### Tree Split Threshold Audit (Why the Accuracy Cliff Exists)
Auditing the Random Forest component actually loaded from the deployed B2 model (200 trees, max_depth 25, min_samples_leaf 2):
1. **High Point-Mass Concentration**: **45.26% of test flows** (11,901 of 26,292 flows) sit at low_duration = -0.0662 (scaled value corresponding to 0.00 duration, single-packet flows). Similar mass points exist at 0 for Covariance (46.03% of flows) and Variance (35.15% of flows).
2. **Dense Tree Split Clustering**: The deployed Random Forest has **26914** split thresholds within $\pm 0.01$ and **45308** split thresholds within $\pm 0.05$ of low_duration = -0.0662. (For comparison, the retrained defense arm forest with 100 trees and max depth 20 contained 13,842 splits within $\pm 0.01$ and 23,190 splits within $\pm 0.05$).
3. **Fragility Finding & Association**: Because nearly half the dataset sits exactly on point masses adjacent to thousands of split thresholds, even tiny isotropic shifts push thousands of single-packet flows across multiple tree split boundaries simultaneously. Unconstrained isotropic Gaussian noise produced degradation similar to constrained noise (e.g., 388/500 vs 382/500 at  = 0.01$). This empirical behavior is consistent with an association between discrete feature concentration and tree threshold clustering; this split density aligns with sensitivity to small shifts, though this association is consistent with fragility and does not prove sole causation.

### FGSM Transfer Evasion vs. Random-Noise Control (500 Test Flows)
White-box gradients were computed on the PyTorch DNN surrogate (77.60% clean baseline) and evaluated for black-box transfer against the B2 Ensemble. An isotropic Gaussian noise control matching the sample-wise $L_2$ magnitude was evaluated in parallel:

| Epsilon ($\epsilon$) | Mean Perturbation ($L_2$) | PyTorch DNN Acc (White-Box) | B2 Ensemble Acc (FGSM Transfer) | Transfer ASR | Ensemble Acc (Random Noise Control) | Random Noise ASR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\epsilon = 0.00$** | 0.000 | 77.60% (388/500) | 95.60% (478/500) [93.4%, 97.1%] | 0.00% (0/478) | 95.60% (478/500) [93.4%, 97.1%] | 0.00% (0/478) |
| **$\epsilon = 0.01$** | 0.031 | 77.40% (387/500) | 57.00% (285/500) [52.6%, 61.3%] | 41.00% (196/478) [36.7%, 45.5%] | 70.60% (353/500) [66.5%, 74.4%] | 26.99% (129/478) |
| **$\epsilon = 0.05$** | 0.152 | 76.00% (380/500) | 53.20% (266/500) [48.8%, 57.5%] | 45.82% (219/478) [41.4%, 50.3%] | 61.00% (305/500) [56.7%, 65.2%] | 37.24% (178/478) |
| **$\epsilon = 0.10$** | 0.294 | 73.60% (368/500) | 51.00% (255/500) [46.6%, 55.4%] | 47.91% (229/478) [43.5%, 52.4%] | 59.20% (296/500) [54.8%, 63.4%] | 39.12% (187/478) |
| **$\epsilon = 0.20$** | 0.569 | 62.40% (312/500) | 47.80% (239/500) [43.5%, 52.2%] | 51.67% (247/478) [47.2%, 56.1%] | 56.60% (283/500) [52.2%, 60.9%] | 42.05% (201/478) |
| **$\epsilon = 0.30$** | 0.841 | 57.00% (285/500) | 39.00% (195/500) [34.8%, 43.3%] | 60.88% (291/478) [56.4%, 65.2%] | 53.00% (265/500) [48.6%, 57.3%] | 45.19% (216/478) |

> **Finding**: FGSM transfer evasion exceeds random noise across all evaluated perturbation levels by **+8.6% to +15.7% ASR**. Even small gradient-guided shifts ($\epsilon = 0.01$, mean $L_2 = 0.031$) drop tree ensemble accuracy to 57.0%, demonstrating that surrogate gradient directions capture shared tabular vulnerabilities.

### HopSkipJump Black-Box Attack (Multi-Budget Comparison on Same 100 Flows)
Evaluated directly on the B2 Tree Ensemble across decision boundaries on the same fixed, stratified set of 100 flows for both budgets:

| Query Budget Setting | Sample Size | Max Iter | Max Eval | Clean Accuracy | Perturbed Accuracy (Constrained) | Attack Success Rate (ASR) | Mean $ | Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Budget 1 (Low)** | 100 flows | 3 | 10 | **93.0%** (93/100) [86.3%, 96.6%] | **31.0%** (31/100) [22.8%, 40.6%] | **68.8%** (64/93) [58.8%, 77.3%] | 0.3983 | 108.88 s |
| **Budget 2 (High)** | 100 flows | 6 | 20 | **93.0%** (93/100) [86.3%, 96.6%] | **38.0%** (38/100) [29.1%, 47.8%] | **61.3%** (57/93) [51.1%, 70.6%] | 0.2375 | 231.01 s |

*Methodological Rigor & Constraint Effects*:
- **Same Flow Comparison**: Both budgets evaluate the exact same 100 flows under identical class distributions.
- **Unconstrained vs. Constrained ASR**:
  - Raw unconstrained attacks achieved 100.0% ASR (93/93) with mean  = 0.5745$ (Budget 1) and  = 0.3382$ (Budget 2).
  - Domain constraint enforcement reversed **29 of 93 (31.2%)** evasions for Budget 1, and **36 of 93 (38.7%)** evasions for Budget 2. Higher iteration budgets achieve finer perturbations closer to the boundary ( = 0.2375$ vs .3983$), making them more susceptible to domain constraint snapping.
- **Execution Safeguards**: Run flow-by-flow with a 30s per-flow timeout and 1,000 query cap. Zero candidate flows timed out (1 non-candidate flow hit the query cap in Budget 1 and was bounded safely).
- **Small-Sample Note**: For  = 100$ flows, the 95% Wilson confidence intervals span approximately $\pm 8\text{--}9\%$, reflecting empirical sample size.

### Zeroth-Order Optimization (ZOO) Attack (Multi-Budget Comparison)
Score-based coordinate descent attack against B2 Ensemble probabilities on a fixed set of 40 flows:

| Query Budget Setting | Sample Size | Max Iter | Clean Accuracy | Perturbed Accuracy | Attack Success Rate (ASR) | Mean $L_2$ | Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Budget 1 (Low: 4 iters)** | 40 flows | 4 | **90.0%** (36/40) [76.9%, 96.0%] | **80.0%** (32/40) [65.2%, 89.5%] | **11.1%** (4/36) [4.4%, 25.3%] | 0.0397 | 52.49 s |
| **Budget 2 (High: 8 iters)** | 40 flows | 8 | **90.0%** (36/40) [76.9%, 96.0%] | **70.0%** (28/40) [54.6%, 81.9%] | **22.2%** (8/36) [11.7%, 38.1%] | 0.0632 | 104.98 s |

> **Artifact**: Unified robustness comparison plot saved to `reports/adversarial_accuracy_vs_epsilon.png`.

---

## 4. Defense Experiment: 4-Arm Multi-Condition Matrix

Four model arms were trained and evaluated on **the same fresh test split of 500 flows** (and 100 fresh flows per arm for HopSkipJump):
- **Arm A**: Original B2 Ensemble (trained on full dataset).
- **Arm B**: Retrained Clean 1x (3,000 clean training flows).
- **Arm C**: Retrained Clean 2x (6,000 clean training flows, data volume control).
- **Arm D**: Retrained Clean + Adversarial (3,000 clean + 3,000 FGSM $\epsilon=0.10$ training flows).

### 4-Arm Benchmark Matrix

| Model Arm | Clean (500 flows) | FGSM $\epsilon=0.10$ (500 flows) | Random Noise (500 flows) | HSJ Budget 1 (100 flows/arm) | Held-Out FGSM $\epsilon=0.20$ (500 flows) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Arm A: Original B2** | **95.6%** (478/500) [93.4%, 97.1%] | 52.6% (263/500) [48.2%, 56.9%] | 61.6% (308/500) [57.3%, 65.8%] | 39.0% (39/100) [30.0%, 48.8%] | 49.6% (248/500) [45.2%, 54.0%] |
| **Arm B: Retrained 1x (3k)** | 92.0% (460/500) [89.3%, 94.1%] | 44.4% (222/500) [40.1%, 48.8%] | 57.0% (285/500) [52.6%, 61.3%] | 41.0% (41/100) [31.9%, 50.8%] | 42.2% (211/500) [38.0%, 46.6%] |
| **Arm C: Retrained 2x (6k)** | 91.6% (458/500) [88.8%, 93.7%] | 48.2% (241/500) [43.9%, 52.6%] | 58.0% (290/500) [53.6%, 62.3%] | 38.0% (38/100) [29.1%, 47.8%] | 45.4% (227/500) [41.1%, 49.8%] |
| **Arm D: Retrained Clean+Adv** | 89.4% (447/500) [86.4%, 91.8%] | **88.8%** (444/500) [85.7%, 91.3%] | **69.0%** (345/500) [64.8%, 72.9%] | **56.0%** (56/100) [46.2%, 65.3%] | **63.0%** (315/500) [58.7%, 67.1%] |

### Honest Defense Assessment
1. **Specific Hardening on Trained Attack**: Arm D hardened effectively against the exact attack family it was trained on (FGSM $\epsilon=0.10$), reaching **88.8% accuracy (444/500)** (compared to 48.2% (241/500) for Arm C).
2. **Limited Generalization**: Arm D **does not generalize broadly** across arbitrary perturbations:
   - Under isotropic random noise of matching $, accuracy drops to **69.0% (345/500)**.
   - Under held-out FGSM at doubled magnitude ($\epsilon=0.20$), accuracy drops to **63.0% (315/500)**.
   - Under black-box HopSkipJump (100 flows), accuracy drops to **56.0% (56/100) [46.2%, 65.3%]**.
   Arm D hardened specifically against the perturbation vector it observed during training.
3. **The Robustness Tax**: Arm D exhibits a **-2.2% drop** in clean test accuracy (89.4% (447/500) vs 91.6% (458/500) for Arm C, and 95.6% (478/500) for full-data Arm A).
4. **Statistical Uncertainty**: For test conditions with  = 100$ flows, the confidence interval is wider ($\pm 9.5\%$, [46.2%, 65.3%]); claims of generalization must reflect this empirical uncertainty.


---

## 5. Part C: Cryptographic Forensic Alert Logging & External Anchor

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

### Threat Model & Anchor Mechanics
- **Mid-Stream Modification**: Detected by verifying that `recompute_hash(entry) == entry['entry_hash']` and `entry['prev_hash'] == previous_entry['entry_hash']`.
- **Tail Deletion / Truncation**: A naive hash chain cannot detect tail deletion (truncating the last $N$ lines still leaves a valid self-consistent chain). To prevent this, the latest hash is recorded in an external anchor file: [`logs/alerts_tail_hash.txt`](logs/alerts_tail_hash.txt). If the log's terminal hash does not match the external anchor, truncation is flagged immediately.
- **HMAC Consideration**: If an adversary possesses local filesystem write permissions to both `alerts.jsonl` and `alerts_tail_hash.txt`, they could rewrite the entire log and recompute all hashes. To defend against this higher threat model, HMAC-SHA256 with a secret key held in a Hardware Security Module (HSM), TPM, or remote syslog server would be required so that attackers cannot forge valid MAC tags.

### Verification Audit Results
- **Pristine Log**: 45/45 records verified (**100% PASS**).
- **Mid-File Tamper**: Modifying line 15 severity triggered immediate detection:
  ```
  [TAMPER DETECTED] Content modification at Line 15 (Alert ALT-55BDF1481737):
    Recorded entry_hash:   fb021f008035ef463ff71ee13eba0f634c5e5a5e96046ed65bd428197d3acc43
    Recomputed entry_hash: 2162037c1e26a21fe40b34e028c9e9aa77de4f9709ee5c0baed407738d42c039
  ```
- **Tail Truncation**: Truncating the last 5 records was detected against `logs/alerts_tail_hash.txt`:
  ```
  [TAIL TRUNCATION DETECTED] Terminal alert entry_hash does not match external anchor!
    Log tail hash:    7cfb2ff608c792aa...
    Anchor file hash: 395c1097fa3d2ea5...
  ```

---

## 6. Hardware & Resource Telemetry

All stages were evaluated on a local Windows laptop without GPU acceleration:
- **SHAP Analysis Peak RAM**: **987.3 MB (0.96 GB)**
- **Adversarial Suite Peak RAM**: **680.9 MB (0.67 GB)**
- **Staged Suite Runtimes**:
  - Stage 1 (FGSM & Controls): **1.97 s**
  - Stage 2 (HopSkipJump, 100+50 flows): **158.32 s** (~2.6 min)
  - Stage 3 (ZOO, 40 flows): **157.47 s** (~2.6 min)
  - Stage 4 (Defense Matrix, 4 arms, 500 flows): **99.62 s** (~1.6 min)
  - Stage 5 (Combine & Plot): **0.45 s**
  - **Total Suite Execution Time**: **417.83 s (~7.0 minutes)**.
- **Hardware Safety Ceiling**: Maintained under 1.0 GB RAM throughout (safety ceiling: < 6.0 GB).

---

## 7. Honest Limitations

1. **Demonstration Sample Sizes**: Evaluations on 40 to 100 flows (such as ZOO on 40 flows and HopSkipJump on 50–100 flows) are small-sample demonstrations. While Wilson 95% confidence intervals are reported, the error margins remain wide ($\pm 10–15\%$).
2. **Surrogate Transferability**: White-box FGSM attacks rely on the differentiable PyTorch DNN surrogate. Although the transfer drops ensemble accuracy by up to 56.6%, the gradients are approximations of the decision landscape of the tree ensemble rather than true subgradients.
3. **Static Tabular Simplification**: Real operational networks feature stateful TCP retransmissions, middlebox normalization, and dynamic round-trip pacing that static tabular CSV records cannot fully capture.
4. **LLM Explanations Are Unverified Text**: Explanations generated by the LLM are synthetic narrative text grounded in classifier outputs; they do not represent verified ground-truth attack semantics or causal forensics.
