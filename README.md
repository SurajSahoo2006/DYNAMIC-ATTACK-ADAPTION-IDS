# Dynamic Attack Adaptation — IDS Pipeline
### B.Tech Sem IV Research Project | NMIMS Design Experience / IROP

> **Project**: AI/ML pipeline for adaptive network intrusion detection in IoT environments.  
> **Dataset**: CICIoT2023 (University of New Brunswick) — 47M+ records, 33 attack types.  
> **Branch**: `phase4-xai-adversarial`  
> **Current Pipeline Scope**:  
> - **Phase 1**: Chunked Ingestion & Invariant Validation  
> - **Phase 2**: Leak-Free Preprocessing & Recursive Feature Elimination (46 → 19 features)  
> - **Phase 3**: Multi-Model Ensemble (Method B2) & LLM Semantic Orchestrator (Ollama Llama-3.2:3b)  
> - **Phase 4**: Explainable AI (SHAP TreeExplainer), Realistic Adversarial Robustness (ART), and Tamper-Evident Forensic Logging

---

## Architecture Overview

```
Network Traffic (PCAP)          Honeypot (Cowrie JSON)*      Benchmark Datasets
        |                               |                            |
        +---------------+---------------+----------------------------+
                        |
                [Phase 1: Ingest]
                load + validate + sample
                        |
                [Phase 2: Preprocess]
                clean -> encode -> SPLIT -> scale -> SMOTE -> RFE (top 19)
                        |
              data/processed/*.parquet    artifacts/
                        |
            [Phase 3A: Ensemble Classifier (Method B2)]
            ├── Random Forest (0.60) + XGBoost (0.40) + PyTorch DNN (0.00)
            │
            +---------------------------------------+
            |                                       |
            ▼                                       ▼
    [Phase 4A: Explainable AI]             [Phase 4B: Adversarial Suite]
    ├── SHAP TreeExplainer (2,000 flows)   ├── Domain Constraints (11 modifiable / 8 fixed)
    ├── Stability check (rho = 0.9088)     ├── FGSM Transfer (White-box DNN -> Ensemble)
    └── Opt-in --use-shap telemetry         ├── HopSkipJump & ZOO Black-Box Attacks
            |                              └── Adversarial Retraining Defense
            |                                       |
            +-------------------+-------------------+
                                |
                                ▼
            [Phase 3B: LLM Semantic Orchestrator]
            └── Ollama Llama-3.2:3b (CPU-only, temp=0) / Deterministic Expert Fallback
                (Translates predictions & SHAP attributions into analyst playbooks)
                                |
                                ▼
            [Phase 4C: Forensic Alert Logging]
            └── Cryptographic SHA-256 Tamper-Evident Audit Trail (logs/alerts.jsonl)
                                |
            [Phase 5+: Future Strategic Phases]
            MARL Tactical Defense Agents (MAPPO) -> Full ELK Stack / SIEM Dashboard

* Note: Honeypot log enrichment is future work; current pipeline consumes statistical flow vectors.
```

---

## Repository Structure

```
.
├── config.py                         # Central configuration (paths, hyperparameters, mappings)
├── requirements.txt
├── run_phase1_2.py                   # Phase 1 & 2 pipeline runner
├── run_phase3_ensemble.py            # Phase 3 ensemble training & evaluation runner
├── run_phase3_llm.py                 # Phase 3/4 LLM orchestrator runner (--use-shap flag)
├── predict.py                        # Standalone flow prediction & severity module (B2 default)
│
├── xai/                              # Phase 4A: Explainable AI Module
│   ├── shap_explain.py               # SHAP TreeExplainer wrapper & stability evaluation
│   └── run_phase4_shap.py            # SHAP runner (2,000 stratified flows, global/class plots)
│
├── adversarial/                      # Phase 4B: Adversarial Robustness Module
│   ├── constraints.py                # 19-feature domain constraints, clipping, and bounds
│   └── run_phase4_adversarial.py     # FGSM transfer, HopSkipJump, ZOO, and defense retraining
│
├── alerts/                           # Phase 4C: Forensic Alert Logging Module
│   ├── alert_logger.py               # Flattened ELK-ready logger with SHA-256 hash chaining
│   ├── verify_log.py                 # Tamper-evident hash chain auditor
│   └── generate_and_test_alerts.py   # Test suite: populates logs/alerts.jsonl and verifies tamper test
│
├── models/
│   ├── load_validation.py            # Leak-free pure validation data loader (NO SMOTE)
│   ├── ensemble.py                   # RF + XGBoost + PyTorch DNN + soft/weighted voting
│   ├── wire_b2_models.py             # Model wiring utility for Method B2
│   ├── investigate_dnn.py            # Tabular neural network limit investigation
│   ├── ensemble_weights_8class.json
│   └── ensemble_weights_34class.json
│
├── orchestrator/
│   ├── __init__.py
│   └── llm_orchestrator.py           # Ollama client + deterministic expert fallback + SHAP injection
│
├── ingest/
│   ├── loader.py                     # Chunked CSV reader with per-class sampling
│   ├── validate.py                   # Validation report + class distribution chart
│   └── cowrie_loader.py              # Honeypot enrichment (future work)
│
├── preprocess/
│   ├── clean.py                      # Drop NaN / Inf / duplicates
│   ├── encode.py                     # LabelEncoder for 34-class and 8-class labels
│   ├── split.py                      # Stratified 80/20 split (before scaling)
│   ├── scale.py                      # StandardScaler (fit on train only)
│   ├── balance.py                    # SMOTE (applied to train only)
│   └── select_features.py            # RFE: 46 features → top 19
│
├── reports/                          # Generated charts and reports
│   ├── class_distribution.png
│   ├── rfe_ranking.png
│   ├── confusion_matrix_8class.png
│   ├── phase3_summary.md             # Comprehensive Phase 3 report
│   ├── phase4_summary.md             # Comprehensive Phase 4 report
│   ├── shap_global_importance.png    # Global SHAP feature importance plot
│   ├── shap_per_class_importance.png # Per-class SHAP importance heatmap
│   ├── shap_metrics.json
│   ├── adversarial_accuracy_vs_epsilon.png # Evasion curves: DNN vs Ensemble transfer
│   ├── adversarial_metrics.json
│   ├── llm_demo_outputs.csv          # Standard 40-flow triage outputs
│   └── llm_demo_outputs_shap.csv     # SHAP-grounded 40-flow triage outputs
│
├── logs/
│   └── alerts.jsonl                  # Cryptographically chained forensic alert log
│
└── artifacts/                        # Saved preprocessing objects and attributions
    ├── standard_scaler.pkl
    ├── label_encoder_8class.pkl
    ├── selected_features.json
    ├── shap_local_40flows.json       # Top-5 SHAP features per demo flow
    └── adversarial_demo_flows.parquet# Sample adversarial evaluation flows
```

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone -b phase4-xai-adversarial https://github.com/SurajSahoo2006/DYNAMIC-ATTACK-ADAPTION-IDS
cd DYNAMIC-ATTACK-ADAPTION-IDS
pip install -r requirements.txt
```
*Dependencies include `shap` (0.52.0) and `adversarial-robustness-toolbox` (1.20.1).*

### 2. Ollama Local LLM Setup

The LLM Semantic Orchestrator uses local [Ollama](https://ollama.com) running `llama3.2:3b`:
```bash
# Pull model
ollama pull llama3.2:3b

# Verify connection
curl http://localhost:11434/api/tags
```

---

## Running the Pipeline (Step-by-Step)

### Step 1: Preprocessing & Feature Selection (Phases 1 & 2)
```bash
python run_phase1_2.py --sample 5000
```

### Step 2: Train & Evaluate Ensemble Classifiers (Phase 3A)
```bash
python run_phase3_ensemble.py
```

### Step 3: Run the LLM Semantic Orchestrator (Phase 3B)
```bash
# Standard fixed telemetry run:
python run_phase3_llm.py

# Or with real top SHAP feature grounding:
python run_phase3_llm.py --use-shap
```

### Step 4: Run Phase 4A Explainable AI (SHAP Analysis)
```bash
python xai/run_phase4_shap.py
```
*Computes TreeExplainer SHAP values across 2,000 stratified test flows, evaluates XGBoost vs RF stability ($\rho = 0.9088$), generates global and per-class plots, and exports `artifacts/shap_local_40flows.json`.*

### Step 5: Run Phase 4B Adversarial Robustness Assessment
```bash
python adversarial/run_phase4_adversarial.py
```
*Enforces domain constraints (freezes protocols/flags, bounds packet sizes/timing), verifies zero-attack pipeline invariance (0.000000 drift), evaluates FGSM transfer vs isotropic random-noise controls (+8.6% to +15.7% attack advantage), runs multi-budget black-box HopSkipJump & ZOO, and tests a 4-arm defense matrix (+42.97% gain under FGSM, +20.00% under HopSkipJump).*

### Step 6: Generate & Audit Forensic Alert Logs (Phase 4C)
```bash
# Populate logs/alerts.jsonl and execute tamper verification tests:
python alerts/generate_and_test_alerts.py

# Audit an existing log file directly:
python alerts/verify_log.py --file logs/alerts.jsonl
```

---

## Phase Roadmap

| Phase | Component | Status | Empirical Highlights |
|:---:|:---|:---:|:---|
| **1** | Ingestion & Sampling | **COMPLETE** | Chunked reservoir sampling, validation reporting |
| **2** | Preprocessing & Feature Selection | **COMPLETE** | Clean, encode, 80/20 split, scale, SMOTE, RFE (46 → 19 features) |
| **3A** | Ensemble Classifier (RF + XGB + DNN) | **COMPLETE** | 8-class (95.53% acc, 87.66% F1); 34-class B2 (93.43% acc, 85.97% F1) |
| **3B** | LLM Semantic Orchestrator | **COMPLETE** | 40/40 real Ollama outputs (100%), 0 fallbacks, 12.0s/flow, peak 315 MB RAM |
| **4A** | Explainable AI (SHAP TreeExplainer) | **COMPLETE** | 2,000 stratified flows, $\rho = 0.9088$ stability, `--use-shap` LLM integration |
| **4B** | Adversarial Robustness (ART) | **COMPLETE** | Domain constraints, zero-attack control (0.0 drift), random noise control, FGSM transfer, multi-budget HSJ & ZOO, 4-arm defense matrix |
| **4C** | Cryptographic Alert Logging | **COMPLETE** | Flat ELK-ready schema, SHA-256 hash chaining, 100% verified tamper detection |
| **5** | MARL Tactical Defense (MAPPO) | UPCOMING | Multi-agent dynamic adaptation against adversarial evasion |
| **6** | SIEM Dashboard (ELK / Kibana) | UPCOMING | Ingestion of `logs/alerts.jsonl` into Elasticsearch & Kibana dashboards |
