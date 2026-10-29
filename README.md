# Dynamic Attack Adaptation — IDS Pipeline
### B.Tech Sem IV Research Project | NMIMS Design Experience / IROP

> **Project**: AI/ML pipeline for adaptive network intrusion detection in IoT environments.  
> **Dataset**: CICIoT2023 (University of New Brunswick) — 47M+ records, 33 attack types.  
> **Branch**: `phase3-ensemble-llm`  
> **Current Pipeline Scope**: Phase 1 (Ingestion) + Phase 2 (Preprocessing & RFE) + Phase 3 (Ensemble Classifier & LLM Semantic Orchestrator).

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
            [Phase 3A: Ensemble Classifier]
            ├── Random Forest (Balanced) + XGBoost (Sample-Weighted) + PyTorch DNN
            │   (Optimal validation weights: w_RF=0.60, w_XGB=0.40, w_DNN=0.00)
            │
            ▼ (Downstream predictions & telemetry)
            [Phase 3B: LLM Semantic Orchestrator]
            └── Ollama Llama-3.2:3b (CPU-only, temp=0) / Deterministic Expert Fallback
                (Explanations + Recommended Mitigations; never overrides classification)
                        |
            [Phase 4+: Future Phases]
            MARL Tactical Agents -> Adversarial Checks -> SHAP -> ELK / Kibana

* Note: Honeypot log enrichment is future work; current pipeline consumes statistical flow vectors.
```

---

## Repository Structure

```
.
├── config.py                    # Central configuration (paths, hyperparameters, mappings)
├── requirements.txt
├── run_phase1_2.py              # Phase 1 & 2 pipeline runner
├── run_phase3_ensemble.py       # Phase 3 ensemble training & evaluation runner
├── run_phase3_llm.py            # Phase 3 LLM semantic orchestrator runner (Ollama / Fallback)
├── predict.py                   # Standalone flow prediction & severity module (B2 default)
│
├── models/
│   ├── load_validation.py       # Leak-free pure validation data loader (NO SMOTE)
│   ├── ensemble.py              # RF + XGBoost + PyTorch DNN + soft/weighted voting
│   ├── wire_b2_models.py        # Model wiring utility for Method B2
│   ├── investigate_dnn.py       # Step 5 DNN outlier & loss investigation script
│   ├── ensemble_weights_8class.json
│   └── ensemble_weights_34class.json
│
├── orchestrator/
│   ├── __init__.py
│   └── llm_orchestrator.py      # Ollama client + deterministic expert fallback
│
├── ingest/
│   ├── loader.py                # Chunked CSV reader with per-class sampling
│   ├── validate.py              # Validation report + class distribution chart
│   ├── pcap_demo.py             # DEMO ONLY: raw PCAP feature extraction
│   └── cowrie_loader.py         # Honeypot enrichment (documented placeholder / future work)
│
├── preprocess/
│   ├── clean.py                 # Drop NaN / Inf / duplicates
│   ├── encode.py                # LabelEncoder for 34-class and 8-class labels
│   ├── split.py                 # Stratified 80/20 split (before scaling)
│   ├── scale.py                 # StandardScaler (fit on train only)
│   ├── balance.py               # SMOTE (applied to train only)
│   └── select_features.py       # RFE: 46 features → top 19
│
├── notebooks/
│   └── phase1_2_walkthrough.ipynb   # Step-by-step notebook (Colab-ready)
│
├── reports/                     # Generated charts and reports
│   ├── class_distribution.png
│   ├── rfe_ranking.png
│   ├── summary.md               # Phase 1 & 2 summary
│   ├── confusion_matrix_8class.png  # Phase 3 8-class test confusion matrix
│   ├── phase3_metrics.json      # Machine-readable test metrics
│   ├── phase3_summary.md        # Comprehensive Phase 3 report
│   ├── rare_class_confusions.md # Rare class confusion slice diagnosis
│   ├── rare_class_experiments.md# 5-experiment study on Backdoor, PingSweep, Uploading
│   ├── llm_demo_outputs.csv     # 40-flow real Ollama triage outputs
│   └── llm_metrics.json         # Real Ollama performance telemetry
│
├── artifacts/                   # Saved sklearn & preprocessing objects
│   ├── standard_scaler.pkl
│   ├── label_encoder_34class.pkl
│   ├── label_encoder_8class.pkl
│   ├── selected_features.json
│   └── rfe_feature_ranking.csv
│
└── data/
    ├── raw/                     # Git-ignored, put raw CSVs here
    └── processed/               # Git-ignored, Parquet outputs (train, test, val)
```

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone -b phase3-ensemble-llm https://github.com/SurajSahoo2006/DYNAMIC-ATTACK-ADAPTION-IDS
cd DYNAMIC-ATTACK-ADAPTION-IDS
pip install -r requirements.txt
```

### 2. Place Dataset Files

The raw CSVs belong in a directory outside the repository:

```
D:\IROP MODEL\CICIOT23\
    train\train.csv          (~1.5 GB)
    test\test.csv            (~330 MB)
    validation\validation.csv (~330 MB)
```

If stored elsewhere, adjust `DATA_RAW_DIR` in `config.py`.

* **CICIoT2023 Official Download**: https://www.unb.ca/cic/datasets/iotdataset-2023.html

---

## Ollama Local LLM Setup

The LLM Semantic Orchestrator uses local [Ollama](https://ollama.com) running `llama3.2:3b` on CPU:

1. **Install Ollama**: Download and run the installer from [ollama.com](https://ollama.com).
2. **Pull the Model**:
   ```bash
   ollama pull llama3.2:3b
   ```
3. **Verify the Service**:
   Check that Ollama is listening locally on port 11434:
   ```bash
   # On PowerShell:
   Invoke-RestMethod http://localhost:11434/api/tags
   # Or in bash / cmd:
   curl http://localhost:11434/api/tags
   ```
   Ensure `llama3.2:3b` appears in the returned models list.

*(Note: If Ollama is offline or unreachable, the orchestrator automatically activates its deterministic rule-based expert fallback.)*

---

## Running the Pipeline (Step-by-Step)

Execute the components sequentially:

### Step 1: Preprocessing & Feature Selection (Phases 1 & 2)
```bash
python run_phase1_2.py --sample 5000
```
*Loads chunked raw data, cleans invalid values, performs a strict train/test split, fits `StandardScaler`, applies SMOTE to the training split, and runs RFE to identify the top 19 features.*

### Step 2: Train & Evaluate Ensemble Classifiers (Phase 3A)
```bash
python run_phase3_ensemble.py
```
*Trains Random Forest, XGBoost, and PyTorch DNN for 8-class and 34-class targets. Computes optimal simplex voting weights against the leak-free `validation.csv` set, and evaluates performance on the untouched `test.csv` split.*

### Step 3: Run the LLM Semantic Orchestrator (Phase 3B)
```bash
python run_phase3_llm.py
```
*Randomly samples 40 test flows with fixed seed 42 (5 flows per 8-class family, guaranteeing representation of rare classes `Backdoor_Malware`, `Recon-PingSweep`, and `Uploading_Attack`). Runs local Ollama (`llama3.2:3b`) at `temperature: 0` to produce concise natural language explanations and mitigation playbooks, recording execution telemetry to `reports/llm_metrics.json`.*

### Step 4: Run Standalone Predictions
```bash
# Predict on 5 test samples using the 8-class model:
python predict.py --target 8class --samples 5

# Predict on 5 test samples using the 34-class B2 model:
python predict.py --target 34class --samples 5
```

---

## Assumptions & Methodological Invariants

1. **Dataset Nature & Scope**: Network flows in the CICIoT2023 benchmark are represented purely as aggregated statistical feature vectors derived from 5-second sliding windows. Application-layer payload strings, full PCAP files, and raw textual logs are not part of the dataset.
2. **Zero-Leakage Validation Split**: `validation.csv` (106,766 real flows) is strictly isolated from model training. No synthetic SMOTE interpolation is ever applied to validation or test splits. All hyperparameter decisions, early stopping, and ensemble weighting simplex optimizations are tuned on validation data only. Test data is evaluated once.
3. **Feature Dimensionality (46 → 19)**: Exactly 19 numerical features selected via Recursive Feature Elimination (RFE) on the training split are used. The slide claim "115 → 19" is corrected to "46 → 19" based on the actual CSV column schema.
4. **Ensemble Architecture & DNN Weighting**: The ensemble combines Random Forest, XGBoost, and a PyTorch Deep Neural Network. Because tree-based models naturally partition heavy-tailed tabular distributions (where standardized flow features reach $z = 89.75$), validation simplex optimization assigned the DNN a weight of 0.00. The active prediction ensemble is effectively **Random Forest + XGBoost**.
5. **Adoption of Method B2 (Balanced Weights, No SMOTE)**: Following a 5-experiment study on rare classes, Method B2 was adopted as the production classifier. It achieves 93.43% test accuracy and 85.97% Macro F1. It raised `Backdoor_Malware` (51.5% -> 57.0%) and `Recon-PingSweep` (53.3% -> 56.3%), but lowered `Uploading_Attack` F1 (51.6% -> 46.0%).
6. **Rare Class Support & Statistical Limits**: True test support for the three rarest classes is minimal (34 flows for `Uploading_Attack`, 59 for `Recon-PingSweep`, 95 for `Backdoor_Malware`). A swing of 1–2 misclassified flows changes recall by 3–6%, so score differences between candidate models may be sample noise. Reaching >85% F1 was not achieved on this data and is likely limited by the scarcity of real rows (only 208 total rows exist across the entire 7.84M benchmark for `Uploading_Attack`).
7. **Downstream LLM Orchestration Invariant**: The LLM sits strictly downstream **AFTER** the ensemble classifier. The ensemble determines the predicted attack class, and a deterministic configuration table determines severity. The LLM produces natural language operator explanations and mitigation recommendations using unscaled features, without overriding ensemble classifications.
8. **Honeypot Enrichment as Future Work**: Cowrie honeypot log integration is recognized as essential future work to resolve intra-family HTTP attack ambiguity, as the current loader ingests purely tabular network headers.
9. **Hardware Constraints**: All components are engineered and verified to run on a local CPU-only laptop with 8 GB RAM, strictly enforcing a peak RAM ceiling below 6.0 GB (measured peak: 1.37 GB for ensemble training, 0.55 GB for LLM inference).

---

## Phase Roadmap

| Phase | Component | Status | Empirical Highlights |
|:---:|:---|:---:|:---|
| **1** | Ingestion & Sampling (CICIoT2023) | **COMPLETE** | Chunked reservoir sampling, validation reporting |
| **2** | Preprocessing & Feature Selection | **COMPLETE** | Clean, encode, 80/20 split, scale, SMOTE, RFE (46 → 19 features) |
| **3A** | Ensemble Classifier (RF + XGB + DNN) | **COMPLETE** | 8-class (95.53% acc, 87.66% F1); 34-class B2 (93.43% acc, 85.97% F1) |
| **3B** | LLM Semantic Orchestrator | **COMPLETE** | 40/40 real Ollama outputs (100%), 0 fallbacks, 12.0s/flow, peak 564.9 MB RAM |
| **4** | MARL Tactical Agents (MAPPO) | UPCOMING | Multi-agent dynamic defense adaptation |
| **5** | Adversarial Robustness Check | UPCOMING | FGSM, ZOO, HopSkipJump evasion testing |
| **6** | SHAP Explainability Layer | UPCOMING | Global and local feature attribution |
| **7** | Alert & Forensic Logging (ELK/Kibana) | UPCOMING | SIEM dashboard and incident response trail |
