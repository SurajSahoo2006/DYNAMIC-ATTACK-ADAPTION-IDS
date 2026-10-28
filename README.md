# Dynamic Attack Adaptation — IDS Pipeline
### B.Tech Sem IV Research Project | NMIMS Design Experience / IROP

> **Project**: AI/ML pipeline for adaptive network intrusion detection in IoT environments.  
> **Dataset**: CICIoT2023 (University of New Brunswick) — 47M+ records, 33 attack types.  
> **Branch**: `phase3-ensemble-llm`  
> **Current Pipeline Scope**: Phase 1 (Ingestion) + Phase 2 (Preprocessing & RFE) + Phase 3 (Ensemble Classifier & LLM Semantic Orchestrator).

---

## Architecture Overview

```
Network Traffic (PCAP)          Honeypot (Cowrie JSON)       Benchmark Datasets
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
            [Phase 3: Classification & Semantic Orchestration]
            ├── Ensemble Classifier (Random Forest + XGBoost + PyTorch DNN)
            └── LLM Semantic Orchestrator (Ollama Llama-3.2 / Expert Fallback)
                        |
            [Phase 4+: Future Phases]
            MARL Tactical Agents -> Adversarial Checks -> SHAP -> ELK / Kibana
```

---

## Repository Structure

```
.
├── config.py                    # All configuration (paths, knobs, label & severity mapping)
├── requirements.txt
├── run_phase1_2.py              # Phase 1 & 2 pipeline runner
├── run_phase3_ensemble.py       # Phase 3 ensemble training & evaluation runner
├── run_phase3_llm.py            # Phase 3 LLM semantic orchestrator demo runner
├── predict.py                   # Standalone flow prediction & severity module
│
├── models/
│   ├── load_validation.py       # Leak-free pure validation data loader (NO SMOTE)
│   ├── ensemble.py              # RF + XGBoost + PyTorch DNN + soft/weighted voting
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
│   └── cowrie_loader.py         # Honeypot enrichment (documented placeholder)
│
├── preprocess/
│   ├── clean.py                 # Drop NaN / Inf / duplicates
│   ├── encode.py                # LabelEncoder for 34-class and 8-class labels
│   ├── split.py                 # Stratified 80/20 split (before scaling!)
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
│   ├── llm_demo_outputs.csv     # 40-flow LLM triage demonstrations
│   └── llm_metrics.json
│
├── artifacts/                   # Saved sklearn & preprocessing objects
│   ├── standard_scaler.pkl
│   ├── label_encoder_34class.pkl
│   ├── label_encoder_8class.pkl
│   ├── selected_features.json
│   └── rfe_feature_ranking.csv
│
└── data/
    ├── raw/    ← git-ignored, put your CSVs here
    └── processed/  ← git-ignored, Parquet outputs (train, test, val)
```

---

## Quick Start

### 1. Clone & setup

```bash
git clone -b phase1-2-prototype https://github.com/SurajSahoo2006/DYNAMIC-ATTACK-ADAPTION-IDS
cd DYNAMIC-ATTACK-ADAPTION-IDS
pip install -r requirements.txt
```

### 2. Place the dataset

The raw CSVs go in a folder **outside** the repo (they are too large for git):

```
D:\IROP MODEL\CICIOT23\
    train\train.csv          (~1.5 GB)
    test\test.csv            (~330 MB)
    validation\validation.csv
```

If you downloaded them elsewhere, edit `DATA_RAW_DIR` in `config.py`.

**Where to download CICIoT2023:**  
Official link: https://www.unb.ca/cic/datasets/iotdataset-2023.html  
Direct download page: https://ciciot2023.unb.ca  
Files needed: `train.csv`, `test.csv` (and optionally `validation.csv`).

### 3. Run the pipeline

```bash
# On your laptop (fast — ~5 min)
python run_phase1_2.py --sample 5000

# Scale up on Colab (more accurate results)
python run_phase1_2.py --sample 50000

# Use ALL rows (needs ~32 GB RAM)
python run_phase1_2.py --full

# Skip SMOTE for a very quick smoke-test
python run_phase1_2.py --sample 2000 --no-smote
```

### 4. Run Phase 3 (Ensemble Classifier & LLM Semantic Orchestrator)

Phase 3 trains the multi-model ensemble (Random Forest + XGBoost + PyTorch DNN) and runs the LLM Orchestrator:

```bash
# 1. Train and evaluate the Ensemble Classifier (Primary 8-class & Secondary 34-class)
python run_phase3_ensemble.py

# Optional flags:
# python run_phase3_ensemble.py --skip-34            # Fast run: 8-class target only (~2.5 min)
# python run_phase3_ensemble.py --val-sample 2000    # Smaller validation sample cap

# 2. Run the LLM Semantic Orchestrator Demo (40 balanced test flows)
python run_phase3_llm.py

# 3. Test standalone flow inference with the predict interface
python predict.py --target 8class --samples 5
python predict.py --target 34class --samples 5
```

#### Optional: Local Ollama LLM Setup
By default, `run_phase3_llm.py` uses an internal deterministic rule-based expert triage engine when Ollama is offline. To run with live local Llama-3:
1. Install [Ollama](https://ollama.com).
2. Pull the model:
   ```bash
   ollama pull llama3.2:3b
   ```
3. Ensure Ollama is running (`http://localhost:11434`), then re-run `python run_phase3_llm.py`.

> **Windows tip**: If you see a Unicode error in your terminal, run:  
> `$env:PYTHONIOENCODING="utf-8"` in PowerShell before running scripts.

---

## Running on Google Colab

1. Upload the CSVs to Google Drive under `MyDrive/CICIoT2023/train/train.csv` etc.
2. Open `notebooks/phase1_2_walkthrough.ipynb` in Colab.
3. In **Cell 3 (Configuration)**, set `SAMPLE_PER_CLASS = 50000` for a larger run.
4. The notebook mounts Drive automatically and clones the repo.
5. Run all cells top to bottom.

---

## What Each Phase Produces

### Phase 1 Outputs
| Output | Description |
|--------|-------------|
| Console validation report | Row counts, NaN/Inf counts, duplicate counts |
| `reports/class_distribution.png` | Bar chart of 34 labels and 8 classes |
| Summary dict (returned from `run_validation_report`) | Saved into `summary.md` |

### Phase 2 Outputs
| Output | Description |
|--------|-------------|
| `data/processed/train_phase2.parquet` | SMOTE-balanced train set, 19 features |
| `data/processed/test_phase2.parquet` | Untouched test set, 19 features |
| `artifacts/standard_scaler.pkl` | Fitted StandardScaler |
| `artifacts/label_encoder_34class.pkl` | 34-class LabelEncoder |
| `artifacts/label_encoder_8class.pkl` | 8-class LabelEncoder |
| `artifacts/selected_features.json` | List of the 19 selected feature names |
| `artifacts/rfe_feature_ranking.csv` | All 46 features with RFE rank |
| `reports/rfe_ranking.png` | Feature importance bar chart |
| `reports/summary.md` | Full numbers, ready to paste into slides |

---

## Important Notes for Your Slides

### 1. Feature count correction (IMPORTANT)
Your current slides say **"115 → 19 features"** — this is from an older dataset.  
CICIoT2023 has **46 numeric features**. The correct statement is:  
> **"RFE reduces 46 features → top 19 features"**

### 2. Label mapping (verified from real data)
The 34 → 8 class mapping was built by scanning the actual CSV labels, not from literature.
Full mapping is in `config.py → LABEL_TO_CLASS` and in `reports/summary.md`.

### 3. Reproducibility
All random operations use `RANDOM_SEED = 42`. Results are fully reproducible.

### 4. Note on SMOTE and flag features (limitation)
Some features are 0/1 binary flags (such as TCP flags, HTTP/DNS protocol indicators) or protocol codes; standard SMOTE creates continuous in-between values for them during interpolation. Tree-based ensembles naturally partition such continuous representations, but specialized variants (e.g., SMOTE-NC or post-synthesis rounding) can be explored in future work.

---

## Data Leakage Prevention

The pipeline follows this strict order to prevent data leakage:

```
Raw data
   ↓ clean()          -- remove bad rows
   ↓ encode_labels()  -- create label_34, label_8 columns
   ↓ split()          ← SPLIT HAPPENS HERE (before any fitting)
   |
   ├── TRAIN set only:
   |      scale()  (fit + transform)
   |      balance() (SMOTE)
   |      select_features() (RFE fit)
   |
   └── TEST set only:
          scale()  (transform only, using train's scaler)
          [NO SMOTE on test]
          [NO RFE refit on test]
```

---

## Phase Roadmap

| Phase | Component | Status | Details |
|:---:|:---|:---:|:---|
| **1** | Ingestion & Sampling (CICIoT2023) | **COMPLETE** | Chunked reservoir sampling, validation |
| **2** | Preprocessing & Feature Selection | **COMPLETE** | Clean, encode, 80/20 split, scale, SMOTE, RFE (19 features) |
| **3A** | Ensemble Classifier (RF + XGBoost + DNN) | **COMPLETE** | 8-class (95.53% acc) & 34-class (93.19% acc), zero leak validation |
| **3B** | LLM Semantic Orchestrator | **COMPLETE** | Flow triage, explanations, mitigations, Ollama + rule fallback |
| **4** | MARL Tactical Agents (MAPPO) | UPCOMING | Multi-agent dynamic adaptation |
| **5** | Adversarial Robustness Check | UPCOMING | FGSM, ZOO, HopSkipJump evasion testing |
| **6** | SHAP Explainability Layer | UPCOMING | Feature contribution explainability |
| **7** | Alert & Forensic Logging (ELK/Kibana) | UPCOMING | Forensic log trail, SIEM dashboard |

