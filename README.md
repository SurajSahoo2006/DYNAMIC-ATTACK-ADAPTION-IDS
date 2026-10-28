# Dynamic Attack Adaptation — IDS Pipeline
### B.Tech Sem IV Research Project | NMIMS Design Experience / IROP

> **Project**: AI/ML pipeline for adaptive network intrusion detection in IoT environments.  
> **Dataset**: CICIoT2023 (University of New Brunswick) — 47M+ records, 33 attack types.  
> **Phase scope of this branch**: Phase 1 (Data Ingestion) + Phase 2 (Preprocessing & Feature Selection).

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
                clean → encode → SPLIT → scale → SMOTE → RFE
                        |
              data/processed/*.parquet    artifacts/
                        |
                [Phase 3+: TODO]
            LLM Orchestrator → Ensemble Classifier → MARL → SHAP → ELK
```

---

## Repository Structure

```
.
├── config.py                    # All configuration (paths, knobs, label mapping)
├── requirements.txt
├── run_phase1_2.py              # ONE command to run everything
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
├── reports/                     # Generated charts and summary
│   ├── class_distribution.png
│   ├── rfe_ranking.png
│   └── summary.md               # Paste-ready numbers for slides
│
├── artifacts/                   # Saved sklearn objects
│   ├── standard_scaler.pkl
│   ├── label_encoder_34class.pkl
│   ├── label_encoder_8class.pkl
│   ├── selected_features.json
│   └── rfe_feature_ranking.csv
│
└── data/
    ├── raw/    ← git-ignored, put your CSVs here
    └── processed/  ← git-ignored, Parquet outputs
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

> **Windows tip**: If you see a Unicode error in your terminal, run:  
> `$env:PYTHONIOENCODING="utf-8"` in PowerShell before running the script.

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

## Phase 3+ Roadmap (Not Yet Implemented)

| Phase | Component | Status |
|-------|-----------|--------|
| 3 | Ensemble Classifier (RF + XGBoost + DNN) | TODO |
| 4 | LLM Semantic Orchestrator (Llama-3-8B) | TODO |
| 5 | MARL Tactical Agents (MAPPO) | TODO |
| 6 | Adversarial Robustness (FGSM, ZOO, HopSkipJump) | TODO |
| 7 | SHAP Explainability | TODO |
| 8 | Alert & Forensic Logging + ELK/Kibana Dashboard | TODO |
