# Phase 1 & 2 - Pipeline Summary

> Generated: 2026-10-29 01:16:40  
> Runtime: 115.4 s  
> Peak memory (RSS): 298.9 MB (0.29 GB)  
> Random seed: 42

---

## Phase 1 - Data Ingestion

### Dataset: CICIoT2023
- Source files: `train.csv` + `test.csv` (raw CSVs located at `D:\IROP MODEL\CICIOT23`)
- Columns in raw CSV: **47** (46 numeric features + 1 label column)
- Sample per fine-grained class: **5000**

### Validation Report
| Metric | Value |
|--------|-------|
| Raw rows loaded | 133,068 |
| Columns | 47 |
| Total NaN cells | 0 |
| Total Inf cells | 0 |
| Exact duplicate rows | 1,611 |
| Fine-grained labels | 34 |

### 34-Label --> 8-Class Mapping
(Read directly from the CSV -- NOT taken from literature)

| Fine-grained Label | 8-Class |
|-------------------|---------|
| `Backdoor_Malware` | Web-based |
| `BenignTraffic` | Benign |
| `BrowserHijacking` | Web-based |
| `CommandInjection` | Web-based |
| `DDoS-ACK_Fragmentation` | DDoS |
| `DDoS-HTTP_Flood` | DDoS |
| `DDoS-ICMP_Flood` | DDoS |
| `DDoS-ICMP_Fragmentation` | DDoS |
| `DDoS-PSHACK_Flood` | DDoS |
| `DDoS-RSTFINFlood` | DDoS |
| `DDoS-SYN_Flood` | DDoS |
| `DDoS-SlowLoris` | DDoS |
| `DDoS-SynonymousIP_Flood` | DDoS |
| `DDoS-TCP_Flood` | DDoS |
| `DDoS-UDP_Flood` | DDoS |
| `DDoS-UDP_Fragmentation` | DDoS |
| `DNS_Spoofing` | Spoofing |
| `DictionaryBruteForce` | BruteForce |
| `DoS-HTTP_Flood` | DoS |
| `DoS-SYN_Flood` | DoS |
| `DoS-TCP_Flood` | DoS |
| `DoS-UDP_Flood` | DoS |
| `MITM-ArpSpoofing` | Spoofing |
| `Mirai-greeth_flood` | Mirai |
| `Mirai-greip_flood` | Mirai |
| `Mirai-udpplain` | Mirai |
| `Recon-HostDiscovery` | Recon |
| `Recon-OSScan` | Recon |
| `Recon-PingSweep` | Recon |
| `Recon-PortScan` | Recon |
| `SqlInjection` | Web-based |
| `Uploading_Attack` | Web-based |
| `VulnerabilityScan` | Recon |
| `XSS` | Web-based |

### Class Distribution (before preprocessing)
See `reports/class_distribution.png`

| 8-Class | Raw count |
|---------|-----------|
| DDoS | 57,459 |
| Recon | 20,302 |
| DoS | 20,000 |
| Mirai | 15,000 |
| Spoofing | 10,000 |
| Benign | 5,000 |
| Web-based | 3,447 |
| BruteForce | 1,860 |

> **Imbalance observation**: Some attack classes have far fewer samples
> than others. SMOTE corrects this for training (see Phase 2 below).

### Class Support Breakdown (Real Rows: Train vs. Test)
> **Low Support Warning**: Attack classes with fewer than 100 test rows are flagged.
> Their performance metrics must be interpreted with caution due to low statistical support.

| # | Fine-Grained Class | 8-Class Group | Train (Real) | Test (Real) | Total Real | Support Note |
|:---|:---|:---|:---:|:---:|:---:|:---|
| 1 | `Backdoor_Malware` | Web-based | 378 | 95 | 473 | ⚠️ **low support, interpret with caution** |
| 2 | `BenignTraffic` | Benign | 3,998 | 1,000 | 4,998 | Normal |
| 3 | `BrowserHijacking` | Web-based | 619 | 155 | 774 | Normal |
| 4 | `CommandInjection` | Web-based | 578 | 144 | 722 | Normal |
| 5 | `DDoS-ACK_Fragmentation` | DDoS | 3,979 | 995 | 4,974 | Normal |
| 6 | `DDoS-HTTP_Flood` | DDoS | 3,178 | 794 | 3,972 | Normal |
| 7 | `DDoS-ICMP_Flood` | DDoS | 3,891 | 973 | 4,864 | Normal |
| 8 | `DDoS-ICMP_Fragmentation` | DDoS | 3,982 | 996 | 4,978 | Normal |
| 9 | `DDoS-PSHACK_Flood` | DDoS | 3,934 | 983 | 4,917 | Normal |
| 10 | `DDoS-RSTFINFlood` | DDoS | 3,877 | 969 | 4,846 | Normal |
| 11 | `DDoS-SYN_Flood` | DDoS | 3,973 | 993 | 4,966 | Normal |
| 12 | `DDoS-SlowLoris` | DDoS | 2,630 | 657 | 3,287 | Normal |
| 13 | `DDoS-SynonymousIP_Flood` | DDoS | 3,966 | 992 | 4,958 | Normal |
| 14 | `DDoS-TCP_Flood` | DDoS | 3,958 | 990 | 4,948 | Normal |
| 15 | `DDoS-UDP_Flood` | DDoS | 3,998 | 1,000 | 4,998 | Normal |
| 16 | `DDoS-UDP_Fragmentation` | DDoS | 3,982 | 996 | 4,978 | Normal |
| 17 | `DNS_Spoofing` | Spoofing | 3,978 | 994 | 4,972 | Normal |
| 18 | `DictionaryBruteForce` | BruteForce | 1,430 | 358 | 1,788 | Normal |
| 19 | `DoS-HTTP_Flood` | DoS | 3,939 | 985 | 4,924 | Normal |
| 20 | `DoS-SYN_Flood` | DoS | 3,985 | 996 | 4,981 | Normal |
| 21 | `DoS-TCP_Flood` | DoS | 3,963 | 991 | 4,954 | Normal |
| 22 | `DoS-UDP_Flood` | DoS | 3,994 | 999 | 4,993 | Normal |
| 23 | `MITM-ArpSpoofing` | Spoofing | 3,986 | 997 | 4,983 | Normal |
| 24 | `Mirai-greeth_flood` | Mirai | 3,920 | 980 | 4,900 | Normal |
| 25 | `Mirai-greip_flood` | Mirai | 3,927 | 982 | 4,909 | Normal |
| 26 | `Mirai-udpplain` | Mirai | 3,994 | 998 | 4,992 | Normal |
| 27 | `Recon-HostDiscovery` | Recon | 3,965 | 991 | 4,956 | Normal |
| 28 | `Recon-OSScan` | Recon | 3,962 | 990 | 4,952 | Normal |
| 29 | `Recon-PingSweep` | Recon | 237 | 59 | 296 | ⚠️ **low support, interpret with caution** |
| 30 | `Recon-PortScan` | Recon | 3,955 | 989 | 4,944 | Normal |
| 31 | `SqlInjection` | Web-based | 574 | 143 | 717 | Normal |
| 32 | `Uploading_Attack` | Web-based | 138 | 34 | 172 | ⚠️ **low support, interpret with caution** |
| 33 | `VulnerabilityScan` | Recon | 3,894 | 973 | 4,867 | Normal |
| 34 | `XSS` | Web-based | 403 | 101 | 504 | Normal |

> **Identified 3 low-support classes (<100 test rows)**: `Backdoor_Malware` (95 rows), `Recon-PingSweep` (59 rows), `Uploading_Attack` (34 rows)

---

## Phase 2 - Preprocessing

### Step 1 - Cleaning
| Step | Rows dropped |
|------|-------------|
| NaN rows | 0 |
| Inf rows | 0 |
| Duplicate rows | 1,611 |
| **Rows after cleaning** | **131,457** |

### Step 2 - Label Encoding
- Two label columns created: `label_34` (int 0-33) and `label_8` (int 0-7)
- Encoders saved to `artifacts/`

### Step 3 - Train / Test Split
- **Ratio**: 80/20  (stratified on `label_34`)
- **Train rows before SMOTE (real)**: 105,165
- **Train rows after SMOTE (real + synthetic)**: 132,061
- **Test rows (never resampled)**: 26,292
- [OK] Zero index overlap confirmed (no data leakage)

### Step 4 - StandardScaler
- Fit on **train set only**
- Applied to both train and test
- Scaler saved to `artifacts/standard_scaler.pkl`

### Step 5 - SMOTE (class balancing)
- Applied to **train set only** (test set unchanged)
- Target per class: **3,998** (size of largest train class, bounded by SAMPLE_PER_CLASS)
- Safety cap: maximum **10x** synthetic rows per class
- Real train rows: **105,165**
- Synthetic train rows created: **26,896** (20.4% synthetic)
- Train rows after SMOTE: **132,061**
- Note on flag features: Some features are 0/1 flags or protocol codes; SMOTE creates in-between values for them.

### Step 6 - Feature Selection (RFE)

> WARNING **SLIDE CORRECTION**: Your presentation states '115 --> 19 features'.
> CICIoT2023 has **46** numeric features (not 115).
> Please update your slides to say **'46 --> 19 features'**.

- Method: Recursive Feature Elimination with RandomForest estimator
- Input features: **46**
- Selected features: **19**
- RFE subsample used: 50,000 rows

#### Top 19 Selected Features
| # | Feature |
|---|---------|
| 1 | `flow_duration` |
| 2 | `Header_Length` |
| 3 | `Protocol Type` |
| 4 | `fin_flag_number` |
| 5 | `syn_count` |
| 6 | `ack_count` |
| 7 | `syn_flag_number` |
| 8 | `urg_count` |
| 9 | `rst_count` |
| 10 | `ICMP` |
| 11 | `Tot size` |
| 12 | `IAT` |
| 13 | `AVG` |
| 14 | `Max` |
| 15 | `Min` |
| 16 | `Tot sum` |
| 17 | `Magnitue` |
| 18 | `Covariance` |
| 19 | `Variance` |

See `reports/rfe_ranking.png` for the full ranking chart.

### Step 7 - Final Output
| File | Shape | Location |
|------|-------|----------|
| train_phase2.parquet | (132061, 21) | `data/processed/` |
| test_phase2.parquet  | (26292, 21)  | `data/processed/` |

Columns in Parquet files: `[19 features] + label_34 + label_8`

---

## Smoke test (not the final result)
> **Notice**: This baseline Random Forest test (100 trees) on the 19 selected features confirms
> preprocessing data integrity. The final detection engine (Phase 3: Random Forest + XGBoost + DNN
> ensemble) will be trained on the full dataset with proper tuning.

- **8-Class Smoke-Test Accuracy**: **95.21%**
- **34-Class Smoke-Test Accuracy**: **92.56%**

### 8-Class Classification Report
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Benign | 0.8446 | 0.7720 | 0.8067 | 1,000 |
| DDoS | 0.9986 | 0.9991 | 0.9989 | 11,338 |
| DoS | 0.9987 | 0.9985 | 0.9986 | 3,971 |
| Recon | 0.9004 | 0.8963 | 0.8983 | 4,002 |
| Web-based | 0.5302 | 0.8363 | 0.6490 | 672 |
| BruteForce | 0.8034 | 0.6620 | 0.7259 | 358 |
| Spoofing | 0.9188 | 0.8182 | 0.8656 | 1,991 |
| Mirai | 1.0000 | 0.9973 | 0.9986 | 2,960 |
| **Overall Accuracy** | | | **0.9521** | 26,292 |
| **Macro Avg** | 0.8743 | 0.8725 | 0.8677 | 26,292 |
| **Weighted Avg** | 0.9573 | 0.9521 | 0.9534 | 26,292 |

---

## Handoff to Phase 3
The following artifacts are ready for Phase 3 (Classifier):
- `data/processed/train_phase2.parquet`  - SMOTE-balanced train set
- `data/processed/test_phase2.parquet`   - untouched test set
- `artifacts/standard_scaler.pkl`        - for scaling live traffic
- `artifacts/label_encoder_34class.pkl`  - decode predictions
- `artifacts/label_encoder_8class.pkl`   - decode predictions
- `artifacts/selected_features.json`     - the 19 feature names

---

## Assumptions & Notes
1. Only `train.csv` and `test.csv` were loaded; `validation.csv` is reserved.
2. RFE used a subsample for speed -- results are deterministic given the seed.
3. Cowrie honeypot module is a documented placeholder (no log file provided).
4. PCAP demo (`ingest/pcap_demo.py`) is standalone; run manually if needed.
5. Benchmark datasets (UNSW-NB15, NSL-KDD, Bot-IoT) are Phase 3+ stubs.
6. '115 --> 19' in slides should be corrected to '**46 --> 19**' for CICIoT2023.
7. Note on SMOTE and flag features: Some features are 0/1 flags or protocol codes; SMOTE creates in-between values for them.