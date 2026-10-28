# Phase 1 & 2 - Pipeline Summary

> Generated: 2026-10-29 00:43:09  
> Runtime: 244.8 s  
> Random seed: 42

---

## Phase 1 - Data Ingestion

### Dataset: CICIoT2023
- Source files: `train.csv` + `test.csv` (raw CSVs in `data/raw/`)
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
- **Train rows**: 340,000
- **Test rows** : 26,292
- [OK] Zero index overlap confirmed (no data leakage)

### Step 4 - StandardScaler
- Fit on **train set only**
- Applied to both train and test
- Scaler saved to `artifacts/standard_scaler.pkl`

### Step 5 - SMOTE (class balancing)
- Applied to **train set only** (test set unchanged)
- Strategy: oversample minority classes up to 10,000 rows each
- Train rows after SMOTE: **340,000**

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
| 4 | `Srate` |
| 5 | `fin_flag_number` |
| 6 | `ack_count` |
| 7 | `syn_count` |
| 8 | `syn_flag_number` |
| 9 | `rst_count` |
| 10 | `ICMP` |
| 11 | `urg_count` |
| 12 | `IAT` |
| 13 | `AVG` |
| 14 | `Max` |
| 15 | `Min` |
| 16 | `Tot sum` |
| 17 | `Magnitue` |
| 18 | `Variance` |
| 19 | `Tot size` |

See `reports/rfe_ranking.png` for the full ranking chart.

### Step 7 - Final Output
| File | Shape | Location |
|------|-------|----------|
| train_phase2.parquet | (340000, 21) | `data/processed/` |
| test_phase2.parquet  | (26292, 21)  | `data/processed/` |

Columns in Parquet files: `[19 features] + label_34 + label_8`

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