# Diagnostic Report: Confusion Analysis of Rare Classes (34-Class Model)

> **Generated**: 2026-10-29  
> **Source**: Phase 3 Weighted Ensemble Classifier evaluated on untouched test split (`test_phase2.parquet`, 26,292 rows)  
> **Scope**: The 3 rare classes with < 100 test samples: `Backdoor_Malware` (95 rows), `Recon-PingSweep` (59 rows), and `Uploading_Attack` (34 rows).

---

## 1. Summary of Current Performance

| Class Name | True Test Support | Predictions (Total) | True Positives (TP) | False Negatives (FN) | False Positives (FP) | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Backdoor_Malware`** | 95 | 142 | 61 | 34 | 81 | **42.96%** | **64.21%** | **51.48%** |
| **`Recon-PingSweep`** | 59 | 46 | 28 | 31 | 18 | **60.87%** | **47.46%** | **53.33%** |
| **`Uploading_Attack`** | 34 | 28 | 16 | 18 | 12 | **57.14%** | **47.06%** | **51.61%** |

---

## 2. Row-Normalized Confusion Slices (Where Did True Samples Go?)

The tables below show what the ensemble predicted for each true instance of these rare attacks (Recall breakdown / False Negatives).

### A. `Backdoor_Malware` (95 True Samples)

| Predicted Class | Predicted Super-Class | Count | Percentage | Confusion Category |
|:---|:---|:---:|:---:|:---|
| **`Backdoor_Malware`** | **Web-based** | **61** | **64.21%** | **Correct (True Positive)** |
| `CommandInjection` | Web-based | 8 | 8.42% | Sibling class (Web-based) |
| `DNS_Spoofing` | Spoofing | 6 | 6.32% | Low-rate stealthy attack |
| `SqlInjection` | Web-based | 5 | 5.26% | Sibling class (Web-based) |
| `DictionaryBruteForce` | BruteForce | 4 | 4.21% | Low-rate authentication probe |
| `XSS` | Web-based | 3 | 3.16% | Sibling class (Web-based) |
| `BrowserHijacking` | Web-based | 2 | 2.11% | Sibling class (Web-based) |
| `Recon-HostDiscovery` | Recon | 2 | 2.11% | Low-rate probe |
| `Recon-PortScan` | Recon | 1 | 1.05% | Low-rate probe |
| `MITM-ArpSpoofing` | Spoofing | 1 | 1.05% | Low-rate probe |
| `Recon-OSScan` | Recon | 1 | 1.05% | Low-rate probe |
| `BenignTraffic` | Benign | 1 | 1.05% | False negative (Benign) |

> **Observation**: **18 of 34 misclassifications (52.9%)** are absorbed by its own sibling Web-based attack types (`CommandInjection`, `SqlInjection`, `XSS`, `BrowserHijacking`).

---

### B. `Recon-PingSweep` (59 True Samples)

| Predicted Class | Predicted Super-Class | Count | Percentage | Confusion Category |
|:---|:---|:---:|:---:|:---|
| **`Recon-PingSweep`** | **Recon** | **28** | **47.46%** | **Correct (True Positive)** |
| `Recon-OSScan` | Recon | 8 | 13.56% | Sibling class (Recon) |
| `DictionaryBruteForce` | BruteForce | 6 | 10.17% | Low-rate TCP/UDP scan |
| `DNS_Spoofing` | Spoofing | 4 | 6.78% | Low-rate network anomaly |
| `SqlInjection` | Web-based | 4 | 6.78% | Cross-family confusion |
| `Recon-PortScan` | Recon | 3 | 5.08% | Sibling class (Recon) |
| `Backdoor_Malware` | Web-based | 2 | 3.39% | Cross-family confusion |
| `BenignTraffic` | Benign | 2 | 3.39% | False negative (Benign) |
| `BrowserHijacking` | Web-based | 1 | 1.69% | Cross-family confusion |
| `XSS` | Web-based | 1 | 1.69% | Cross-family confusion |

> **Observation**: `Recon-PingSweep` is heavily confused with `Recon-OSScan` (13.6%) and `Recon-PortScan` (5.1%). Both involve sending lightweight probe packets across subnet hosts with minimal payload size, resulting in near-identical flow statistics in the top 19 features.

---

### C. `Uploading_Attack` (34 True Samples)

| Predicted Class | Predicted Super-Class | Count | Percentage | Confusion Category |
|:---|:---|:---:|:---:|:---|
| **`Uploading_Attack`** | **Web-based** | **16** | **47.06%** | **Correct (True Positive)** |
| `XSS` | Web-based | 6 | 17.65% | Sibling class (Web-based) |
| `CommandInjection` | Web-based | 4 | 11.76% | Sibling class (Web-based) |
| `Backdoor_Malware` | Web-based | 2 | 5.88% | Sibling class (Web-based) |
| `Recon-PortScan` | Recon | 1 | 2.94% | Cross-family confusion |
| `Recon-OSScan` | Recon | 1 | 2.94% | Cross-family confusion |
| `DNS_Spoofing` | Spoofing | 1 | 2.94% | Cross-family confusion |
| `BenignTraffic` | Benign | 1 | 2.94% | False negative (Benign) |
| `SqlInjection` | Web-based | 1 | 2.94% | Sibling class (Web-based) |
| `Recon-HostDiscovery` | Recon | 1 | 2.94% | Cross-family confusion |

> **Observation**: **13 out of 18 errors (72.2%)** are confused with its sibling Web-based attack types (`XSS`, `CommandInjection`, `Backdoor_Malware`, `SqlInjection`). `Uploading_Attack` consists of HTTP POST requests transmitting web shell code, which closely mirrors the packet sizes, header lengths, and inter-arrival times of HTTP GET/POST attacks like `XSS` and `CommandInjection`.

---

## 3. False Positive Influx (Where Did False Alarms Come From?)

Precision is also depressed by false positive intrusions from other classes:

1. **False Positives on `Backdoor_Malware` (81 rows)**:
   - 18 from `Recon-HostDiscovery`
   - 16 from `CommandInjection`
   - 12 from `DNS_Spoofing`
   - 11 from `XSS`
   - 6 from `SqlInjection`
   - 5 from `Recon-PortScan`
   - 4 from `Recon-OSScan`
   - *Key takeaway*: High-support sibling classes (`CommandInjection`, `XSS`, `SqlInjection`) bleed into `Backdoor_Malware` because their HTTP payload headers are similar, cutting precision to 42.96%.

2. **False Positives on `Recon-PingSweep` (18 rows)**:
   - 5 from `Recon-OSScan`, 4 from `BenignTraffic`, 3 from `MITM-ArpSpoofing`, 2 from `DNS_Spoofing`, 2 from `DictionaryBruteForce`, 1 from `CommandInjection`, 1 from `DDoS-SynonymousIP_Flood`.

3. **False Positives on `Uploading_Attack` (12 rows)**:
   - 6 from `Recon-PortScan`, 3 from `DNS_Spoofing`, 2 from `Recon-HostDiscovery`, 1 from `DictionaryBruteForce`.

---

## 4. Key Diagnostic Takeaways

1. **Intra-Family Feature Similarity**: The dominant source of error is **not** random guessing or confusion with high-volume attacks (DDoS/DoS/Mirai have 0% confusion with these three classes). Rather, errors are concentrated inside the **Web-based** and **Recon** sub-families.
2. **Feature Bottleneck**: The 19 statistical flow features (packet counts, durations, covariance, mean packet size) struggle to distinguish an HTTP POST uploading a file (`Uploading_Attack`) from an HTTP GET/POST carrying an injection payload (`CommandInjection` / `XSS`).
3. **Data Scarcity**: Because the model was trained on only 138 real instances of `Uploading_Attack`, 237 of `Recon-PingSweep`, and 378 of `Backdoor_Malware`, the decision boundaries around them are susceptible to intrusion by larger classes.
