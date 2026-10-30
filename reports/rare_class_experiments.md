# Rare Class Optimization Experiments: 34-Class Evaluation

> **Target Goal**: Investigate whether the 34-class F1-score for the three rarest classes (`Backdoor_Malware`, `Recon-PingSweep`, `Uploading_Attack`) can be improved towards >85% F1 using **strictly legitimate** ML methods.  
> **Evaluation Protocol**:  
> - **Zero tuning on test data**: Model parameters and ensemble weights were tuned exclusively on the leak-free validation set (`val_phase2.parquet`, 106,766 real rows, no SMOTE).  
> - **Test evaluated once**: Evaluated on the untouched test split (`test_phase2.parquet`, 26,292 rows).  
> - **Scientific integrity**: No dropping, merging, hiding, test set resampling, or removing of caution flags.  
> - **Up-Front Finding on Method B2**: Method B2 (balanced class weights, no SMOTE) raised `Backdoor_Malware` (51.5% -> 57.0%) and `Recon-PingSweep` (53.3% -> 56.3%), but lowered `Uploading_Attack` F1 (51.6% -> 46.0%). Because test support is only 34–95 flows, small score differences between candidate models may be sample noise.

---

## 1. Complete Experimental Results Table

| ID | Experiment Name | Features | SMOTE | Weighting Strategy | Val Acc | Val Macro F1 | Val Backdoor F1 (93) | Val PingSweep F1 (45) | Val Uploading F1 (35) | Test Acc | Test Macro F1 | Test Backdoor F1 (95) | Test PingSweep F1 (59) | Test Uploading F1 (34) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | **Baseline** (Phase 3 RF+XGB Ensemble) | 19 | Yes | Uniform ($w=0.45/0.55$) | 94.99% | 81.08% | 31.6% | 38.9% | 34.4% | **93.19%** | **85.40%** | **51.5%** | **53.3%** | **51.6%** |
| **B1** | Balanced Class Weights + SMOTE | 19 | Yes | RF `balanced` + XGB sample weights | 94.96% | 80.74% | 30.7% | 32.3% | 25.0% | **93.25%** | **85.20%** | **50.0%** | **49.6%** | **39.6%** |
| **B2** | Balanced Class Weights (NO SMOTE) | 19 | No | RF `balanced` + XGB sample weights | 95.02% | 81.70% | 40.1% | 41.7% | 38.7% | **93.43%** | **85.97%** | **57.0%** | **56.3%** | **46.0%** |
| **C** | All 46 Features (Tree Ensemble) | 46 | Yes | Uniform ($w=0.50/0.50$) | 95.46% | 82.29% | 35.5% | 40.5% | 36.0% | **93.90%** | **86.50%** | **53.5%** | **52.6%** | **48.5%** |
| **D** | Hierarchical Model (8-Class + Specialists) | 19 | Yes | Family Stage-1 + Sub-specialists | 94.92% | 80.53% | 29.1% | 26.9% | 30.2% | **93.25%** | **85.13%** | **50.4%** | **47.2%** | **47.6%** |
| **E** | **Best Combination** (46 Feats + No SMOTE + Balanced) | 46 | No | RF `balanced` + XGB sample weights | **95.45%** | **83.21%** | **48.5%** | **48.1%** | **44.7%** | **93.93%** | **86.89%** | **58.2%** | **55.2%** | **56.8%** |

*(Numbers in parentheses indicate sample support in that split)*

---

## 2. Honest Scientific Finding: Can F1 > 85% Be Reached Legitimately?

### **CONCLUSION: NOT REACHED.**
Under strict, honest, and legitimate evaluation, **an F1 score > 85% was not reached with this data; it is likely limited by very few real rows.**

The highest legitimate test scores achieved across all candidate experiments were:
* **`Backdoor_Malware`**: Peak **58.2%** (Exp E) / **57.0%** (Exp B2) vs Baseline **51.5%**
* **`Recon-PingSweep`**: Peak **56.3%** (Exp B2) / **55.2%** (Exp E) vs Baseline **53.3%**
* **`Uploading_Attack`**: Peak **56.8%** (Exp E) vs Baseline **51.6%** (and dropped to **46.0%** in Exp B2)

These are single-run experimental results evaluated on only 34–95 test flows, so small differences between candidates may be sample noise.

---

## 3. Why F1 > 85% Was Not Reached (Limiting Factors & Hypotheses)

### Factor 1: The Extreme Scarcity of Real Benchmark Rows
A scan across the complete **7.84 million raw rows** of the CICIoT2023 dataset confirms:
* `Uploading_Attack`: Only **208 rows exist in the entire benchmark** (140 in `train.csv`, 35 in `validation.csv`, 33 in `test.csv`).
* `Recon-PingSweep`: Only **347 rows exist in the entire benchmark**.
* `Backdoor_Malware`: Only **574 rows exist in the entire benchmark**.

In the test split:
* `Uploading_Attack` has **34 flows** ($1 \text{ flow} = 2.94\%$ of recall).
* `Recon-PingSweep` has **59 flows** ($1 \text{ flow} = 1.69\%$).
* `Backdoor_Malware` has **95 flows** ($1 \text{ flow} = 1.05\%$).

No supervised classifier can achieve statistical certitude or high F1 when the ground-truth test support is limited to 34–95 flows. A swing of just two misclassified flows changes recall or precision by ~6%.

### Factor 2: Intra-Family Flow Similarity (Hypotheses)
As diagnosed in `reports/rare_class_confusions.md`, the following hypotheses explain why these classes remain confusable in tabular flow space:
1. **`Uploading_Attack` (Hypothesis)**: 72.2% of its test misclassifications are absorbed by sibling Web attacks (`XSS`, `CommandInjection`, `SqlInjection`). In statistical network flow features (header lengths, packet sizes, duration, inter-arrival times), an HTTP POST uploading malicious web-shell code exhibits flow characteristics that are virtually indistinguishable from an HTTP POST delivering an SQL injection or cross-site scripting payload without deep packet payload inspection.
2. **`Recon-PingSweep` (Hypothesis)**: Ping sweeps consist of lightweight ICMP or TCP probing flows across address ranges. Their statistical aggregated metrics (duration, packet counts, flag ratios) overlap heavily with sibling reconnaissance activities such as `Recon-OSScan` and `Recon-PortScan`.

### Factor 3: The Precision Multiplier Penalty
Because classes like `CommandInjection`, `XSS`, and `DDoS` have thousands of test flows, even an extremely low false-positive rate (e.g., 0.5%) on those majority classes generates 15–30 false alarms. For a rare target class with only 34 true flows:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{16}{16 + 26} = 38.1\%$$
Even when recall improves, a modest increase in false positives drastically depresses precision.

---

## 4. Analysis of Experimental Interventions

### Experiment B: Class Weighting vs. SMOTE
* **B1 (SMOTE + Balanced Weights)**: Performed worse than baseline. Combining synthetic SMOTE interpolation with heavy inverse-frequency weights caused the tree splits to overfit synthetic artifacts, dropping `Uploading_Attack` F1 to 39.6%.
* **B2 (No SMOTE + Balanced Weights)**: **Selected winner among 19-feature models.** Training on 100% real data with scikit-learn `class_weight='balanced'` in RF and inverse-frequency sample weights in XGBoost improved test accuracy to **93.43%** (+0.24%), test Macro F1 to **85.97%** (+0.57%), `Backdoor_Malware` F1 to **57.0%** (+5.5%), and `Recon-PingSweep` F1 to **56.3%** (+3.0%).
  * **Trade-off**: As noted up front, `Uploading_Attack` F1 dropped from **51.6% to 46.0%** in B2. While recall rose from 41.2% to 58.8% (20 true flows caught vs. 14), precision fell from 70.0% to 37.7% due to 33 false positives from sibling Web attacks.

### Experiment C: 46 Features vs. 19 Features
* Retaining all 46 features provided modest disambiguation for Web attacks, lifting overall accuracy to **93.90%** and `Backdoor_Malware` F1 to **53.5%**.
* **Trade-off**:
  1. **Latency**: Inference latency increased from **0.0447 ms/flow** to **0.1033 ms/flow** (a 2.3x slowdown).
  2. **Pipeline Invariant**: Contradicts the project's explicit architectural claim of reducing features from 46 to 19 via RFE.

### Experiment D: Hierarchical Architecture
* Training dedicated Stage-2 specialists for Web-based (6 classes) and Recon (5 classes) did not improve rare-class F1 (test F1: 50.4% Backdoor, 47.2% PingSweep, 47.6% Uploading).
* **Reason**: Errors from Stage 1 (8-class family classification) cascade directly to Stage 2. Furthermore, inside the Web specialist, the lack of payload-level inspection in tabular flow features prevents tree models from cleanly separating web shell uploads from injection queries.

### Experiment E: Best Combination (Winner in Raw Accuracy)
* Combining all 46 features with pure un-SMOTEd real training data and balanced class weights achieved the highest scores:
  * Accuracy: **93.93%**
  * Macro F1: **86.89%**
  * `Backdoor_Malware` F1: **58.2%**
  * `Recon-PingSweep` F1: **55.2%**
  * `Uploading_Attack` F1: **56.8%**

---

## 5. Architectural Recommendation & Honeypot Note

1. **For 19-Feature Compliance**: Adopted **Method B2** (Balanced sample weighting on pure real training rows without SMOTE). It improves 34-class accuracy from 93.19% to **93.43%**, Macro F1 from 85.40% to **85.97%**, lifts `Backdoor_Malware` to 57.0% and `PingSweep` to 56.3%, while maintaining the 19-feature specification and sub-0.05 ms latency.
2. **For Presentation Defense**: Be transparent about the ~55-58% F1 limit on these 3 classes. Use the diagnostic findings to explain that this is a **benchmark data scarcity bottleneck**.
3. **Cowrie Honeypot Enrichment (Future Work)**: Enriching flows with payload logs from a Cowrie SSH/Telnet honeypot is identified as essential future work to resolve intra-family ambiguity, as the current network dataset loader processes purely statistical flow headers without raw log payloads.
