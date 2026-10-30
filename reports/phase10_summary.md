# Phase 10: Zero-Day (Unseen Attack) Detection Layer

## Setup
- **Anomaly Detector Architecture**: PyTorch fully-connected Autoencoder (`19 -> 12 -> 6 -> 12 -> 19`) with ReLU activations and MSE reconstruction loss.
- **Benign-Only Training**: Fitted strictly on 3998 real Benign flows (no SMOTE rows). Note that the available real Benign training rows (3998) are fewer than the 100,000 cap. Training ran for 150 epochs with early stopping (patience 10) on 4996 validation Benign flows.
- **Operating Point 1 (Original)**: Selected with AE threshold at the 95.0th percentile of validation Benign error (`threshold = 0.374170`) and confidence cutoff `0.60`.
- **Operating Point 2 (FP Budget <= 10%)**: Selected via grid search over validation data (percentiles {95, 97.5, 99}, confidence cutoffs {0.30, 0.40, 0.50, 0.60}). The chosen setting uses the 97.5th percentile (`threshold = 0.534869`) and confidence cutoff `0.40`, yielding validation Benign FPR of 9.33%. Selection used validation rows of the held-out classes (a mild mismatch with a true zero-day setting).

## Executive Headline Finding
- **(a) Benign False-Positive Rate**: At the original operating point (95th percentile, confidence 0.60), the test Benign false-positive rate is 26.30%. At the FP-budget operating point (97.5th percentile, confidence 0.40), the test Benign false-positive rate is 8.00%.
- **(b) Detection Lift Over Known Control**: When attack classes are held out during training, combined flagging increases relative to the known-class control across all 7 classes (BruteForce: +47.21% [+41.34%, +53.07%]; DDoS: +19.25% [+18.49%, +19.94%]; DoS: +13.57% [+12.49%, +14.68%]; Mirai: +23.38% [+21.82%, +24.80%]; Recon: +43.10% [+41.38%, +44.75%]; Spoofing: +35.06% [+32.75%, +37.47%]; Web-based: +10.71% [+7.59%, +13.84%]).
- **(c) Held-Out Missed Rates (Unflagged Traffic)**: Flows not flagged as UNKNOWN (sum of silently Benign and silently other attack) under the original operating point are: BruteForce: 48 of 358 (13.41%); DDoS: 3958 of 11338 (34.91%); DoS: 2078 of 3971 (52.33%); Mirai: 66 of 2960 (2.23%); Recon: 710 of 4002 (17.74%); Spoofing: 472 of 1991 (23.71%); Web-based: 305 of 672 (45.39%).
- **(d) Detector Contributions**: Across held-out classes, confidence-only flagging averages 60.72% (ranging from 15.26% to 93.82%), whereas AE-only flagging averages 30.11% (ranging from 6.25% to 73.95%).

## Table 1: Known-Class Control (Full Test Set)
Evaluation of the frozen Phase 3 B2 ensemble across the test set showing individual detector rates (AE-only, confidence-only) and combined flagged UNKNOWN rates at both operating points.

| Class | Test Flows | Combined (OP 1: 95th, 0.60) | AE-Only (OP 1) | Confidence-Only (OP 1) | Combined (OP 2: 97.5th, 0.40) |
|:---|---:|:---|:---|:---|:---|
| Benign | 1000 | 263 of 1000 (26.30%) [23.67%, 29.12%] | 37 of 1000 (3.70%) [2.70%, 5.06%] | 230 of 1000 (23.00%) [20.50%, 25.71%] | 80 of 1000 (8.00%) [6.47%, 9.85%] |
| BruteForce | 358 | 141 of 358 (39.39%) [34.46%, 44.53%] | 29 of 358 (8.10%) [5.70%, 11.39%] | 120 of 358 (33.52%) [28.83%, 38.56%] | 71 of 358 (19.83%) [16.03%, 24.27%] |
| DDoS | 11338 | 5198 of 11338 (45.85%) [44.93%, 46.76%] | 5171 of 11338 (45.61%) [44.69%, 46.53%] | 34 of 11338 (0.30%) [0.21%, 0.42%] | 5084 of 11338 (44.84%) [43.93%, 45.76%] |
| DoS | 3971 | 1354 of 3971 (34.10%) [32.64%, 35.59%] | 1345 of 3971 (33.87%) [32.41%, 35.36%] | 11 of 3971 (0.28%) [0.15%, 0.50%] | 1266 of 3971 (31.88%) [30.45%, 33.35%] |
| Mirai | 2960 | 2202 of 2960 (74.39%) [72.79%, 75.93%] | 2189 of 2960 (73.95%) [72.34%, 75.50%] | 15 of 2960 (0.51%) [0.31%, 0.83%] | 2067 of 2960 (69.83%) [68.15%, 71.46%] |
| Recon | 4002 | 1567 of 4002 (39.16%) [37.65%, 40.68%] | 755 of 4002 (18.87%) [17.68%, 20.11%] | 862 of 4002 (21.54%) [20.29%, 22.84%] | 997 of 4002 (24.91%) [23.60%, 26.28%] |
| Spoofing | 1991 | 821 of 1991 (41.24%) [39.09%, 43.41%] | 480 of 1991 (24.11%) [22.28%, 26.04%] | 369 of 1991 (18.53%) [16.89%, 20.30%] | 490 of 1991 (24.61%) [22.77%, 26.55%] |
| Web-based | 672 | 295 of 672 (43.90%) [40.19%, 47.68%] | 42 of 672 (6.25%) [4.66%, 8.34%] | 258 of 672 (38.39%) [34.79%, 42.13%] | 119 of 672 (17.71%) [15.01%, 20.78%] |

## Table 2: Leave-One-Class-Out Evaluation (Original Operating Point: 95th, 0.60)
For each held-out attack class, a fresh Random Forest (100 trees, max depth 15, balanced class weights, seed 42) is trained with that class excluded (capped at 50,000 stratified samples).

| Held-Out Class | Test Flows | Combined Flagged | AE-Only Flagged | Confidence-Only (<0.60) | Silently Benign | Silently Other Attack | Total Missed |
|:---|---:|:---|:---|:---|:---|:---|:---|
| BruteForce | 358 | 310 of 358 (86.59%) [82.67%, 89.74%] | 29 of 358 (8.10%) [5.70%, 11.39%] | 301 of 358 (84.08%) [79.93%, 87.50%] | 5 of 358 (1.40%) [0.60%, 3.23%] | 43 of 358 (12.01%) [9.04%, 15.79%] | 48 of 358 (13.41%) |
| DDoS | 11338 | 7380 of 11338 (65.09%) [64.21%, 65.96%] | 5171 of 11338 (45.61%) [44.69%, 46.53%] | 3401 of 11338 (30.00%) [29.16%, 30.85%] | 3 of 11338 (0.03%) [0.01%, 0.08%] | 3955 of 11338 (34.88%) [34.01%, 35.76%] | 3958 of 11338 (34.91%) |
| DoS | 3971 | 1893 of 3971 (47.67%) [46.12%, 49.23%] | 1345 of 3971 (33.87%) [32.41%, 35.36%] | 606 of 3971 (15.26%) [14.18%, 16.41%] | 0 of 3971 (0.00%) [0.00%, 0.10%] | 2078 of 3971 (52.33%) [50.77%, 53.88%] | 2078 of 3971 (52.33%) |
| Mirai | 2960 | 2894 of 2960 (97.77%) [97.17%, 98.24%] | 2189 of 2960 (73.95%) [72.34%, 75.50%] | 2777 of 2960 (93.82%) [92.89%, 94.63%] | 0 of 2960 (0.00%) [0.00%, 0.13%] | 66 of 2960 (2.23%) [1.76%, 2.83%] | 66 of 2960 (2.23%) |
| Recon | 4002 | 3292 of 4002 (82.26%) [81.04%, 83.41%] | 755 of 4002 (18.87%) [17.68%, 20.11%] | 3211 of 4002 (80.23%) [78.97%, 81.44%] | 276 of 4002 (6.90%) [6.15%, 7.72%] | 434 of 4002 (10.84%) [9.92%, 11.85%] | 710 of 4002 (17.74%) |
| Spoofing | 1991 | 1519 of 1991 (76.29%) [74.38%, 78.11%] | 480 of 1991 (24.11%) [22.28%, 26.04%] | 1430 of 1991 (71.82%) [69.81%, 73.76%] | 179 of 1991 (8.99%) [7.81%, 10.33%] | 293 of 1991 (14.72%) [13.23%, 16.34%] | 472 of 1991 (23.71%) |
| Web-based | 672 | 367 of 672 (54.61%) [50.83%, 58.34%] | 42 of 672 (6.25%) [4.66%, 8.34%] | 335 of 672 (49.85%) [46.08%, 53.62%] | 0 of 672 (0.00%) [0.00%, 0.57%] | 305 of 672 (45.39%) [41.66%, 49.17%] | 305 of 672 (45.39%) |

## Table 3: Leave-One-Class-Out Evaluation (FP-Budget Operating Point: 97.5th, 0.40)
Side-by-side evaluation of leave-one-class-out zero-day simulation under the tuned false-positive budget operating point.

| Held-Out Class | Test Flows | Combined Flagged | AE-Only Flagged | Confidence-Only (<0.40) | Silently Benign | Silently Other Attack | Total Missed |
|:---|---:|:---|:---|:---|:---|:---|:---|
| BruteForce | 358 | 130 of 358 (36.31%) [31.50%, 41.42%] | 19 of 358 (5.31%) [3.42%, 8.14%] | 120 of 358 (33.52%) [28.83%, 38.56%] | 19 of 358 (5.31%) [3.42%, 8.14%] | 209 of 358 (58.38%) [53.21%, 63.37%] | 228 of 358 (63.69%) |
| DDoS | 11338 | 5923 of 11338 (52.24%) [51.32%, 53.16%] | 5081 of 11338 (44.81%) [43.90%, 45.73%] | 1765 of 11338 (15.57%) [14.91%, 16.25%] | 26 of 11338 (0.23%) [0.16%, 0.34%] | 5389 of 11338 (47.53%) [46.61%, 48.45%] | 5415 of 11338 (47.76%) |
| DoS | 3971 | 1340 of 3971 (33.74%) [32.29%, 35.23%] | 1263 of 3971 (31.81%) [30.38%, 33.27%] | 84 of 3971 (2.12%) [1.71%, 2.61%] | 0 of 3971 (0.00%) [0.00%, 0.10%] | 2631 of 3971 (66.26%) [64.77%, 67.71%] | 2631 of 3971 (66.26%) |
| Mirai | 2960 | 2206 of 2960 (74.53%) [72.93%, 76.06%] | 2067 of 2960 (69.83%) [68.15%, 71.46%] | 147 of 2960 (4.97%) [4.24%, 5.81%] | 0 of 2960 (0.00%) [0.00%, 0.13%] | 754 of 2960 (25.47%) [23.94%, 27.07%] | 754 of 2960 (25.47%) |
| Recon | 4002 | 1717 of 4002 (42.90%) [41.38%, 44.44%] | 672 of 4002 (16.79%) [15.67%, 17.98%] | 1295 of 4002 (32.36%) [30.93%, 33.82%] | 501 of 4002 (12.52%) [11.53%, 13.58%] | 1784 of 4002 (44.58%) [43.04%, 46.12%] | 2285 of 4002 (57.10%) |
| Spoofing | 1991 | 754 of 1991 (37.87%) [35.77%, 40.02%] | 365 of 1991 (18.33%) [16.69%, 20.09%] | 563 of 1991 (28.28%) [26.34%, 30.30%] | 310 of 1991 (15.57%) [14.04%, 17.23%] | 927 of 1991 (46.56%) [44.38%, 48.76%] | 1237 of 1991 (62.13%) |
| Web-based | 672 | 157 of 672 (23.36%) [20.32%, 26.71%] | 27 of 672 (4.02%) [2.78%, 5.78%] | 130 of 672 (19.35%) [16.54%, 22.50%] | 9 of 672 (1.34%) [0.71%, 2.53%] | 506 of 672 (75.30%) [71.90%, 78.41%] | 515 of 672 (76.64%) |

## Table 4: Detection Lift Over Known Control (Paired Bootstrap 95% CI)
Lift equals held-out combined flagged rate minus known-class control combined flagged rate on the same test flows, with paired bootstrap 95% confidence intervals (1,000 resamples, seed 42).

| Attack Class | Held-Out Rate (OP 1) | Known-Class Rate (OP 1) | Lift (OP 1: 95th, 0.60) | Lift (OP 2: 97.5th, 0.40) |
|:---|---:|---:|:---|:---|
| BruteForce | 86.59% | 39.39% | +47.21% [+41.34%, +53.07%] | +16.48% [+10.89%, +21.79%] |
| DDoS | 65.09% | 45.85% | +19.25% [+18.49%, +19.94%] | +7.40% [+6.90%, +7.89%] |
| DoS | 47.67% | 34.10% | +13.57% [+12.49%, +14.68%] | +1.86% [+1.44%, +2.32%] |
| Mirai | 97.77% | 74.39% | +23.38% [+21.82%, +24.80%] | +4.70% [+3.89%, +5.47%] |
| Recon | 82.26% | 39.16% | +43.10% [+41.38%, +44.75%] | +17.99% [+16.49%, +19.54%] |
| Spoofing | 76.29% | 41.24% | +35.06% [+32.75%, +37.47%] | +13.26% [+11.35%, +15.17%] |
| Web-based | 54.61% | 43.90% | +10.71% [+7.59%, +13.84%] | +5.65% [+2.52%, +8.93%] |

## Table 5: Threshold-Free Autoencoder Quality (AUROC on Test Set)
Area under the ROC curve evaluating AE reconstruction score discrimination (Benign vs each attack class) on the untouched test set, with 200 bootstrap resamples 95% confidence intervals.

| Attack Class | AE Reconstruction Score AUROC (95% Bootstrap CI) |
|:---|:---|
| BruteForce | 0.4986 [0.4639, 0.5333] |
| DDoS | 0.8174 [0.8059, 0.8313] |
| DoS | 0.8311 [0.8185, 0.8448] |
| Mirai | 0.9502 [0.9436, 0.9571] |
| Recon | 0.4628 [0.4479, 0.4783] |
| Spoofing | 0.6650 [0.6459, 0.6800] |
| Web-based | 0.4637 [0.4351, 0.4904] |

## Limitations
- **Detector Contribution Balance**: The confidence rule contributes most of the detection; the autoencoder adds a smaller share (see AE-only rates).
- **Heterogeneous Benign Class**: The Autoencoder is trained on one aggregated Benign class that encompasses diverse normal protocols; subtle anomalies within complex normal behaviors may not trigger reconstruction error.
- **Attacks Mimicking Normal Patterns**: Attacks that closely resemble normal application traffic (such as certain Web-based and Spoofing variants) can exhibit low reconstruction error and rely primarily on uncertainty thresholds for detection.
- **Duration Point Mass**: A point mass of 45.26% exists in test flow durations, which can affect feature scaling and reconstruction residuals across specific temporal profiles.
- **Simulated Zero-Days**: Held-out classes originate from the known dataset distribution and serve as proxy unseen attacks rather than novel in-the-wild zero-day vulnerabilities.
- **Single Random Seed**: Training and evaluation were conducted with seed 42 without hyperparameter tuning on the test set.
- **Threshold Trade-off**: The 95th percentile threshold establishes an empirical trade-off between the Benign false-positive rate and open-set recall.
