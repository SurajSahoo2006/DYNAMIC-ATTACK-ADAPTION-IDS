# Phase 10: Zero-Day (Unseen Attack) Detection Layer

## Setup
- **Anomaly Detector Architecture**: PyTorch fully-connected Autoencoder (`19 -> 12 -> 6 -> 12 -> 19`) with ReLU activations and MSE reconstruction loss.
- **Benign-Only Training**: Fitted strictly on 3998 real Benign flows (no SMOTE rows) using Adam (learning rate `1e-3`, batch size 512, maximum 40 epochs) with early stopping on 4996 validation Benign flows.
- **Reconstruction Threshold**: Fixed at the 95th percentile of validation Benign reconstruction error (`threshold = 0.998414`). The test set is never accessed to select this cutoff.
- **Combined Decision Rule**: A flow is flagged UNKNOWN if `AE reconstruction score > threshold` OR `max predicted class probability < 0.60` (uncertainty threshold matching Phase 3).

## Headline Finding
Across the 7 held-out attack classes, 5 of 7 were flagged UNKNOWN in at least 50% of flows, and 0 of 7 were silently labelled Benign in more than 20% of flows.

## Table 1: Known-Class Control (Full Test Set)
Flagged-UNKNOWN rates on the full test set evaluated against the Phase 3 B2 ensemble (RF 0.6 + XGBoost 0.4). For Benign traffic, this represents the false-positive rate.

| Class | Test Flows | Flagged UNKNOWN (95% Wilson CI) |
|:---|---:|:---|
| Benign | 1000 | 266 of 1000 (26.60%) [23.95%, 29.42%] |
| BruteForce | 358 | 136 of 358 (37.99%) [33.11%, 43.12%] |
| DDoS | 11338 | 2917 of 11338 (25.73%) [24.93%, 26.54%] |
| DoS | 3971 | 144 of 3971 (3.63%) [3.09%, 4.25%] |
| Mirai | 2960 | 1015 of 2960 (34.29%) [32.60%, 36.02%] |
| Recon | 4002 | 1095 of 4002 (27.36%) [26.00%, 28.76%] |
| Spoofing | 1991 | 700 of 1991 (35.16%) [33.09%, 37.28%] |
| Web-based | 672 | 298 of 672 (44.35%) [40.63%, 48.12%] |

## Table 2: Leave-One-Class-Out Evaluation (Unseen Attack Simulation)
For each attack class, a fresh Random Forest classifier is trained on all training flows with that class excluded (capped at 50,000 stratified samples). Evaluation is performed on test flows of the held-out class using the fixed Autoencoder and uncertainty rule.

| Held-Out Class | Test Flows | Flagged UNKNOWN (Either) | Flagged by AE | Flagged by Uncertainty (<0.60) | Silently Benign | Silently Other Attack |
|:---|---:|:---|:---|:---|:---|:---|
| BruteForce | 358 | 308 of 358 (86.03%) [82.06%, 89.24%] | 22 of 358 (6.15%) [4.09%, 9.13%] | 301 of 358 (84.08%) [79.93%, 87.50%] | 5 of 358 (1.40%) [0.60%, 3.23%] | 45 of 358 (12.57%) [9.53%, 16.41%] |
| DDoS | 11338 | 5507 of 11338 (48.57%) [47.65%, 49.49%] | 2885 of 11338 (25.45%) [24.65%, 26.26%] | 3401 of 11338 (30.00%) [29.16%, 30.85%] | 3 of 11338 (0.03%) [0.01%, 0.08%] | 5828 of 11338 (51.40%) [50.48%, 52.32%] |
| DoS | 3971 | 734 of 3971 (18.48%) [17.31%, 19.72%] | 134 of 3971 (3.37%) [2.86%, 3.98%] | 606 of 3971 (15.26%) [14.18%, 16.41%] | 0 of 3971 (0.00%) [0.00%, 0.10%] | 3237 of 3971 (81.52%) [80.28%, 82.69%] |
| Mirai | 2960 | 2828 of 2960 (95.54%) [94.74%, 96.23%] | 1001 of 2960 (33.82%) [32.14%, 35.54%] | 2777 of 2960 (93.82%) [92.89%, 94.63%] | 0 of 2960 (0.00%) [0.00%, 0.13%] | 132 of 2960 (4.46%) [3.77%, 5.26%] |
| Recon | 4002 | 3246 of 4002 (81.11%) [79.87%, 82.29%] | 263 of 4002 (6.57%) [5.84%, 7.38%] | 3211 of 4002 (80.23%) [78.97%, 81.44%] | 281 of 4002 (7.02%) [6.27%, 7.86%] | 475 of 4002 (11.87%) [10.90%, 12.91%] |
| Spoofing | 1991 | 1496 of 1991 (75.14%) [73.19%, 76.99%] | 346 of 1991 (17.38%) [15.78%, 19.11%] | 1430 of 1991 (71.82%) [69.81%, 73.76%] | 199 of 1991 (9.99%) [8.75%, 11.39%] | 296 of 1991 (14.87%) [13.37%, 16.50%] |
| Web-based | 672 | 368 of 672 (54.76%) [50.98%, 58.49%] | 42 of 672 (6.25%) [4.66%, 8.34%] | 335 of 672 (49.85%) [46.08%, 53.62%] | 0 of 672 (0.00%) [0.00%, 0.57%] | 304 of 672 (45.24%) [41.51%, 49.02%] |

## Limitations
- **Heterogeneous Benign Class**: The Autoencoder is trained on one aggregated Benign class that encompasses diverse normal protocols; subtle anomalies within complex normal behaviors may not trigger reconstruction error.
- **Attacks Mimicking Normal Patterns**: Attacks that closely resemble normal application traffic (such as certain Web-based and Spoofing variants) can exhibit low reconstruction error and rely primarily on uncertainty thresholds for detection.
- **Duration Point Mass**: A point mass of 45.26% exists in test flow durations, which can affect feature scaling and reconstruction residuals across specific temporal profiles.
- **Simulated Zero-Days**: Held-out classes originate from the known dataset distribution and serve as proxy unseen attacks rather than novel in-the-wild zero-day vulnerabilities.
- **Single Random Seed**: Training and evaluation were conducted with seed 42 without hyperparameter tuning on the test set.
- **Threshold Trade-off**: The 95th percentile threshold establishes an empirical trade-off between the Benign false-positive rate (26.60%) and open-set recall, without subsequent optimization.
