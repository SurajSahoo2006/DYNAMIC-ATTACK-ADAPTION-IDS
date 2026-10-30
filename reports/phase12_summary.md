# Phase 12 -- Probability Calibration and Selective Prediction

## Overview and Objectives

Phase 12 examines whether post-hoc probability calibration improves the confidence estimates of the Phase 3 B2 ensemble (Random Forest 0.60 + XGBoost 0.40, 19 selected features) and investigates two operational questions:
1. **Selective Prediction**: Does calibrating maximum predicted probability improve the risk-coverage trade-off (AURC) and increase test flow coverage at a strict validation-chosen operating point (accepted-flow accuracy $\ge 99.0\%$)?
2. **Open-Set Detection at Matched False-Positive Budget**: When applying the "low confidence = UNKNOWN" rule to unseen attack families, does probability calibration yield higher detection rates than raw confidence when both methods are held to the exact same false-positive budget (at most 10% of validation Benign flows flagged)?

### Key Empirical Findings

- **Calibration Quality**: On the untouched test set ($N = 26,292$), Expected Calibration Error (ECE, 15 bins) decreased from **0.0270** ([0.0259, 0.0291]) to **0.0087** ([0.0076, 0.0109]), multiclass Brier score decreased from **0.0648** to **0.0586**, and NLL decreased from **0.1404** to **0.1231**.
- **Calibrator Selection**: On a stratified 50/50 validation split, `isotonic_regression` was selected over temperature scaling due to lower select-half NLL (0.0897 vs 0.0930).
- **Selective Prediction (AURC)**: Area Under the Risk-Coverage curve on the test set decreased from **0.002035** (0.002035 [0.001870, 0.002234]) to **0.001830** (0.001830 [0.001665, 0.002034]), reflecting an AURC delta of **-0.000205** (paired bootstrap 95% CI: -0.000205 [-0.000287, -0.000118]).
- **Operating Point at $\ge 99.0\%$ Accuracy**: At the lowest validation cutoff achieving $\ge 99.0\%$ accepted accuracy (raw $\theta = 0.5376$, calibrated $\theta = 0.6220$), test set coverage was **24,740 of 26,292 flows (94.10%)** for raw vs **24,898 of 26,292 flows (94.70%)** for calibrated, an increase of **+0.60%** (+158 flows).
- **Open-Set Detection at Matched 10% Benign Budget**: In **3** of the 7 held-out classes, the calibrated rule flagged significantly more flows than the raw rule; in **0** classes differences were within sampling uncertainty; and in **4** classes it flagged significantly fewer flows.

## Table 1 -- Probability Calibration Metrics (Raw vs Calibrated)

Evaluation on the validation set ($N = 106,766$) and untouched test set ($N = 26,292$) with 200-sample bootstrap 95% confidence intervals:

| Split | Metric | Raw Ensemble Point Est | Raw 95% CI | Calibrated Point Est | Calibrated 95% CI | Absolute Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation** | ECE (15 equal-width bins) | 0.0187 | [0.0182, 0.0197] | 0.0053 | [0.0048, 0.0062] | -0.0134 |
| **Validation** | Multiclass Brier Score | 0.0490 | [0.0478, 0.0502] | 0.0397 | [0.0386, 0.0410] | -0.0093 |
| **Validation** | Negative Log-Likelihood | 0.1048 | [0.1027, 0.1070] | 0.0763 | [0.0741, 0.0783] | -0.0285 |
| **Validation** | Classification Accuracy | 96.45% | -- | 97.14% | -- | +0.69% |
| **Test** | ECE (15 equal-width bins) | 0.0270 | [0.0259, 0.0291] | 0.0087 | [0.0076, 0.0109] | -0.0183 |
| **Test** | Multiclass Brier Score | 0.0648 | [0.0625, 0.0676] | 0.0586 | [0.0559, 0.0617] | -0.0062 |
| **Test** | Negative Log-Likelihood | 0.1404 | [0.1356, 0.1461] | 0.1231 | [0.1155, 0.1310] | -0.0173 |
| **Test** | Classification Accuracy | 95.42% | -- | 95.76% | -- | +0.34% |

### Note on Baseline B2 8-Class Test Accuracy

The 8-class test accuracy recorded in `models/ensemble_weights_8class_b2.json` telemetry is **0.9536** (95.36%), produced during Phase 3 model wiring (`models/wire_b2_models.py`). In contrast, the evaluation from the unchanged prediction path (`calibration/measure.py` executing `get_or_load_model('8class', 'b2').predict_probabilities(X_test)['weighted']` on untouched `test.parquet`, $N = 26,292$) yielded **0.9542** (95.42%, 25089 of 26292 flows (95.42%)). The exact factor accounting for this 0.0006 discrepancy (0.9536 vs 0.9542) between the static telemetry metadata and the active prediction pipeline is cause not determined.

## Table 2 -- Selective Prediction and Operating Point Analysis

Evaluates the risk-coverage curve on the test partition ($N = 26,292$) alongside an operating point selected strictly on the validation partition:

| Evaluation Aspect | Metric | Raw Maximum Probability | Calibrated Maximum Probability | Delta / Comparison |
| :--- | :--- | :---: | :---: | :---: |
| **Risk-Coverage Curve** | Test AURC (mean risk over coverage) | 0.002035 | 0.001830 | -0.000205 |
| **Risk-Coverage Curve** | Bootstrap 95% CI on Test AURC | [0.001870, 0.002234] | [0.001665, 0.002034] | Shifted lower |
| **Risk-Coverage Curve** | Paired AURC Difference (Cal - Raw) | -- | -- | -0.000205 [-0.000287, -0.000118] |
| **Validation Operating Point** | Target Accepted Accuracy | $\ge 99.00\%$ | $\ge 99.00\%$ | Identical threshold requirement |
| **Validation Operating Point** | Selected Cutoff Threshold $\theta^*$ | 0.5376 | 0.6220 | Cutoff adjusted post-calibration |
| **Validation Operating Point** | Achieved Validation Accuracy | 99.00% | 99.00% | Meets criterion |
| **Validation Operating Point** | Validation Flow Coverage | 95.57% | 96.21% | +0.64% on validation |
| **Test Set at $\theta^*$** | Accepted Flow Count | 24,740 of 26,292 | 24,898 of 26,292 | +158 accepted flows |
| **Test Set at $\theta^*$** | Test Flow Coverage | 94.10% [93.81%, 94.38%] | 94.70% [94.42%, 94.96%] | +0.60% coverage delta |
| **Test Set at $\theta^*$** | Accepted Flow Accuracy | 98.73% [98.58%, 98.86%] | 98.61% [98.46%, 98.75%] | -0.12% accuracy difference |

## Table 3 -- Open-Set Leave-One-Class-Out at Matched 10% Benign False-Positive Budget

For each held-out attack family, a fresh Random Forest (100 trees, depth 15, balanced class weights, seed 42, 50,000-row cap) was trained without seeing that attack class. Its calibrator and confidence cutoffs were fitted on validation rows excluding the held-out class so that at most 10.0% of validation Benign flows were flagged. Paired bootstrap 95% confidence intervals (1,000 resamples, seed 42) report the flagged rate difference (Calibrated - Raw):

| Held-Out Attack Class | Test Support | Raw Flagged Rate (%) | Calibrated Flagged Rate (%) | Paired Difference (Cal - Raw, 95% CI) | Silently-Benign (Raw vs Cal) | Silently-Other (Raw vs Cal) | Test Benign FPR (Raw vs Cal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BruteForce` | 358 | 137 of 358 (38.27%) | 97 of 358 (27.09%) | **-11.17% [-17.33%, -5.03%]** | 5.87% vs 4.19% | 55.87% vs 68.72% | 9.60% vs 11.40% |
| `DDoS` | 11,338 | 1360 of 11338 (12.00%) | 1764 of 11338 (15.56%) | **+3.56% [+3.06%, +4.04%]** | 0.25% vs 0.02% | 87.76% vs 84.42% | 9.50% vs 9.30% |
| `DoS` | 3,971 | 51 of 3971 (1.28%) | 79 of 3971 (1.99%) | **+0.71% [+0.35%, +1.08%]** | 0.00% vs 0.00% | 98.72% vs 98.01% | 11.20% vs 10.40% |
| `Mirai` | 2,960 | 86 of 2960 (2.91%) | 187 of 2960 (6.32%) | **+3.41% [+2.36%, +4.53%]** | 0.00% vs 0.00% | 97.09% vs 93.68% | 10.30% vs 10.40% |
| `Recon` | 4,002 | 1977 of 4002 (49.40%) | 1366 of 4002 (34.13%) | **-15.27% [-16.97%, -13.57%]** | 11.22% vs 12.79% | 39.38% vs 53.07% | 10.40% vs 12.50% |
| `Spoofing` | 1,991 | 666 of 1991 (33.45%) | 614 of 1991 (30.84%) | **-2.61% [-5.13%, -0.30%]** | 20.34% vs 27.32% | 46.21% vs 41.84% | 9.80% vs 11.10% |
| `Web-based` | 672 | 166 of 672 (24.70%) | 123 of 672 (18.30%) | **-6.40% [-9.08%, -3.57%]** | 0.74% vs 0.15% | 74.55% vs 81.55% | 9.70% vs 9.40% |

### Test Benign False-Positive Rate Budget Analysis

Across the held-out evaluation classes, the test Benign false-positive rate exceeded the 10.0% validation budget under the raw rule in 3 of 7 classes (`DoS` (11.20%), `Mirai` (10.30%), `Recon` (10.40%)) and under the calibrated rule in 5 of 7 classes (`BruteForce` (11.40%), `DoS` (10.40%), `Mirai` (10.40%), `Recon` (12.50%), `Spoofing` (11.10%)). This indicates that the validation budget was not exactly held on test.

### Headline Summary of Leave-One-Class-Out Evaluation

In **3** of the 7 held-out classes (`DDoS`, `DoS`, `Mirai`), the calibrated rule flagged significantly more unseen flows as UNKNOWN than the raw rule (paired bootstrap 95% CI strictly positive). In **0** classes differences were within sampling noise. In **4** classes (`BruteForce`, `Recon`, `Spoofing`, `Web-based`), the calibrated rule flagged significantly fewer unseen flows than the raw rule (paired bootstrap 95% CI strictly negative).

## Table 4 -- Known-Class Control Evaluation (Full B2 Ensemble)

Evaluates the known-class control on the full test set under the matched validation Benign budget:

| Class Name | Test Support | Raw Flagged Rate (95% CI) | Calibrated Flagged Rate (95% CI) | Paired Difference (Cal - Raw, 95% CI) | Status / Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Benign` | 1,000 | 105 (10.50%) [8.75%, 12.55%] | 122 (12.20%) [10.31%, 14.37%] | +1.70% [-0.20%, +3.60%] | Unchanged within CI |
| `BruteForce` | 358 | 72 (20.11%) [16.29%, 24.57%] | 79 (22.07%) [18.08%, 26.65%] | +1.96% [-2.79%, +6.42%] | Unchanged within CI |
| `DDoS` | 11,338 | 6 (0.05%) [0.02%, 0.12%] | 2 (0.02%) [0.00%, 0.06%] | -0.04% [-0.07%, -0.01%] | Decreased flagging |
| `DoS` | 3,971 | 5 (0.13%) [0.05%, 0.29%] | 4 (0.10%) [0.04%, 0.26%] | -0.03% [-0.08%, +0.00%] | Unchanged within CI |
| `Mirai` | 2,960 | 1 (0.03%) [0.01%, 0.19%] | 0 (0.00%) [0.00%, 0.13%] | -0.03% [-0.10%, +0.00%] | Unchanged within CI |
| `Recon` | 4,002 | 475 (11.87%) [10.90%, 12.91%] | 294 (7.35%) [6.58%, 8.20%] | -4.52% [-5.40%, -3.55%] | Decreased flagging |
| `Spoofing` | 1,991 | 191 (9.59%) [8.38%, 10.97%] | 171 (8.59%) [7.44%, 9.90%] | -1.00% [-2.16%, +0.30%] | Unchanged within CI |
| `Web-based` | 672 | 130 (19.35%) [16.54%, 22.50%] | 160 (23.81%) [20.74%, 27.17%] | +4.46% [+0.74%, +7.89%] | Increased flagging |

## Observations and Insights

1. **Calibration Metric Agreement**: Isotonic regression consistently decreased Expected Calibration Error across validation and test partitions (test ECE decreased from 0.0270 to 0.0087), aligning output probabilities with empirical observed frequencies.
2. **Risk-Coverage Curve Improvement**: Calibrated probabilities improved the test-set AURC from 0.002035 to 0.001830. When applying an identical 99.0% accuracy requirement selected on validation, calibrated probabilities accepted 158 additional test flows (+0.60% coverage) while maintaining 98.61% accuracy on the test set.
3. **Heterogeneous Impact on Open-Set Detection**: At a matched 10% false-positive budget on Benign validation traffic, probability calibration did not uniformly increase detection of unseen attacks across all classes. Across the 7 held-out classes, calibration yielded a higher flagged rate in 3 classes (`DDoS` (+3.56% [+3.06%, +4.04%]), `DoS` (+0.71% [+0.35%, +1.08%]), `Mirai` (+3.41% [+2.36%, +4.53%])) and a lower flagged rate in 4 classes (`BruteForce` (-11.17% [-17.33%, -5.03%]), `Recon` (-15.27% [-16.97%, -13.57%]), `Spoofing` (-2.61% [-5.13%, -0.30%]), `Web-based` (-6.40% [-9.08%, -3.57%])), with 0 classes within sampling uncertainty. The underlying mechanism driving differences in flagged rates across attack classes was not tested (mechanism not tested).

## Limitations and Boundaries

- **Validation Sample Distribution**: Calibrators were fitted on a class-capped validation sample (5,000 samples per fine-grained class) whose class balance differs from raw network traffic distributions.
- **Simulated Zero-Day Partition**: Held-out classes represent known canonical classes in CICIoT2023 treated as unseen; actual zero-day attacks in production environments may exhibit structural features outside the 19 selected dimensions.
- **Independent Autoencoder Baseline**: The autoencoder reconstruction detector developed in Phase 10 was left unchanged; this phase specifically isolated the effect of calibrating the classifier's confidence rule.
- **Single Evaluation Seed**: All data splits, subsampling, and bootstrap resamplings utilized random seed `42`.
- **Semantic Feature Overlap**: Probability calibration modifies threshold behavior but cannot distinguish attacks whose statistical network features mimic benign baseline traffic.