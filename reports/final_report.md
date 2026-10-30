# Dynamic Attack Adaptation IDS -- Final Report

## 1. Project Overview
- **Dataset**: CICIoT2023, 34 attack labels mapped to 8 canonical classes (Benign, DDoS, DoS, Recon, Web-based, BruteForce, Spoofing, Mirai).
- **Features Used**: 19 features selected via Recursive Feature Elimination (RFE) from 46 raw telemetry attributes.
- **Team**: See README.

## 2. Phase 1-2: Data Pipeline
- **Feature Selection**: 19 continuous, counter, and protocol dimensions selected via RFE.
- **Selected Features List**: `flow_duration`, `Header_Length`, `Protocol Type`, `fin_flag_number`, `syn_count`, `ack_count`, `syn_flag_number`, `urg_count`, `rst_count`, `ICMP`, `Tot size`, `IAT`, `AVG`, `Max`, `Min`, `Tot sum`, `Magnitue`, `Covariance`, `Variance`.
- **Preprocessing & Normalization**: Median imputation, protocol categorical encoding, standard z-score scaling, and bounded SMOTE interpolation for minor attack classes.

## 3. Phase 3: Ensemble + LLM
- **8-Class Test Accuracy**: **95.53%** (B2 Ensemble: RF 0.60 + XGBoost 0.40).
- **34-Class Test Accuracy**: **93.19%** (Macro F1: **0.8540**).
- **LLM Orchestrator**: Local SLM (Llama 3.2 3B via Ollama) consumes structured ensemble telemetry and SHAP explanations to output human-interpretable incident narrative briefs with fixed severity classification.
- **Empirical Limitation**: Rare attack classes (`Uploading_Attack`, `Recon-PingSweep`, `Backdoor_Malware`) achieve sub-60% F1 scores due to extreme dataset scarcity in network traffic samples.

## 4. Phase 4: XAI and Adversarial Robustness
- **Top Global Explanatory Feature**: `IAT` (Inter-Arrival Time), exhibiting highest mean absolute SHAP value across attack families.
- **Adversarial Transferability**: PyTorch DNN surrogate gradients transfer to tree ensembles under realistic domain constraints, yielding **50.37% Attack Success Rate** (48.90% retained accuracy at epsilon = 0.10).
- **Empirical Limitation**: Tree ensembles exhibit decision boundary sensitivity near point-mass concentrations (e.g. single-packet zero flow duration); differentiable DNN surrogates facilitate black-box transfer attacks.

## 5. Phase 5: MARL Response Simulation
- **Simulation Architecture**: Decentralized Multi-Agent PPO (MAPPO) with 3 specialized defense agents (Triage, Containment, Escalation) and a centralized critic.
- **Evaluated Reward Profile**: `balanced`.
- **Aggregate Test Return**: **-7.67** across 500 test episodes.
- **Attack Containment Rate**: **96.98%** of attack flows successfully mitigated.
- **Benign Disruption Rate**: **8.08%** collateral penalty on benign flows.
- **Empirical Limitation**: Environmental damage and containment rewards are simulated via canonical cost models rather than measured on live physical production switches.

## 6. Phase 6: Hardening
- **Adversarial Pool Size**: Enlarged evaluation pool of 2000 total flows with 200 benign flows.
- **Reward Settings Comparison Matrix**:

| Reward Setting | Aggregate Return | Attacks Contained (%) | Benign Disrupted (%) | Truncated |
|:---|:---:|:---:|:---:|:---:|
| `balanced` | -7.67 | 96.98% | 8.08% | False |
| `security_first` | -38.80 | 100.00% | 100.00% | False |
| `availability_first` | -36.33 | 77.99% | 4.27% | False |

## 7. Limitations and Honest Assessment
1. **Rare-Class Scarcity**: Minority classes with extremely low sample counts in CICIoT2023 exhibit sub-60% macro F1 scores.
2. **Simulated MARL Dynamics**: Tactical agent reward functions and state transitions reflect simulated operational costs rather than hardware latency.
3. **Synthetic Evaluation Pool**: Adversarial attack perturbations are evaluated on offline held-out test splits rather than in-flight packet streams.
4. **Generated Text Non-Grounding**: LLM incident narratives are structured generative summaries and require verification before automated SOAR ticket execution.
5. **Offline ELK Compatibility**: Alert schemas are ECS-compliant and verified via ndjson exports; live ELK container clusters were out of scope for local 8 GB hardware.

## 8. Reproducibility
- Reviewers can verify and reproduce all pipeline stages, model metrics, and artifact generations using the unified master runner [`reproduce.py`](reproduce.py).
