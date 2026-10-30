# Dynamic Attack Adaptation — IDS Pipeline
### B.Tech Sem IV Research Project | NMIMS Design Experience / IROP

> **Project**: AI/ML pipeline for adaptive network intrusion detection in IoT environments.  
> **Dataset**: CICIoT2023 (University of New Brunswick) — 47M+ records, 33 attack types (34 fine-grained classes).  
> **Branch**: `phase5-marl-dashboard`  
> **Pipeline Scope**:  
> - **Phase 1**: Chunked Ingestion & Invariant Validation  
> - **Phase 2**: Leak-Free Preprocessing & Recursive Feature Elimination (46 → 19 features)  
> - **Phase 3**: Multi-Model Ensemble (Method B2: RF 0.6 + XGBoost 0.4) & LLM Semantic Orchestrator (Ollama Llama-3.2:3b)  
> - **Phase 4**: Explainable AI (SHAP TreeExplainer), Realistic Adversarial Robustness (ART), and Tamper-Evident Forensic Logging  
> - **Phase 5**: Multi-Agent Reinforcement Learning (MAPPO) Tactical Defense & Standalone SOC Operational Dashboard  

---

## Quick Start
1. Set `CIC_DATA_RAW_DIR` to your CICIoT2023 CSV folder.
2. Run `python run_phase1_2.py` (Phases 1-2: data pipeline + features).
3. Run `python run_phase3_ensemble.py` (Phase 3: ensemble training).
4. Run `python run_phase3_llm.py` (Phase 3: LLM — requires Ollama + `ollama pull llama3.2:3b`).
5. Run `python adversarial/run_phase4_adversarial.py --stage fgsm` (Phase 4: SHAP + adversarial).
6. Run `python marl/mappo.py --reward_setting balanced` (Phase 5: MARL).
7. Run `python package_model.py` (Phase 6: packaging).
8. Run `python reproduce.py` to verify all artifacts.
9. Run `python export_database.py` to export CSV splits for submission.
10. Open `dashboard/dashboard.html` in a browser.

---

## Architecture Overview

```
Network Traffic (PCAP)          Honeypot (Cowrie JSON)*      Benchmark Datasets
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
            [Phase 3A: Ensemble Classifier (Method B2)]
            ├── Random Forest (0.60) + XGBoost (0.40) + PyTorch DNN (0.00)
            │
            +---------------------------------------+
            |                                       |
            ▼                                       ▼
    [Phase 4A: Explainable AI]             [Phase 4B: Adversarial Suite]
    ├── SHAP TreeExplainer (2,000 flows)   ├── Domain Constraints (11 modifiable / 8 fixed)
    ├── Stability check (rho = 0.9088)     ├── FGSM Transfer (White-box DNN -> Ensemble)
    └── Opt-in --use-shap telemetry        ├── HopSkipJump & ZOO Black-Box Attacks
            |                              └── Adversarial Retraining Defense
            |                                       |
            +-------------------+-------------------+
                                |
                                ▼
            [Phase 3B: LLM Semantic Orchestrator]
            └── Ollama Llama-3.2:3b (CPU-only, temp=0) / Deterministic Expert Fallback
                (Translates predictions & SHAP attributions into analyst playbooks)
                                |
                                ▼
            [Phase 4C & 5: Forensic Alert Logging & ECS Export]
            ├── Cryptographic SHA-256 Tamper-Evident Hash Chaining (logs/alerts_phase5.jsonl)
            ├── External Immutable Anchor (logs/alerts_phase5_tail_hash.txt)
            └── Elastic Common Schema (ECS) ndjson Export (logs/alerts_phase5_ecs.ndjson)
                                |
                                ▼
            [Phase 5: MARL Tactical Defense Team (MAPPO)]
            ├── Triage Agent (3 actions): Accept / Flag for Review / Defer
            ├── Containment Agent (4 actions): None / Rate Limit / Block / Isolate (K=3 steps)
            ├── Escalation Agent (3 actions): None / Notify Analyst / Forensic Capture (Budget B=5)
            └── Centralized Critic: Concatenated Multi-Agent State Evaluation (84 dims)
                                |
                                ▼
            [Phase 5: Autonomous SOC Operational Dashboard]
            └── Standalone HTML Dashboard with Base64 Visualizations (reports/dashboard.html)

* Note: Honeypot log enrichment is future work; current pipeline consumes statistical flow vectors.
```

---

## Repository Structure

```
.
├── config.py                         # Central configuration for Phases 1-3
├── config_marl.py                    # Single Source of Truth for Phase 5 (MARL, rewards, budgets)
├── requirements.txt                  # Full project dependencies (including torch, shap, art, pytest)
├── run_phase1_2.py                   # Phase 1 & 2 pipeline runner
├── run_phase3_ensemble.py            # Phase 3 ensemble training & evaluation runner
├── run_phase3_llm.py                 # Phase 3/4 LLM orchestrator runner (--use-shap flag)
├── predict.py                        # Standalone flow prediction & severity module (B2 default)
├── docker-compose.yml                # Optional ELK stack compose manifest (reference only; not run)
│
├── marl/                             # Phase 5: Multi-Agent Tactical Defense Module
│   ├── build_cache.py                # Precomputes ensemble probabilities and 70/30 validation split
│   ├── env.py                        # Vectorized campaign defense environment (T=50, K=3, B=5)
│   ├── stats.py                      # Statistical tools (Wilson score intervals, 10k bootstrap CIs)
│   ├── baselines.py                  # Benchmark policies (Random, None, Max, Severity Rule, Oracle)
│   ├── mappo.py                      # 3 Actor MLPs + Centralized Critic with PPO & GAE
│   ├── evaluate.py                   # Single-pass 500 test episode evaluation & decision rule
│   ├── stress_test.py                # Adversarial FGSM (eps=0.10) evasion stress testing
│   └── tests/
│       └── test_env.py               # Pytest suite: no leakage, determinism, reward math, bounds
│
├── alerts/                           # Forensic Alert Logging Module
│   ├── alert_logger.py               # Tamper-evident logger with SHA-256 chaining and MARL fields
│   ├── verify_log.py                 # Cryptographic auditor: detects mid-chain tamper & truncation
│   ├── generate_and_test_alerts.py   # Phase 4 alert generation & tamper test suite
│   └── generate_phase5_alerts.py     # Phase 5 alert generator (45 demo flows with MARL actions)
│
├── export/                           # SIEM / ELK Export Module
│   └── elk_export.py                 # Transforms alerts to Elastic Common Schema (ndjson)
│
├── dashboard/                        # Autonomous Dashboard Module
│   └── build_dashboard.py            # Generates standalone reports/dashboard.html (inline Base64 PNGs)
│
├── tools/                            # Verification & Consistency Auditors
│   ├── check_report_consistency.py   # Comprehensive mathematical, empirical & formatting auditor
│   └── rerender_check.py             # LaTeX math delimiter & header integrity checker
│
├── xai/                              # Phase 4A: Explainable AI Module
│   ├── shap_explain.py               # SHAP TreeExplainer wrapper & stability evaluation
│   └── run_phase4_shap.py            # SHAP runner (2,000 stratified flows, global/class plots)
│
├── adversarial/                      # Phase 4B: Adversarial Robustness Module
│   ├── constraints.py                # 19-feature domain constraints, clipping, and bounds
│   └── run_phase4_adversarial.py     # FGSM transfer, HopSkipJump, ZOO, and defense retraining
│
├── models/                           # Model Definitions & Saved Weights
│   ├── ensemble.py                   # RF + XGBoost + PyTorch DNN + soft/weighted voting
│   ├── wire_b2_models.py             # Model wiring utility for Method B2
│   └── marl/                         # MAPPO Checkpoints (best rl_val return per seed)
│
├── reports/                          # Generated empirical artifacts, plots, and markdown summaries
│   ├── marl_cache_metrics.json       # Cache row counts and classifier accuracies
│   ├── marl_tests.txt                # Pytest verification output
│   ├── marl_baselines.json           # Baseline benchmark metrics across settings
│   ├── marl_pilot.json               # Pilot steps/sec benchmark and training budget selection
│   ├── marl_eval.json                # Single-pass 500 episode test evaluation metrics
│   ├── marl_stress_test.json         # FGSM perturbation stress test results
│   ├── marl_returns.png              # Test returns comparison chart
│   ├── marl_learning_curves.png      # MAPPO training convergence curves
│   ├── dashboard.html                # Standalone operational SOC dashboard
│   ├── phase5_log_verification.txt   # Cryptographic 3-test log verification record
│   ├── phase5_summary.md             # Programmatically generated Phase 5 summary report
│   └── phase4_summary.md             # Programmatically audited Phase 4 summary report
│
└── logs/                             # Audit Trails and ECS Exports
    ├── alerts.jsonl                  # Phase 4 forensic alert log
    ├── alerts_phase5.jsonl           # Phase 5 forensic alert log with MARL actions
    ├── alerts_phase5_tail_hash.txt   # External immutable tail anchor
    └── alerts_phase5_ecs.ndjson      # Elastic Common Schema formatted alert stream
```

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone -b phase5-marl-dashboard https://github.com/SurajSahoo2006/DYNAMIC-ATTACK-ADAPTION-IDS
cd DYNAMIC-ATTACK-ADAPTION-IDS
pip install -r requirements.txt
```

> [!NOTE]
> Raw CICIoT2023 CSV datasets (`train.csv`, `test.csv`, `validation.csv`) are not tracked in the git repository due to size. Place raw CSVs in `data/raw/` or set the environment variable `CIC_DATA_RAW_DIR` to your raw data path.

---

## Running the Complete Pipeline (Step-by-Step)

### Phases 1 to 4: Ingestion, Training, XAI & Adversarial Robustness
```bash
# 1. Ingestion & Preprocessing (46 -> 19 features via RFE):
python run_phase1_2.py --sample 5000

# 2. Method B2 Ensemble Training (RF 0.6 + XGBoost 0.4):
python run_phase3_ensemble.py

# 3. Explainable AI (SHAP TreeExplainer across 2,000 flows):
python xai/run_phase4_shap.py

# 4. Realistic Adversarial Suite (Constrained FGSM, HopSkipJump, ZOO, Defense Matrix):
python adversarial/run_phase4_adversarial.py --stage all
```

---

### Phase 5: Multi-Agent Tactical Defense & Operational Dashboard

Run the Phase 5 pipeline stages in sequence. All commands execute locally in the foreground, track peak memory via `psutil` (under 4 GB), and write structured JSON artifacts:

| Step | Command | Description | Measured Runtime | Peak RAM |
|:---:|:---|:---|:---:|:---:|
| **1** | `python marl/build_cache.py` | Runs frozen B2 ensemble over val & test sets; generates 70/30 `rl_train`/`rl_val` splits | ~6.1 s | 610 MB |
| **2** | `pytest marl/tests/test_env.py -v` | Unit tests: no label leakage, determinism, reward arithmetic, budget/posture limits | ~1.6 s | 95 MB |
| **3** | `python marl/baselines.py` | Benchmarks 7 baselines (Random, None, Max, Severity Rule, Budget-Aware Severity, Label-Aware Reference Rules) over 200 episodes | ~31.9 s | 129 MB |
| **4** | `python marl/mappo.py --pilot` | Runs 20,000-step pilot, computes steps/sec, sets budget (200,000 steps) | ~97.9 s | 367 MB |
| **5** | `python marl/mappo.py --resume` | Trains 5 seeds under `balanced` setting; selects checkpoints by best `rl_val` return | ~48.6 min | 410 MB |
| **6** | `python marl/evaluate.py` | Evaluates MAPPO and baselines on SAME 500 test episodes; applies pre-registered decision rule | ~160.8 s | 297 MB |
| **7** | `python marl/stress_test.py` | Paired like-for-like test of all policies against 500 clean vs FGSM-perturbed flows ($\epsilon=0.10$) | ~299.8 s | 52 MB |
| **8** | `python alerts/generate_phase5_alerts.py` | Emits 45 alerts with MARL actions to `logs/alerts_phase5.jsonl`; executes 3-test cryptographic audit | ~2.0 s | 135 MB |
| **9** | `python export/elk_export.py` | Exports alerts to Elastic Common Schema (`logs/alerts_phase5_ecs.ndjson`) | ~0.1 s | 60 MB |
| **10** | `python dashboard/build_dashboard.py` | Compiles self-contained `reports/dashboard.html` with embedded Base64 PNG charts | ~1.5 s | 120 MB |
| **11** | `python reports/generate_phase5_report.py` | Compiles `reports/phase5_summary.md` 100% programmatically from JSON artifacts | ~0.2 s | 60 MB |
| **12** | `python tools/check_report_consistency.py` | Automated auditor verifying mathematical, empirical & formatting consistency | ~1.0 s | 80 MB |

---

## Phase Roadmap

| Phase | Component | Status | Empirical Highlights |
|:---:|:---|:---:|:---|
| **1** | Ingestion & Sampling | **COMPLETE** | Chunked reservoir sampling, validation reporting |
| **2** | Preprocessing & Feature Selection | **COMPLETE** | Clean, encode, 80/20 split, scale, SMOTE, RFE (46 → 19 features) |
| **3A** | Ensemble Classifier (RF + XGB + DNN) | **COMPLETE** | 8-class (95.53% acc, 87.66% F1); 34-class B2 (93.43% acc, 85.97% F1) |
| **3B** | LLM Semantic Orchestrator | **COMPLETE** | 40/40 real Ollama outputs (100%), 0 fallbacks, 12.0s/flow, peak 315 MB RAM |
| **4A** | Explainable AI (SHAP TreeExplainer) | **COMPLETE** | 2,000 stratified flows, $\rho = 0.9088$ stability, `--use-shap` LLM integration |
| **4B** | Adversarial Robustness (ART) | **COMPLETE** | Domain constraints, zero-attack control (0.0 drift), random noise control, FGSM transfer, multi-budget HSJ & ZOO, 4-arm defense matrix |
| **4C** | Cryptographic Alert Logging | **COMPLETE** | Flat ELK-ready schema, SHA-256 hash chaining, 100% verified tamper detection |
| **5A** | MARL Tactical Defense (MAPPO) | **COMPLETE** | 3 cooperative agents (Triage, Containment, Escalation); paired diff +21.63 [20.39, 22.85] vs best non-learned baseline (severity_rule_budget_aware) (**better**) |
| **5B** | Autonomous SOC Dashboard & ECS Export | **COMPLETE** | Elastic Common Schema ndjson export, 3-test cryptographic log verification, standalone Base64 HTML dashboard |

---

### Phase 10: Zero-day detection

Run the Phase 10 zero-day detection pipeline:

```bash
# 1. Train Benign-Only PyTorch Autoencoder:
python zeroday/train_autoencoder.py

# 2. Run Open-Set Evaluation (Known Control & Leave-One-Class-Out):
python zeroday/evaluate_openset.py --resume

# 3. Generate Programmatic Phase 10 Summary Report:
python reports/generate_phase10_report.py
```
