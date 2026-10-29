# Phase 5 Summary Report: Multi-Agent Tactical Defense & Operational Dashboard

**Generated:** 2026-10-29 18:25:30 UTC  
**Repository Branch:** `phase5-marl-dashboard`  
**Target Architecture:** Multi-Agent PPO (MAPPO) Cooperative Response + Elastic Common Schema + Forensic Dashboard  

---

## 1. Executive Overview and Simulation Framing

Phase 5 implements the final two operational tiers of the Dynamic Attack Adaptation IDS: (A) Multi-Agent Reinforcement Learning (MARL) tactical defense adaptation using MAPPO, and (B) a self-contained operational security dashboard with Elastic Common Schema (ECS) alert export. Because the CICIoT2023 dataset contains isolated feature flows without ground-truth network timestamps or IP addresses, Phase 5 operates as a **simulated sequential decision problem** grounded on real classifier outputs from the frozen Phase 3 B2 ensemble (Random Forest 0.6 + XGBoost 0.4).

> **Note on Simulation Framing:** The simulation environment models sequential campaign dynamics and trade-offs between cyber damage mitigation, collateral disruption of benign traffic, triage latency, and SOC escalation overhead. All reward functions and cost coefficients represent explicit analytical assumptions defined in `config_marl.py`. These results reflect policy optimization within the simulated decision process and do not depict unconstrained real-world network operational performance.

## 2. Multi-Agent Environment Architecture

- **Episode Horizon:** $T = 50$ steps per campaign episode.
- **Campaign Burst Dynamics:** Each episode samples a target attack family and embeds a burst of $10$ to $25$ consecutive attack flows within a stream of benign flows. Temporal correlation ensures that sequential posture persistence directly impacts downstream flows.
- **Three Cooperative Agents:**
  1. **Triage Agent** ($3$ discrete actions): `accept_verdict` ($0$), `flag_for_review` ($1$, grants $50\%$ discount to escalation), `defer` ($2$).
  2. **Containment Agent** ($4$ discrete actions): `none` ($0$), `rate_limit` ($1$), `block` ($2$), `isolate` ($3$). Active postures persist for $K = 3$ steps.
  3. **Escalation Agent** ($3$ discrete actions): `none` ($0$), `notify_analyst` ($1$), `forensic_capture` ($2$). Subject to per-episode budget $B = 5$.

- **Information Isolation:** Agents observe the $8$-class probability distribution, ensemble confidence score, one-hot severity, step fraction, remaining escalation budget, active posture flag, posture duration remaining, and rolling history of recent predicted classes ($25$ dims) concatenated with an agent identity vector ($3$ dims). Ground truth attack labels are **strictly withheld** from observation vectors.

### Mathematical Reward Formulation

At each environment step $t$ with active weights $(w_D, w_C, w_{cost}, w_B)$:
- If true class is an attack flow:
  $$\text{damage\_term} = - w_D \cdot \text{BaseDamage}[\text{severity}] \cdot \text{LeakMultiplier}[P_{\text{active}}]$$
  $$\text{collateral\_term} = 0.0$$
  $$\text{bonus\_term} = + w_B \cdot 2.0 \quad \text{if } (P_{\text{active}} \neq \text{none} \land a_{\text{escalation}} \neq \text{none}) \text{ else } 0.0$$
- If true class is benign:
  $$\text{damage\_term} = 0.0$$
  $$\text{collateral\_term} = - w_C \cdot \text{CollateralCost}[P_{\text{active}}]$$
  $$\text{bonus\_term} = 0.0$$
- Action execution costs and penalties:
  $$\text{cost\_term} = - w_{cost} \cdot (\text{DelayPenalty}[a_{\text{triage}}] + \text{ActionCost}[a_{\text{containment}}] + \text{EscalationCost}[a_{\text{escalation}}] + \text{ExcessBudgetPenalty})$$
- Team reward: $R_t = \text{damage\_term} + \text{collateral\_term} + \text{cost\_term} + \text{bonus\_term}$.

## 3. Data Cache Partitioning and Split Bounds

The frozen B2 ensemble inferences were computed in Stage 0 over `val_phase2.parquet` and `test_phase2.parquet`. The validation cache was split by row index using seed $42$ into `rl_train` ($70\%$) and `rl_val` ($30\%$). Agents were trained exclusively on `rl_train`; checkpoint selection used `rl_val`; and the test cache was held out until the final single-pass evaluation.

| Split Partition | Row Count | B2 Classifier Accuracy | Split Ratio |
| :--- | :--- | :--- | :--- |
| Validation Total | 106766 | 96.45% | 100% |
| `rl_train` Partition | 74736 | 96.42% | 70% |
| `rl_val` Partition | 32030 | 96.54% | 30% |
| `test` Cache Partition | 26292 | 95.42% | Held-out |

## 4. Environment Unit Testing and Verification

The unit test suite in `marl/tests/test_env.py` executed under `pytest` with $5$ test categories and $100\%$ pass rate (recorded in `reports/marl_tests.txt`):
1. **No True Label Leakage:** Observation vectors remained identical under artificial ground-truth label mutations.
2. **Deterministic Seeding:** Trajectories generated under identical seeds matched bit-for-bit.
3. **Reward Arithmetic:** Exact verification of three hand-computed test cases across reward weight configurations.
4. **Posture and Budget Constraints:** Verified $K=3$ step persistence and excess budget penalty trigger after $B=5$ actions.
5. **Split Boundary Isolation:** Verified disjoint index boundaries between `rl_train`, `rl_val`, and `test`.

## 5. Benchmark Baselines Evaluation

Five benchmark policies were evaluated over $200$ validation episodes and $200$ test episodes under the 'balanced' reward setting (recorded in `reports/marl_baselines.json`).

| Policy | Mean Return | 95% Bootstrap CI | Attacks Contained (x of y) | Benign Disrupted (x of y) | Notifications / Ep |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `random` | -261.56 | [-265.51, -257.71] | 3485 of 3544 (98.34% [97.86%, 98.71%]) | 6316 of 6456 (97.83% [97.45%, 98.16%]) | 33.18 |
| `always_none` | -99.55 | [-108.89, -90.46] | 0 of 3544 (0.00% [0.00%, 0.11%]) | 0 of 6456 (0.00% [0.00%, 0.06%]) | 0.0 |
| `max_containment` | -425.96 | [-430.30, -421.48] | 3544 of 3544 (100.00% [99.89%, 100.00%]) | 6456 of 6456 (100.00% [99.94%, 100.00%]) | 50.0 |
| `severity_rule` | -41.65 | [-44.76, -38.65] | 3443 of 3544 (97.15% [96.55%, 97.65%]) | 1579 of 6456 (24.46% [23.42%, 25.52%]) | 10.21 |
| `oracle` | -54.42 | [-56.96, -51.95] | 3544 of 3544 (100.00% [99.89%, 100.00%]) | 377 of 6456 (5.84% [5.29%, 6.44%]) | 17.72 |

## 6. MAPPO Training Protocol and Pilot Benchmark

A pilot run of $20000$ steps completed in $42.08$ seconds ($475.28$ steps/second). In accordance with the pre-registered budget formula `budget = min(MAX_TRAIN_STEPS, steps_per_sec * 1500s)` ($\ge 100,000$ steps), the chosen training budget was $200000$ steps per seed.

> **Budget Cap Constraint:** An evaluation of $15$ runs ($3$ reward settings $\times$ $5$ seeds) yielded an estimated runtime exceeding the $45$-minute ceiling ($6312.0$ s > $2700$ s). In strict adherence to the Stage 4 protocol, training was restricted to the `balanced` setting. The reason recorded in `reports/marl_pilot.json`: *"Estimated training runtime (6312.1s) exceeds 45-minute limit (2700s). Reduced to 'balanced' setting only as per Stage 4 specification."*.

### Multi-Seed Training Convergence (`balanced` Setting)

| Training Seed | Budget (Steps) | Best `rl_val` Return | Final `rl_val` Return | Runtime (s) | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Seed 42 | 200000 | -4.89 | -5.46 | 383.3 | 396.2 |
| Seed 123 | 200000 | 1.02 | -0.15 | 344.6 | 405.2 |
| Seed 456 | 200000 | -7.17 | -10.16 | 384.4 | 409.7 |
| Seed 789 | 200000 | -6.60 | -7.47 | 345.9 | 410.4 |
| Seed 1011 | 200000 | -11.90 | -20.73 | 339.7 | 410.2 |

## 7. Final Test Cache Evaluation (Single-Pass Protocol)

### Pre-Registered Decision Rule
> *"MAPPO is reported as better than the severity-rule baseline only if the paired-difference interval excludes zero in the positive direction for that reward setting; if it excludes zero in the negative direction it is reported as worse; otherwise it is reported as not distinguishable."*

### Empirical Test Results (500 Episodes)

| Policy | Mean Return | 95% Bootstrap CI | Paired Diff vs Severity Rule | Decision Outcome | Attacks Contained (x of y) | Benign Disrupted (x of y) | Notifs / Ep |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random` | -260.08 | [-262.73, -257.50] | -- | -- | 8494 of 8616 (98.58% [98.31%, 98.81%]) | 15947 of 16384 (97.33% [97.07%, 97.57%]) | 33.16 |
| `always_none` | -92.75 | [-98.10, -87.49] | -- | -- | 0 of 8616 (0.00% [0.00%, 0.04%]) | 0 of 16384 (0.00% [0.00%, 0.02%]) | 0.0 |
| `max_containment` | -429.38 | [-432.18, -426.63] | -- | -- | 8616 of 8616 (100.00% [99.96%, 100.00%]) | 16384 of 16384 (100.00% [99.98%, 100.00%]) | 50.0 |
| `severity_rule` | -39.19 | [-40.99, -37.45] | -- | -- | 8328 of 8616 (96.66% [96.26%, 97.02%]) | 4097 of 16384 (25.01% [24.35%, 25.68%]) | 9.29 |
| `oracle` | -52.60 | [-54.14, -51.00] | -- | -- | 8616 of 8616 (100.00% [99.96%, 100.00%]) | 953 of 16384 (5.82% [5.47%, 6.19%]) | 17.23 |
| MAPPO Seed 42 | -6.41 | [-7.05, -5.80] | +32.78 [31.15, 34.46] | **better** | 7877 of 8616 (91.42% [90.81%, 92.00%]) | 737 of 16384 (4.50% [4.19%, 4.83%]) | 4.79 |
| MAPPO Seed 123 | -1.30 | [-1.98, -0.65] | +37.89 [35.99, 39.83] | **better** | 8347 of 8616 (96.88% [96.49%, 97.22%]) | 1530 of 16384 (9.34% [8.90%, 9.79%]) | 4.93 |
| MAPPO Seed 456 | -9.41 | [-10.09, -8.75] | +29.78 [28.28, 31.33] | **better** | 8004 of 8616 (92.90% [92.34%, 93.42%]) | 830 of 16384 (5.07% [4.74%, 5.41%]) | 4.64 |
| MAPPO Seed 789 | -6.12 | [-6.81, -5.46] | +33.07 [31.35, 34.75] | **better** | 8091 of 8616 (93.91% [93.38%, 94.39%]) | 753 of 16384 (4.60% [4.29%, 4.93%]) | 4.9 |
| MAPPO Seed 1011 | -9.48 | [-10.14, -8.82] | +29.71 [28.23, 31.25] | **better** | 7808 of 8616 (90.62% [89.99%, 91.22%]) | 794 of 16384 (4.85% [4.53%, 5.19%]) | 3.78 |
| **MAPPO Aggregate** | **-6.55** | **[-7.12, -6.00]** | **+32.65 [31.05, 34.30]** | **better** | **40127 of 43080 (93.15% [92.90%, 93.38%])** | **4644 of 81920 (5.67% [5.51%, 5.83%])** | **4.61** |

### Attack Family Containment Breakdown (Test Cache)

| Attack Family | Severity Rule Containment (x of y) | MAPPO Aggregate Containment (x of y) |
| :--- | :--- | :--- |
| `BruteForce` | 808 of 854 (94.61% [92.89%, 95.94%]) | 3789 of 4270 (88.74%) |
| `DDoS` | 1215 of 1215 (100.00% [99.68%, 100.00%]) | 6074 of 6075 (99.98%) |
| `DoS` | 1316 of 1316 (100.00% [99.71%, 100.00%]) | 6572 of 6580 (99.88%) |
| `Mirai` | 1343 of 1343 (100.00% [99.71%, 100.00%]) | 6714 of 6715 (99.99%) |
| `Recon` | 1195 of 1264 (94.54% [93.15%, 95.66%]) | 5516 of 6320 (87.28%) |
| `Spoofing` | 1145 of 1236 (92.64% [91.05%, 93.97%]) | 5298 of 6180 (85.73%) |
| `Web-based` | 1306 of 1388 (94.09% [92.73%, 95.22%]) | 6164 of 6940 (88.82%) |

## 8. Adversarial Stress Testing

To evaluate policy behavior under classifier evasion degradation, a subset of $500$ test flows was perturbed using the Phase 4 FGSM transfer attack at $\epsilon = 0.10$ with strict physical domain constraint enforcement. Under this evasion attack, the B2 ensemble classifier accuracy degraded from 95.60% down to 51.00%.

| Evaluated Arm | Clean Return | Adversarial Return | Return Delta | Clean Attacks Contained | Adversarial Attacks Contained | Containment Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Severity Rule Baseline | -39.19 | -33.86 | +5.33 | 96.66% | 63.46% | -33.20% |
| Max Containment Baseline | -429.38 | -429.38 | +0.00 | 100.00% | 100.00% | +0.00% |
| Oracle Policy | -52.60 | -52.60 | +0.00 | 100.00% | 100.00% | +0.00% |
| MAPPO Aggregate | -6.55 | -109.97 | -103.42 | 93.15% | 74.98% | -18.17% |

Adversarial perturbations alter feature values such that the underlying ensemble produces lower-confidence and incorrect class predictions, directly cascading into degraded tactical decisions. The aggregate MAPPO policy incurred a return change of -103.42, while the severity-rule baseline containment decreased by -33.20%.

## 9. Forensic Alert Logging, ECS Export, and Dashboard

1. **Forensic Alert Logging:** Extended `alerts/alert_logger.py` with MARL fields (`marl_triage_action`, `marl_containment_action`, `marl_escalation_action`, `marl_policy_version`). A new log `logs/alerts_phase5.jsonl` was generated for 45 demonstration flows with SHA-256 hash chaining and external tail anchor `logs/alerts_phase5_tail_hash.txt`.
2. **Cryptographic Verification Audit:** Verified via `alerts/verify_log.py` under three tests:
   - *Pristine Log Check:* **PASS** ($45$ of $45$ records verified).
   - *Content Modification Tamper Test:* **PASS** (tampering at line 16 detected).
   - *Tail Truncation Tamper Test:* **PASS** (dropped final record detected via external tail anchor mismatch).
3. **Elastic Common Schema (ECS) Export:** Implemented in `export/elk_export.py` to write `logs/alerts_phase5_ecs.ndjson`. An optional `docker-compose.yml` was provided for reference deployment.
4. **Autonomous SOC Dashboard:** Built by `dashboard/build_dashboard.py` as a standalone HTML file (`reports/dashboard.html`). Contains zero external CDN/script dependencies, embeds all five charts as Base64 PNGs, and was verified in headless browser execution.

## 10. Hardware, Runtime, and Peak Memory Audit

All Phase 5 workloads executed on a local Windows laptop CPU (8 GB physical RAM). Peak process memory was tracked via `psutil` and remained strictly below the 4.0 GB ceiling across all stages.

| Pipeline Stage | Task Description | Runtime (s) | Peak Process RSS (MB) |
| :--- | :--- | :--- | :--- |
| Stage 0 | Data Cache & Split Generation | 5.54 s | 607.26 MB |
| Stage 2 | Environment Unit Tests (`pytest`) | 1.58 s | 95.40 MB |
| Stage 3 | Heuristic & Oracle Baselines | 7.68 s | 122.00 MB |
| Stage 4 | MAPPO Pilot Benchmark | 42.08 s | 368.33 MB |
| Stage 4 | MAPPO 5-Seed Training | 1797.96 s | 410.41 MB |
| Stage 5 | Single-Pass Test Cache Evaluation | 103.85 s | 359.44 MB |
| Stage 6 | Adversarial Perturbation Stress Test | 104.71 s | 696.06 MB |
| Stage 7 | Alert Generation, ECS Export & Dashboard | 4.80 s | 135.20 MB |

## 11. Limitations

- **Simulated Environment:** Because CICIoT2023 lacks ground-truth packet timestamps and IP topology, campaign episodes are synthesized by sampling flows from cache partitions. This constitutes a controlled decision process rather than an actual live network deployment.
- **Reward Model Assumptions:** The reward weights, delay penalties, and collateral costs are defined assumptions. Policy optimality depends on these analytical parameters.
- **Flow Independence:** Flows within bursts are drawn from cache pools of the same attack family, assuming independent flow distributions within a burst rather than session-level state transitions.
- **Classifier Error Propagation:** Upstream classifier confusions and misclassifications directly degrade the observation vector presented to tactical agents.
- **Seed Count:** Results reflect 5 random seeds; broader variance across large initialization ensembles remains unexamined.
- **Out-of-Scope Infrastructure:** The ELK/Kibana cluster is provided as an export format and optional compose manifest, but was not run locally to adhere strictly to the 4.0 GB memory ceiling.

## 12. Skipped or Not Run

1. **Security-First and Availability-First MAPPO Training:** The Stage 4 pilot benchmark determined that training all 15 runs (3 settings $\times$ 5 seeds) would require an estimated 6312.0 seconds, exceeding the strict 45-minute limit (2700 s). As specified by the Stage 4 protocol, training was restricted to the `balanced` reward setting. Baselines were fully benchmarked across all 3 settings.
2. **Live Elasticsearch/Kibana Service Execution:** Marked optional and omitted due to local system memory constraints (requires $\ge 4$ GB dedicated RAM).
