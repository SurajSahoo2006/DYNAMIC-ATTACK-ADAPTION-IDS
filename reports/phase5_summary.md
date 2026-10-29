# Phase 5 Summary Report: Multi-Agent Tactical Defense & Operational Dashboard

**Generated:** 2026-10-30 01:03:51 UTC  
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
- If true class is an attack flow (grounded in the true class severity from `get_true_class_severity(true_label)`):
  $$\text{damage\_term} = - w_D \cdot \text{BaseDamage}[\text{true\_severity}] \cdot \text{LeakMultiplier}[P_{\text{active}}]$$
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

The unit test suite in `marl/tests/test_env.py` executed under `pytest` with $6$ test categories and $100\%$ pass rate (recorded in `reports/marl_tests.txt`):
1. **No True Label Leakage:** Observation vectors remained identical under artificial ground-truth label mutations.
2. **Deterministic Seeding:** Trajectories generated under identical seeds matched bit-for-bit.
3. **Reward Arithmetic:** Exact verification of hand-computed test cases across reward weight configurations.
4. **Posture and Budget Constraints:** Verified $K=3$ step persistence and excess budget penalty trigger after $B=5$ actions.
5. **Split Boundary Isolation:** Verified disjoint index boundaries between `rl_train`, `rl_val`, and `test`.
6. **Damage Grounding Invariance:** Verified that mutating cached predicted class or severity while keeping true class fixed does not alter damage terms.

## 5. Benchmark Baselines Evaluation

Seven benchmark policies were evaluated over $200$ test episodes under the 'balanced' reward setting (recorded in `reports/marl_baselines.json`). Policies include heuristic baselines, rule-based baselines with and without budget constraints, and hand-written label-aware reference rules accessing ground truth (non-deployable, not an upper bound).

| Policy | Mean Return | 95% Bootstrap CI | Attacks Contained (x of y) | Benign Disrupted (x of y) | Notifications / Ep |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `random` | -265.72 | [-269.53, -262.01] | 3485 of 3544 (98.34% [97.86%, 98.71%]) | 6316 of 6456 (97.83% [97.45%, 98.16%]) | 33.18 |
| `always_none` | -115.82 | [-124.16, -107.62] | 0 of 3544 (0.00% [0.00%, 0.11%]) | 0 of 6456 (0.00% [0.00%, 0.06%]) | 0.0 |
| `max_containment` | -425.96 | [-430.30, -421.48] | 3544 of 3544 (100.00% [99.89%, 100.00%]) | 6456 of 6456 (100.00% [99.94%, 100.00%]) | 50.0 |
| `severity_rule` | -49.65 | [-52.16, -47.22] | 3443 of 3544 (97.15% [96.55%, 97.65%]) | 1579 of 6456 (24.46% [23.42%, 25.52%]) | 10.21 |
| `severity_rule_budget_aware` | -28.60 | [-30.32, -26.91] | 3443 of 3544 (97.15% [96.55%, 97.65%]) | 1579 of 6456 (24.46% [23.42%, 25.52%]) | 3.2 |
| `label_aware_rule` | -54.42 | [-56.96, -51.95] | 3544 of 3544 (100.00% [99.89%, 100.00%]) | 377 of 6456 (5.84% [5.29%, 6.44%]) | 17.72 |
| `label_aware_rule_budget_aware` | -14.73 | [-15.27, -14.20] | 3544 of 3544 (100.00% [99.89%, 100.00%]) | 377 of 6456 (5.84% [5.29%, 6.44%]) | 4.05 |

> **Budget-Aware Severity Rule Finding:** Comparing `severity_rule` to `severity_rule_budget_aware` illustrates the impact of budget management: `severity_rule_budget_aware` achieves a mean return of -28.60 versus -49.65 for unconstrained `severity_rule`. By halting analyst notifications once the episode budget $B=5$ is exhausted, it avoids repeated excess budget penalties ($-5.0$ per step) while preserving budget for Critical severity threats.

## 6. MAPPO Training Protocol and Pilot Benchmark

A pilot run of $20000$ steps completed in $97.92$ seconds ($204.26$ steps/second). In accordance with the pre-registered budget formula `budget = min(MAX_TRAIN_STEPS, steps_per_sec * 1500s)` ($\ge 100,000$ steps), the chosen training budget was $200000$ steps per seed.

> **Training Budget & Setting Allocation:** Measuring 204.26 steps/second yields an estimated runtime of 14687.2 seconds for 15 runs (3 settings $\times$ 5 seeds), which exceeds the 45-minute ceiling (2700 s). Furthermore, training two settings (8 runs: 5 balanced + 3 security-first) would require approximately 7833.2 seconds, which exceeds the 75-minute limit (4500 s). Consequently, multi-seed training was conducted on the primary `balanced` setting across 5 random seeds. Reason documented in `reports/marl_pilot.json`: *"All 3 settings x 5 seeds (14687.2s) and 2 settings (7833.2s) exceed 75-minute limit (4500s). Trained 'balanced' setting only (5 seeds)."*.

### Multi-Seed Training Convergence (`balanced` Setting)

| Training Seed | Budget (Steps) | Best `rl_val` Return | Final `rl_val` Return | Runtime (s) | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Seed 42 | 200000 | -3.14 | -3.14 | 928.6 | 112.4 |
| Seed 123 | 200000 | -4.55 | -4.55 | 821.7 | 84.5 |
| Seed 456 | 200000 | -8.18 | -8.69 | 400.4 | 110.3 |
| Seed 789 | 200000 | -10.38 | -22.52 | 367.0 | 109.5 |
| Seed 1011 | 200000 | -8.12 | -8.12 | 380.6 | 108.6 |

## 7. Final Test Cache Evaluation (Single-Pass Protocol)

### Pre-Registered Decision Rule
> *"MAPPO is reported as better than the best non-learned baseline only if the paired-difference interval excludes zero in the positive direction for that reward setting; if it excludes zero in the negative direction it is reported as worse; otherwise it is reported as not distinguishable."*

For the `balanced` setting, the best non-learned baseline is **`severity_rule_budget_aware`** (Mean Return: -29.30).

### Empirical Test Results (500 Episodes)

| Policy | Mean Return | 95% Bootstrap CI | Paired Diff vs Best Non-Learned | Decision Outcome | Attacks Contained (x of y) | Benign Disrupted (x of y) | Notifs / Ep |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random` | -264.07 | [-266.67, -261.57] | -- | -- | 8494 of 8616 (98.58% [98.31%, 98.81%]) | 15947 of 16384 (97.33% [97.07%, 97.57%]) | 33.16 |
| `always_none` | -108.86 | [-113.71, -104.11] | -- | -- | 0 of 8616 (0.00% [0.00%, 0.04%]) | 0 of 16384 (0.00% [0.00%, 0.02%]) | 0.0 |
| `max_containment` | -429.38 | [-432.18, -426.63] | -- | -- | 8616 of 8616 (100.00% [99.96%, 100.00%]) | 16384 of 16384 (100.00% [99.98%, 100.00%]) | 50.0 |
| `severity_rule` | -47.67 | [-49.19, -46.14] | -- | -- | 8328 of 8616 (96.66% [96.26%, 97.02%]) | 4097 of 16384 (25.01% [24.35%, 25.68%]) | 9.29 |
| `severity_rule_budget_aware` | -29.30 | [-30.53, -28.04] | -- | -- | 8328 of 8616 (96.66% [96.26%, 97.02%]) | 4097 of 16384 (25.01% [24.35%, 25.68%]) | 3.11 |
| `label_aware_rule` | -52.60 | [-54.14, -51.00] | -- | -- | 8616 of 8616 (100.00% [99.96%, 100.00%]) | 953 of 16384 (5.82% [5.47%, 6.19%]) | 17.23 |
| `label_aware_rule_budget_aware` | -14.56 | [-14.91, -14.20] | -- | -- | 8616 of 8616 (100.00% [99.96%, 100.00%]) | 953 of 16384 (5.82% [5.47%, 6.19%]) | 4.02 |
| MAPPO Seed 42 | -5.01 | [-5.82, -4.21] | +24.29 [23.04, 25.52] | **better** | 8435 of 8616 (97.90% [97.57%, 98.18%]) | 1987 of 16384 (12.13% [11.64%, 12.64%]) | 4.35 |
| MAPPO Seed 123 | -6.43 | [-7.16, -5.73] | +22.87 [21.48, 24.24] | **better** | 8376 of 8616 (97.21% [96.85%, 97.54%]) | 1350 of 16384 (8.24% [7.83%, 8.67%]) | 4.52 |
| MAPPO Seed 456 | -7.86 | [-8.53, -7.21] | +21.44 [20.22, 22.69] | **better** | 8307 of 8616 (96.41% [96.00%, 96.79%]) | 1617 of 16384 (9.87% [9.42%, 10.34%]) | 4.89 |
| MAPPO Seed 789 | -11.23 | [-11.98, -10.50] | +18.07 [16.64, 19.49] | **better** | 8381 of 8616 (97.27% [96.91%, 97.60%]) | 948 of 16384 (5.79% [5.44%, 6.15%]) | 4.28 |
| MAPPO Seed 1011 | -7.82 | [-8.47, -7.19] | +21.48 [20.28, 22.66] | **better** | 8279 of 8616 (96.09% [95.66%, 96.48%]) | 721 of 16384 (4.40% [4.10%, 4.73%]) | 4.32 |
| **MAPPO Aggregate** | **-7.67** | **[-8.27, -7.11]** | **+21.63 [20.39, 22.85]** | **better** | **41778 of 43080 (96.98% [96.81%, 97.14%])** | **6623 of 81920 (8.08% [7.90%, 8.27%])** | **4.47** |

### Interpretation of Test Evaluation Results

The aggregate MAPPO policy achieves a mean episode return of -7.67 compared to -29.30 for `severity_rule_budget_aware`. The paired-difference 95% bootstrap confidence interval [20.39, 22.85] excludes zero in the positive direction. In accordance with the pre-registered decision rule, MAPPO is reported as **better** than the best non-learned baseline.

Importantly, policy returns depend directly on the analytical cost weights $(w_D, w_C, w_{cost}, w_B)$ and specific penalty structures defined in `config_marl.py`. If operational environments prioritize analyst attention or collateral disruption differently, policy preference orderings may shift accordingly.

### Attack Family Containment Breakdown (Test Cache)

| Attack Family | Severity Rule Containment (x of y) | MAPPO Aggregate Containment (x of y) |
| :--- | :--- | :--- |
| `BruteForce` | 808 of 854 (94.61% [92.89%, 95.94%]) | 4045 of 4270 (94.73% [94.02%, 95.36%]) |
| `DDoS` | 1215 of 1215 (100.00% [99.68%, 100.00%]) | 6066 of 6075 (99.85% [99.72%, 99.92%]) |
| `DoS` | 1316 of 1316 (100.00% [99.71%, 100.00%]) | 6565 of 6580 (99.77% [99.62%, 99.86%]) |
| `Mirai` | 1343 of 1343 (100.00% [99.71%, 100.00%]) | 6713 of 6715 (99.97% [99.89%, 99.99%]) |
| `Recon` | 1195 of 1264 (94.54% [93.15%, 95.66%]) | 6000 of 6320 (94.94% [94.37%, 95.45%]) |
| `Spoofing` | 1145 of 1236 (92.64% [91.05%, 93.97%]) | 5692 of 6180 (92.10% [91.40%, 92.75%]) |
| `Web-based` | 1306 of 1388 (94.09% [92.73%, 95.22%]) | 6697 of 6940 (96.50% [96.04%, 96.91%]) |

## 8. Adversarial Stress Testing (Paired Like-for-Like Protocol)

To evaluate tactical defense behavior when classifier inferences suffer adversarial degradation, a stratified sample of $500$ test flows was evaluated under both clean conditions and physically-constrained FGSM evasion perturbations ($\epsilon = 0.10$). Under this attack, B2 ensemble accuracy degraded from 95.60% down to 51.00%.

Every policy was evaluated on the exact same 500 perturbed flows and the clean version of those same 500 flows (like-for-like paired protocol over 500 episodes).

| Evaluated Policy | Clean Return (95% CI) | Adversarial Return (95% CI) | Paired Diff (adv - clean) | Attacks Contained (Clean -> Adv) | Benign Disrupted (Clean -> Adv) | Notifs / Ep (Clean -> Adv) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `always_none` | -106.50 [-111.29, -101.95] | -106.50 [-111.29, -101.95] | 0.00 [0.00, 0.00] | 0.0% [0.0%, 0.0%] -> 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] -> 0.0% [0.0%, 0.0%] | 0.00 -> 0.00 |
| `severity_rule` | -44.83 [-46.37, -43.32] | -74.49 [-76.78, -72.30] | -29.66 [-31.42, -28.00] | 95.5% [95.1%, 95.9%] -> 61.0% [59.9%, 62.0%] | 5.1% [4.8%, 5.4%] -> 66.5% [65.8%, 67.3%] | 9.10 -> 3.25 |
| `severity_rule_budget_aware` | -27.02 [-28.36, -25.75] | -68.61 [-71.37, -65.90] | -41.59 [-44.01, -39.23] | 95.5% [95.1%, 95.9%] -> 61.0% [59.9%, 62.0%] | 5.1% [4.8%, 5.4%] -> 66.5% [65.8%, 67.3%] | 3.05 -> 1.43 |
| `max_containment` | -429.07 [-431.91, -426.23] | -429.07 [-431.91, -426.23] | 0.00 [0.00, 0.00] | 100.0% [100.0%, 100.0%] -> 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] -> 100.0% [100.0%, 100.0%] | 50.00 -> 50.00 |
| `label_aware_rule` | -52.88 [-54.47, -51.28] | -52.88 [-54.47, -51.28] | 0.00 [0.00, 0.00] | 100.0% [100.0%, 100.0%] -> 100.0% [100.0%, 100.0%] | 5.9% [5.5%, 6.3%] -> 5.9% [5.5%, 6.3%] | 17.28 -> 17.28 |
| `label_aware_rule_budget_aware` | -14.76 [-15.10, -14.42] | -14.76 [-15.10, -14.42] | 0.00 [0.00, 0.00] | 100.0% [100.0%, 100.0%] -> 100.0% [100.0%, 100.0%] | 5.9% [5.5%, 6.3%] -> 5.9% [5.5%, 6.3%] | 3.98 -> 3.98 |
| **MAPPO Aggregate** | **-4.55 [-4.87, -4.24]** | **-150.98 [-152.85, -149.07]** | **-146.43 [-148.44, -144.37]** | **96.7% [96.5%, 96.8%] -> 94.2% [93.9%, 94.4%]** | **7.0% [6.9%, 7.2%] -> 90.2% [90.0%, 90.5%]** | **4.48 -> 4.47** |

### Degradation Mechanism and Comparative Assessment Under Attack

An audit of the underlying flow classification reveals the precise operational failure mechanism:
- **Benign Flow Inversion:** Out of 19 benign flows in the test sample, 17 (89.47%) were inverted by FGSM into attack classes. Classifier accuracy on benign traffic dropped from 100.00% to 10.53%.
- **Severity Escalation:** Inverted benign flows were mapped into elevated severities (e.g. Critical: 0, High: 0, Medium: 6), causing containment agents to trigger active blocking postures on legitimate traffic.
- **Budget Depletion:** Increased false positive alarms rapidly depleted the escalation budget ($B = 5$), incurring repeated excess budget penalties on subsequent alerts.
- **Attack Evasion:** Simultaneously, 2 of 481 (0.42%) attack flows evaded detection as Benign, leaking past containment unmitigated.

Under adversarial attack, the MAPPO aggregate return dropped to -150.98. In comparison, the inaction baseline (`always_none`) achieved an adversarial return of -106.50, and the best non-learned baseline under attack (`severity_rule_budget_aware`) scored -68.61. 
Under adversarial evasion, MAPPO scores lower than the do-nothing (`always_none`) baseline (-150.98 vs -106.50, delta -44.48). Because the evasion attack tricks the classifier into mislabeling benign traffic as high-severity attacks, the learned policy aggressively blocks legitimate flows and exhausts its escalation budget, accumulating severe collateral and excess budget penalties that exceed the damage of inaction.

## 9. Forensic Alert Logging, ECS Export, and Dashboard

1. **Forensic Alert Logging:** Extended `alerts/alert_logger.py` with MARL fields (`marl_triage_action`, `marl_containment_action`, `marl_escalation_action`, `marl_policy_version`). A new log `logs/alerts_phase5.jsonl` was generated for 45 demonstration flows with SHA-256 hash chaining and external tail anchor `logs/alerts_phase5_tail_hash.txt`.
2. **Cryptographic Verification Audit:** Verified via `alerts/verify_log.py` under three tests:
   - *Pristine Log Check:* **PASS** ($45$ of $45$ records verified).
   - *Content Modification Tamper Test:* **PASS** (tampering at line 16 detected).
   - *Tail Truncation Tamper Test:* **PASS** (dropped final record detected via external tail anchor mismatch).
3. **Elastic Common Schema (ECS) Export:** Implemented in `export/elk_export.py` to write `logs/alerts_phase5_ecs.ndjson`. An optional `docker-compose.yml` was provided for reference deployment.
4. **Autonomous SOC Dashboard:** Built by `dashboard/build_dashboard.py` as a standalone HTML file (`reports/dashboard.html`). Contains zero external CDN/script dependencies, embeds all charts as Base64 PNGs, and was verified in headless browser execution.

## 10. Hardware, Runtime, and Peak Memory Audit

All Phase 5 workloads executed on a local Windows laptop CPU (8 GB physical RAM). Peak process memory was tracked via `psutil` and remained strictly below the 4.0 GB ceiling across all stages.

| Pipeline Stage | Task Description | Runtime (s) | Peak Process RSS (MB) |
| :--- | :--- | :--- | :--- |
| Stage 0 | Data Cache & Split Generation | 5.54 s | 607.26 MB |
| Stage 2 | Environment Unit Tests (`pytest`) | 1.49 s | 95.40 MB |
| Stage 3 | Heuristic & Rule Baselines | 31.92 s | 128.80 MB |
| Stage 4 | MAPPO Pilot Benchmark | 97.92 s | 366.98 MB |
| Stage 4 | MAPPO Multi-Seed Training | 2898.34 s | 112.38 MB |
| Stage 5 | Single-Pass Test Cache Evaluation | 160.75 s | 297.30 MB |
| Stage 6 | Adversarial Stress Test (Paired Protocol) | 299.80 s | 51.83 MB |
| Stage 7 | Alert Generation, ECS Export & Dashboard | 4.80 s | 135.20 MB |

## 11. Limitations

- **Simulated Environment:** Because CICIoT2023 lacks ground-truth packet timestamps and IP topology, campaign episodes are synthesized by sampling flows from cache partitions. This constitutes a controlled decision process rather than an actual live network deployment.
- **Reward Model Assumptions:** The reward weights, delay penalties, and collateral costs are defined assumptions. Policy optimality depends on these analytical parameters.
- **Flow Independence:** Flows within bursts are drawn from cache pools of the same attack family, assuming independent flow distributions within a burst rather than session-level state transitions.
- **Classifier Error Propagation:** Upstream classifier confusions and misclassifications directly degrade the observation vector presented to tactical agents.
- **Seed Count:** Results reflect 5 random seeds; broader variance across large initialization ensembles remains unexamined.
- **Out-of-Scope Infrastructure:** The ELK/Kibana cluster is provided as an export format and optional compose manifest, but was not run locally to adhere strictly to the 4.0 GB memory ceiling.

## 12. Skipped or Not Run

1. **Security-First and Availability-First MAPPO Training:** The Stage 4 pilot benchmark determined that training all 15 runs (3 settings $\times$ 5 seeds) would require an estimated 14687.2 seconds, exceeding the strict 45-minute limit (2700 s). Training 2 settings (8 runs) would require 7833.2 s, exceeding the 75-minute limit (4500 s). In strict adherence to the protocol, training was restricted to the `balanced` reward setting. Baselines were fully benchmarked across all 3 settings.
2. **Live Elasticsearch/Kibana Service Execution:** Marked optional and omitted due to local system memory constraints (requires $\ge 4$ GB dedicated RAM).
