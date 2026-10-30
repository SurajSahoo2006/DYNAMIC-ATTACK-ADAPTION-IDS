# Phase 9: Federated Learning Simulation

## Setup
- 5 simulated IoT gateway clients (non-IID class distribution)
- Max 8,000 rows per client; no data shared between clients
- Aggregation: FedAvg (tree pooling)

## Client Partition Summary
| Client | Dominant Class | Rows |
|:---|:---|---:|
| Client 0 | DDoS | 8,000 |
| Client 1 | DoS | 8,000 |
| Client 2 | Recon | 8,000 |
| Client 3 | Balanced | 8,000 |
| Client 4 | Benign | 4,000 |

## Local vs Global Performance on Full Test Set
| Model | Test Accuracy | Macro F1 |
|:---|---:|---:|
| Client 0 (DDoS) | 90.60% | 78.06% |
| Client 1 (DoS) | 90.83% | 78.36% |
| Client 2 (Recon) | 91.02% | 79.21% |
| Client 3 (Balanced) | 90.29% | 77.86% |
| Client 4 (Benign) | 87.16% | 72.86% |
| Global FedAvg | 91.55% | 80.50% |
| Centralized (Phase 3) | 95.23% | 86.97% |

## Key Finding
Global FedAvg achieves 91.55% accuracy vs 95.23% centralized (gap: 3.69%), demonstrating that collaborative training without sharing raw data retains most predictive performance.

## Limitations
- Tree pooling is not true gradient-based FedAvg; it is a simulation approximation suitable for tree ensembles
- Clients trained for one round only (no multi-round convergence)
- Non-IID partition is synthetic, not from real distributed IoT deployments
- No differential privacy or secure aggregation applied
