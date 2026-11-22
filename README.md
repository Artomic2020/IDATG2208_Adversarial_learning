**Best clean baseline: 40.53% test accuracy** after 10 epochs on an NVIDIA GeForce RTX 4060 Laptop GPU (seed `42`).

## Checkpoint 1 · Dataset 1 (CIFAR-100) · Standard ResNet-18


| Epoch | Train loss | Train accuracy | Test accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 4.1699 | 5.01% | 6.35% |
| 2 | 3.8348 | 9.36% | 10.39% |
| 3 | 3.5492 | 14.39% | 14.97% |
| 4 | 3.2550 | 19.77% | 22.33% |
| 5 | 2.9587 | 25.19% | 19.38% |
| 6 | 2.6932 | 30.26% | 23.45% |
| 7 | 2.4737 | 35.01% | 30.01% |
| 8 | 2.3086 | 38.33% | 37.39% |
| 9 | 2.1816 | 41.27% | 37.28% |
| 10 | 2.0658 | 43.84% | **40.53%** |

Each epoch was evaluated on all 10,000 CIFAR-100 test images.

[Per-epoch results for all three runs (CSV)](results/resnet18_cifar100_runs.csv) 2.0658 | Train Acc: 43.84% | Test Acc:  40.53%
Test images counted: 10000
Runtime approx 8 minutes on Cuda RTX 4060 gompared to 55 minutes on i9 CPU

## CIFAR-100 — Full PGD-10 baseline

ResNet-18 with residual shortcuts enabled, trained on CUDA for 10 epochs
(3,910 batches) using seed 42. Training used 10 PGD steps per batch at
ε = 8/255. Robustness was evaluated separately using PGD-20 at the same
perturbation limit.

| Metric | Result |
|---|---:|
| Clean test accuracy | **36.50%** |
| PGD-20 robust test accuracy | **18.85%** |
| Training time | **32.94 min** |
| Whole-run elapsed time, including evaluation | **34.51 min** |

This run is the full PGD reference for the matched-training-time comparison
with early-stopped PGD and Spiking-PGD.
### Full PGD-10 training by epoch

| Epoch | Adversarial train loss | Adversarial train accuracy | Clean test accuracy | Cumulative training time |
|---:|---:|---:|---:|---:|
| 1 | 4.3430 | 5.11% | 11.07% | 3.30 min |
| 2 | 4.0569 | 8.45% | 15.42% | 6.67 min |
| 3 | 3.9213 | 10.46% | 19.15% | 9.93 min |
| 4 | 3.8080 | 12.14% | 21.97% | 13.16 min |
| 5 | 3.7130 | 13.84% | 26.65% | 16.56 min |
| 6 | 3.6185 | 15.25% | 28.12% | 19.94 min |
| 7 | 3.5423 | 16.53% | 29.71% | 23.24 min |
| 8 | 3.4804 | 17.62% | 31.86% | 26.49 min |
| 9 | 3.4254 | 18.69% | 34.39% | 29.73 min |
| 10 | 3.3884 | 19.04% | **36.50%** | **32.94 min** |

Each epoch processed 391 training batches. PGD-20 robust test accuracy was
measured after the final epoch and reached **18.85%** at ε = 8/255.
