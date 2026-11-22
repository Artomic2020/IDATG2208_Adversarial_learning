### Checkpoint 1 · CIFAR-100 · Initial standard-training run

The initial standard-training run reached **40.53% clean test accuracy**
after 10 epochs (seed 42, NVIDIA GeForce RTX 4060 Laptop GPU).
Evaluation used all 10,000 CIFAR-100 test images.

| Training method | Training batches | Clean test accuracy | PGD‑20 robust accuracy at ε = 8/255 | Training time |
|---|---:|---:|---:|---:|
| Full PGD‑10 | 3,910 | 36.50% | **18.85%** | 32.94 min |
| Early-stopped PGD‑3 | 9,261 | **47.68%** | 16.38% | 32.94 min |
| Spiking‑PGD | Pending | Pending | Pending | Pending |

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

### CIFAR-100 — Early-stopped PGD-3

ResNet-18 was trained with 3 PGD steps per batch at ε = 8/255, using seed 42.
The run processed 9,261 batches in 32.94 minutes of training. It completed
23 full epochs and 268 of 391 batches in epoch 24. Whole-run time,
including evaluation, was 35.12 minutes.

| Epoch | Adversarial train loss | Adversarial train accuracy | Clean test accuracy | Cumulative training time |
|---:|---:|---:|---:|---:|
| 1 | 4.2590 | 5.67% | 11.20% | 1.35 min |
| 5 | 3.3911 | 18.62% | 30.58% | 6.90 min |
| 10 | 2.9506 | 26.24% | 41.31% | 13.88 min |
| 15 | 2.7599 | 29.76% | 45.01% | 20.75 min |
| 20 | 2.6484 | 31.83% | 46.46% | 27.74 min |
| 24 (partial) | 2.5640 | 33.33% | **47.68%** | **32.94 min** |

PGD-20 robust test accuracy was measured at the end of training and reached
**16.38%** at ε = 8/255. At matched training time, PGD-3 gained 11.18
percentage points in clean accuracy and lost 2.47 points in robust accuracy
relative to full PGD-10.


