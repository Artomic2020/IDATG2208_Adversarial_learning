### Checkpoint 1 · CIFAR-100 · Initial standard-training run

The initial standard-training run reached **40.53% clean test accuracy**
after 10 epochs (seed 42, NVIDIA GeForce RTX 4060 Laptop GPU).
Evaluation used all 10,000 CIFAR-100 test images.

| Training method | Batches trained | Clean accuracy | Robust accuracy |
|---|---:|---:|---:|
| Full PGD‑10 | 3,910 | 36.50% | **18.85%** |
| Early-stopped PGD‑3 | 9,261 | **47.68%** | 16.38% |
| Spiking‑PGD (ρ = 0.02) | 3,005 | 31.17% | 16.51% |
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

### Spiking-PGD training by epoch

ResNet-18 was trained with 10 PGD steps per batch, ε = 8/255, and a
spiking threshold of ρ = 0.02. The run completed seven full epochs and
268 of 391 batches in epoch 8.

| Epoch | Adversarial train loss | Adversarial train accuracy | Clean test accuracy | Cumulative training time | Layer recompute fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.3382 | 5.07% | 9.23% | 3.74 min | 37.93% |
| 2 | 4.0796 | 8.01% | 14.19% | 8.10 min | 35.90% |
| 3 | 3.9443 | 9.99% | 18.51% | 12.57 min | 37.10% |
| 4 | 3.8324 | 11.54% | 21.46% | 17.03 min | 36.69% |
| 5 | 3.7417 | 13.10% | 24.56% | 21.47 min | 38.11% |
| 6 | 3.6499 | 14.62% | 26.55% | 25.94 min | 39.53% |
| 7 | 3.5643 | 16.09% | 29.27% | 30.33 min | 40.38% |
| 8 (partial) | 3.5033 | 16.95% | **31.17%** | **32.94 min** | 40.63% |

Final PGD-20 robust test accuracy was **16.51%**. The recompute fraction
tracks how often the instrumented layers recomputed their outputs during
PGD. Despite reusing outputs, this Spiking-PGD run processed **3,005
batches**, compared with **3,910** for full PGD-10 under the same
training-time budget. At ρ = 0.02, layer reuse did not translate into
higher training throughput on this setup.
