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


## CIFAR-100 — Full PGD-10 adversarial training

Preliminary results for ResNet-18 after 10 training epochs. Adversarial training used 10 PGD steps per batch with a perturbation limit of ε = 8/255.

| Metric | Final result |
|---|---:|
| Adversarial training loss | 3.6939 |
| Adversarial training accuracy | 14.27% |
| Clean test accuracy (10,000 images) | **23.98%** |
| PGD-20 robust test accuracy | Pending evaluation |
| Whole-run elapsed time | 31.21 min |

The elapsed time includes training, clean evaluation, and checkpoint saves. The matched-compute comparison with early-stopped PGD and Spiking-PGD is pending.

