**Best clean baseline: 40.53% test accuracy** after 10 epochs on an NVIDIA GeForce RTX 4060 Laptop GPU (seed `42`).

## RTX 4060 results · Seed 42

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


