# Step 3.4 GMM Posterior and Stability

Scope: PC1-PC5 diagonal-covariance GMM, K=4-8. No K is finalized here.

## Posterior Quality
| K | BIC | AIC | mean max posterior | median max posterior | P10 max posterior | <0.60 | <0.70 | mean entropy | min cluster % | max cluster % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1414791.2 | 1414379.7 | 0.9640 | 0.9999 | 0.8703 | 2.47% | 4.88% | 0.0584 | 6.71% | 52.00% |
| 5 | 1324842.6 | 1324325.9 | 0.9705 | 1.0000 | 0.9125 | 1.93% | 3.79% | 0.0424 | 6.72% | 44.38% |
| 6 | 1261869.4 | 1261247.4 | 0.9716 | 0.9997 | 0.9167 | 1.44% | 2.79% | 0.0437 | 6.72% | 32.54% |
| 7 | 1228167.1 | 1227439.8 | 0.9449 | 0.9963 | 0.7924 | 3.41% | 6.43% | 0.0767 | 5.73% | 24.69% |
| 8 | 1154207.9 | 1153375.4 | 0.9499 | 0.9989 | 0.8021 | 3.17% | 6.17% | 0.0615 | 5.65% | 22.34% |

## Seed Stability
| K | mean ARI | min ARI | max ARI |
|---:|---:|---:|---:|
| 4 | 0.5535 | 0.2835 | 0.9199 |
| 5 | 0.4534 | 0.3454 | 0.6274 |
| 6 | 0.5138 | 0.3459 | 0.8160 |
| 7 | 0.5616 | 0.4399 | 0.6997 |
| 8 | 0.5469 | 0.3388 | 0.7463 |

## 80% Subsample Stability
| K | full mean ARI | full min ARI | subsample mean ARI | subsample min ARI |
|---:|---:|---:|---:|---:|
| 4 | 0.5753 | 0.4852 | 0.5751 | 0.4848 |
| 5 | 0.6470 | 0.4242 | 0.6464 | 0.4243 |
| 6 | 0.5370 | 0.3536 | 0.5363 | 0.3543 |
| 7 | 0.5034 | 0.4332 | 0.5034 | 0.4333 |
| 8 | 0.5890 | 0.5422 | 0.5890 | 0.5420 |

## Reading Notes
- High mean max posterior and low entropy indicate crisper GMM assignment.
- ARI is label-invariant; values near 1 mean nearly identical partitions.
- A K with good BIC but weak ARI or tiny clusters should not be finalized.