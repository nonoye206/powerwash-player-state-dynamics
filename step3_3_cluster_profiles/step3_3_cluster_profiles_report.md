# Step 3.3 Candidate Cluster Profiles

No state names are assigned here. Clusters are described only by numeric profiles.

## Candidate Metrics
| candidate | silhouette | CH | DB | BIC | AIC | min cluster % | max cluster % |
|---|---:|---:|---:|---:|---:|---:|---:|
| main_standardized_kmeans_k8 | 0.2949 | 29073.7 | 1.2929 |  |  | 2.47% | 25.20% |
| pc1_pc5_gmm_diag_k5 | 0.4055 | 61832.0 | 0.8670 | 1395857.1 | 1395340.4 | 6.76% | 44.14% |
| pc1_pc5_kmeans_k6 | 0.3248 | 49657.7 | 1.1470 |  |  | 7.16% | 33.67% |
| pc1_pc5_kmeans_k7 | 0.3667 | 55312.7 | 0.9129 |  |  | 3.76% | 29.82% |

## Cluster Profiles
### main_standardized_kmeans_k8
| cluster | size | pct | duration med | gap med | jobs started med | jobs resumed med | tasks med | purchase rate | completion med | exit med | progression med | mode diversity med | top mode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 7534 | 7.12% | 0.65 | 2.57 | 0.00 | 0.00 | 0.00 | 0.76% | 0.00 | 0.00 | 0.00 | 1.00 | (missing) (98.9%) |
| 1 | 11887 | 11.23% | 30.68 | 0.85 | 1.00 | 0.00 | 0.00 | 8.50% | 0.00 | 1.00 | 0.00 | 1.00 | Career (52.5%) |
| 2 | 8466 | 8.00% | 59.85 | nan | 2.00 | 0.00 | 2.00 | 45.19% | 0.75 | 1.00 | 1.00 | 1.00 | Career (98.2%) |
| 3 | 2612 | 2.47% | 105.58 | 0.72 | 4.00 | 0.00 | 4.00 | 52.99% | 0.75 | 1.00 | 1.00 | 2.00 | Career (67.3%) |
| 4 | 14891 | 14.07% | 57.48 | 0.76 | 1.00 | 1.00 | 1.00 | 7.04% | 0.67 | 1.00 | 1.00 | 1.00 | Career (97.7%) |
| 5 | 26671 | 25.20% | 122.33 | 0.74 | 2.00 | 1.00 | 3.00 | 80.99% | 0.75 | 1.00 | 1.00 | 1.00 | Career (97.2%) |
| 6 | 23245 | 21.97% | 28.95 | 0.73 | 0.00 | 1.00 | 0.00 | 10.04% | 0.00 | 1.00 | 0.19 | 1.00 | Career (99.2%) |
| 7 | 10512 | 9.93% | 62.63 | 0.79 | 2.00 | 0.00 | 1.00 | 9.17% | 1.00 | 1.00 | 0.00 | 1.00 | Career (52.3%) |

### pc1_pc5_gmm_diag_k5
| cluster | size | pct | duration med | gap med | jobs started med | jobs resumed med | tasks med | purchase rate | completion med | exit med | progression med | mode diversity med | top mode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 8645 | 8.17% | 61.48 | 0.16 | 2.00 | 0.00 | 2.00 | 46.91% | 0.75 | 1.00 | 1.00 | 1.00 | Career (98.2%) |
| 1 | 46709 | 44.14% | 86.83 | 0.78 | 2.00 | 1.00 | 2.00 | 48.08% | 0.71 | 1.00 | 1.00 | 1.00 | Career (91.6%) |
| 2 | 35784 | 33.82% | 28.73 | 0.74 | 0.00 | 1.00 | 0.00 | 7.18% | 0.00 | 1.00 | 0.06 | 1.00 | Career (81.6%) |
| 3 | 7158 | 6.76% | 92.15 | 0.74 | 3.00 | 1.00 | 3.00 | 43.11% | 0.60 | 1.00 | 1.00 | 2.00 | Career (67.8%) |
| 4 | 7522 | 7.11% | 0.65 | 2.59 | 0.00 | 0.00 | 0.00 | 0.72% | 0.00 | 0.00 | 0.00 | 1.00 | (missing) (99.0%) |

### pc1_pc5_kmeans_k6
| cluster | size | pct | duration med | gap med | jobs started med | jobs resumed med | tasks med | purchase rate | completion med | exit med | progression med | mode diversity med | top mode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 12608 | 11.91% | 181.21 | 0.81 | 4.00 | 1.00 | 5.00 | 90.69% | 0.84 | 1.00 | 1.00 | 1.00 | Career (94.5%) |
| 1 | 20684 | 19.55% | 36.69 | 1.11 | 1.00 | 0.00 | 0.00 | 6.20% | 0.00 | 1.00 | 0.00 | 1.00 | Career (55.6%) |
| 2 | 35633 | 33.67% | 72.92 | 0.69 | 1.00 | 1.00 | 2.00 | 39.04% | 0.67 | 1.00 | 1.00 | 1.00 | Career (93.7%) |
| 3 | 19732 | 18.65% | 26.82 | 0.54 | 0.00 | 1.00 | 0.00 | 6.88% | 0.00 | 1.00 | 0.18 | 1.00 | Career (99.1%) |
| 4 | 9580 | 9.05% | 60.49 | 0.13 | 2.00 | 0.00 | 2.00 | 43.64% | 0.75 | 1.00 | 1.00 | 1.00 | Career (93.5%) |
| 5 | 7581 | 7.16% | 0.65 | 2.55 | 0.00 | 0.00 | 0.00 | 0.78% | 0.00 | 0.00 | 0.00 | 1.00 | (missing) (98.3%) |

### pc1_pc5_kmeans_k7
| cluster | size | pct | duration med | gap med | jobs started med | jobs resumed med | tasks med | purchase rate | completion med | exit med | progression med | mode diversity med | top mode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 21734 | 20.54% | 51.72 | 0.92 | 1.00 | 0.00 | 1.00 | 14.83% | 0.50 | 1.00 | 0.00 | 1.00 | Career (58.5%) |
| 1 | 6891 | 6.51% | 92.15 | 0.70 | 3.00 | 1.00 | 2.00 | 42.78% | 0.57 | 1.00 | 1.00 | 2.00 | Career (68.1%) |
| 2 | 24807 | 23.44% | 25.58 | 0.65 | 0.00 | 1.00 | 0.00 | 7.22% | 0.00 | 1.00 | 0.17 | 1.00 | Career (95.8%) |
| 3 | 31555 | 29.82% | 88.47 | 0.65 | 1.00 | 1.00 | 2.00 | 54.07% | 0.67 | 1.00 | 1.00 | 1.00 | Career (100.0%) |
| 4 | 7583 | 7.17% | 0.65 | 2.56 | 0.00 | 0.00 | 0.00 | 0.78% | 0.00 | 0.00 | 0.00 | 1.00 | (missing) (98.3%) |
| 5 | 3980 | 3.76% | 184.75 | 2.72 | 4.00 | 0.00 | 5.00 | 80.53% | 1.00 | 1.00 | 1.00 | 1.00 | Career (92.1%) |
| 6 | 9268 | 8.76% | 56.12 | 0.00 | 2.00 | 0.00 | 2.00 | 42.45% | 0.75 | 1.00 | 1.00 | 1.00 | Career (96.3%) |

## Reading Notes
- `purchase rate` is the mean of `purchase_any`, not raw purchase count.
- Median gap is blank/NaN for first-session-heavy clusters only in the detailed CSV; Markdown renders numeric medians when available.
- Use this file to decide which configuration is interpretable enough before assigning state names.