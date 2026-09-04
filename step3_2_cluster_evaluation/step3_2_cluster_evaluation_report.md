# Step 3.2 Cluster Evaluation

No state names assigned in this step.

## Scope
- Main space: original standardized discovery features, 12 dimensions.
- Control space: PC1-PC5 PCA scores.
- Models: KMeans and diagonal-covariance GMM, K=2-8.
- Silhouette: exact on a fixed random sample of 6,000 sessions; CH/DB and cluster sizes use all sessions.

## Metrics
### main_standardized
| model | K | silhouette | CH | DB | BIC | AIC |
|---|---:|---:|---:|---:|---:|---:|
| kmeans | 2 | 0.2406 | 33400.0 | 1.5948 |  |  |
| gmm_diag | 2 | 0.4615 | 33034.5 | 0.7336 | 1553665.4 | 1553196.4 |
| kmeans | 3 | 0.2459 | 26472.1 | 1.5972 |  |  |
| gmm_diag | 3 | 0.2647 | 33981.8 | 1.4109 | 899662.6 | 898954.5 |
| kmeans | 4 | 0.2601 | 33388.8 | 1.6328 |  |  |
| gmm_diag | 4 | 0.2525 | 33263.5 | 1.5931 | 682228.5 | 681281.1 |
| kmeans | 5 | 0.2515 | 28529.9 | 1.5477 |  |  |
| gmm_diag | 5 | 0.2811 | 33520.6 | 1.2030 | 273897.6 | 272710.9 |
| kmeans | 6 | 0.1825 | 19082.5 | 1.6943 |  |  |
| gmm_diag | 6 | 0.2284 | 27899.3 | 1.7024 | -542942.4 | -544368.3 |
| kmeans | 7 | 0.2240 | 24553.8 | 1.6344 |  |  |
| gmm_diag | 7 | 0.2874 | 29501.1 | 1.2951 | -1106841.2 | -1108506.3 |
| kmeans | 8 | 0.2949 | 29073.7 | 1.2929 |  |  |
| gmm_diag | 8 | 0.2941 | 26388.9 | 1.3259 | -1134572.7 | -1136477.0 |

### pc1_pc5
| model | K | silhouette | CH | DB | BIC | AIC |
|---|---:|---:|---:|---:|---:|---:|
| kmeans | 2 | 0.3419 | 50393.3 | 1.2631 |  |  |
| gmm_diag | 2 | 0.5377 | 44154.5 | 0.5943 | 1629980.7 | 1629779.8 |
| kmeans | 3 | 0.3564 | 61291.6 | 0.9732 |  |  |
| gmm_diag | 3 | 0.3515 | 58423.6 | 0.8963 | 1517339.4 | 1517033.1 |
| kmeans | 4 | 0.3742 | 54847.5 | 0.9176 |  |  |
| gmm_diag | 4 | 0.3787 | 55429.7 | 0.9847 | 1457263.0 | 1456851.6 |
| kmeans | 5 | 0.3782 | 43894.8 | 0.8052 |  |  |
| gmm_diag | 5 | 0.4055 | 61832.0 | 0.8670 | 1395857.1 | 1395340.4 |
| kmeans | 6 | 0.3248 | 49657.7 | 1.1470 |  |  |
| gmm_diag | 6 | 0.3505 | 47481.7 | 1.0563 | 1237842.1 | 1237220.1 |
| kmeans | 7 | 0.3667 | 55312.7 | 0.9129 |  |  |
| gmm_diag | 7 | 0.3805 | 51206.7 | 1.1101 | 1208136.2 | 1207408.9 |
| kmeans | 8 | 0.3663 | 52249.5 | 0.9965 |  |  |
| gmm_diag | 8 | 0.3293 | 50786.7 | 0.9313 | 1179046.0 | 1178213.5 |

## Cluster Size Notes
- main_standardized / kmeans: min cluster pct by K = K2:40.93%, K3:11.90%, K4:7.18%, K5:1.20%, K6:5.33%, K7:2.86%, K8:2.47%
- main_standardized / kmeans: max cluster pct by K = K2:59.07%, K3:49.02%, K4:38.88%, K5:31.73%, K6:26.45%, K7:26.61%, K8:25.20%
- main_standardized / gmm_diag: min cluster pct by K = K2:7.24%, K3:7.12%, K4:7.11%, K5:7.11%, K6:0.19%, K7:0.33%, K8:1.52%
- main_standardized / gmm_diag: max cluster pct by K = K2:92.76%, K3:64.22%, K4:35.11%, K5:45.60%, K6:49.00%, K7:28.31%, K8:35.72%
- pc1_pc5 / kmeans: min cluster pct by K = K2:39.24%, K3:7.15%, K4:5.42%, K5:0.69%, K6:7.16%, K7:3.76%, K8:0.55%
- pc1_pc5 / kmeans: max cluster pct by K = K2:60.76%, K3:48.18%, K4:51.37%, K5:54.59%, K6:33.67%, K7:29.82%, K8:32.86%
- pc1_pc5 / gmm_diag: min cluster pct by K = K2:7.09%, K3:7.10%, K4:7.12%, K5:6.76%, K6:6.48%, K7:0.58%, K8:3.03%
- pc1_pc5 / gmm_diag: max cluster pct by K = K2:92.91%, K3:60.43%, K4:50.85%, K5:44.14%, K6:41.17%, K7:29.24%, K8:29.79%

## Selection Reminder
Do not choose only by the best metric. Inspect cluster size balance and downstream interpretability before naming states.