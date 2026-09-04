# Step 3.5 Minimal-session Cluster Audit

Model source: `pc1_pc5_gmm_diag_k5` labels from Step 3.2. No models were refit.
Audited cluster: `4`

## Minimal Cluster Identification
| cluster | size | duration median | event count median | no-event rate | missing mode rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 8645 | 61.48 | 10.00 | 0.00% | 0.00% |
| 1 | 46709 | 86.83 | 10.00 | 0.00% | 0.00% |
| 2 | 35784 | 28.73 | 2.00 | 0.00% | 0.00% |
| 3 | 7158 | 92.15 | 13.00 | 0.00% | 1.06% |
| 4 | 7522 | 0.65 | 0.00 | 98.38% | 99.03% |

## Main Audit
- cluster: 4
- cluster_sessions: 7522
- cluster_pct_all_sessions: 7.10843145778601
- players_with_cluster: 3988
- pct_all_players_with_cluster: 36.81004245892561
- first_session_rate_pct: 9.611805370911991
- single_session_player_rate_pct: 5.251262961978197
- median_minimal_sessions_per_player_among_owners: 1.0
- max_minimal_sessions_per_player: 33
- median_session_index: 11.0
- median_total_sessions_player: 15.0
- median_trajectory_position_pct: 86.36363636363636
- has_previous_session_rate_pct: 90.388194629088
- has_next_session_return_rate_pct: 61.845253921829304
- median_days_to_next_session: 0.5118460648148149
- p75_days_to_next_session: 7.194864004629629
- any_job_task_item_event_rate_pct: 1.6219090667375697
- no_job_task_item_event_rate_pct: 98.37809093326243
- median_event_count: 0.0
- max_event_count: 317
- duration_minutes_median: 0.65
- duration_minutes_p95: 5.583333333333333
- top_date: 2023-01-31
- top_date_pct: 12.390321722946025
- top_7_dates_pct: 28.782238766285563
- date_range_start: 2022-08-18
- date_range_end: 2023-03-31

## Per-player Occurrence
| minimal sessions per player | players | pct of owning players |
|---:|---:|---:|
| 1 | 2390 | 59.93% |
| 2 | 850 | 21.31% |
| 3 | 351 | 8.80% |
| 4 | 165 | 4.14% |
| 5 | 82 | 2.06% |
| 6 | 56 | 1.40% |
| 7 | 20 | 0.50% |
| 8 | 26 | 0.65% |
| 9 | 10 | 0.25% |
| 10 | 12 | 0.30% |
| 11 | 8 | 0.20% |
| 12 | 5 | 0.13% |
| 13 | 3 | 0.08% |
| 14 | 1 | 0.03% |
| 17 | 2 | 0.05% |
| 18 | 1 | 0.03% |
| 22 | 1 | 0.03% |
| 24 | 1 | 0.03% |
| 26 | 1 | 0.03% |
| 28 | 1 | 0.03% |

## Trajectory Position
| trajectory bin | sessions | pct |
|---|---:|---:|
| first | 723 | 9.61% |
| early | 527 | 7.01% |
| middle | 822 | 10.93% |
| late | 1022 | 13.59% |
| last | 4428 | 58.87% |

## Previous Cluster
| previous cluster | sessions | pct |
|---|---:|---:|
| 4.0 | 2147 | 28.54% |
| 1.0 | 2091 | 27.80% |
| 2.0 | 1678 | 22.31% |
| none | 723 | 9.61% |
| 3.0 | 477 | 6.34% |
| 0.0 | 406 | 5.40% |

## Next Cluster
| next cluster | sessions | pct |
|---|---:|---:|
| none | 2870 | 38.15% |
| 4.0 | 2147 | 28.54% |
| 1.0 | 1182 | 15.71% |
| 2.0 | 1121 | 14.90% |
| 3.0 | 185 | 2.46% |
| 0.0 | 17 | 0.23% |

## Job/task/item Events Between Login and Exit
| event feature | sessions with event | pct sessions with event | total events | median | max |
|---|---:|---:|---:|---:|---:|
| jobs_started | 2 | 0.03% | 2 | 0.00 | 1 |
| jobs_resumed | 0 | 0.00% | 0 | 0.00 | 0 |
| jobs_completed | 13 | 0.17% | 13 | 0.00 | 1 |
| jobs_exited | 70 | 0.93% | 70 | 0.00 | 1 |
| tasks_completed | 18 | 0.24% | 25 | 0.00 | 4 |
| items_purchased | 54 | 0.72% | 1662 | 0.00 | 317 |

## Date Concentration
- Date range: 2022-08-18 to 2023-03-31
- Top date: 2023-01-31 (12.39% of audited cluster)
- Top 7 dates: 28.78% of audited cluster

Top dates:
| date | minimal sessions | pct of cluster | all sessions that day | minimal share of day |
|---|---:|---:|---:|---:|
| 2023-01-31 | 932 | 12.39% | 1418 | 65.73% |
| 2023-02-01 | 543 | 7.22% | 1737 | 31.26% |
| 2023-02-02 | 216 | 2.87% | 1350 | 16.00% |
| 2023-02-04 | 131 | 1.74% | 1173 | 11.17% |
| 2023-02-03 | 126 | 1.68% | 1098 | 11.48% |
| 2023-02-05 | 110 | 1.46% | 960 | 11.46% |
| 2022-08-19 | 107 | 1.42% | 2650 | 4.04% |
| 2022-08-22 | 90 | 1.20% | 2286 | 3.94% |
| 2022-08-20 | 84 | 1.12% | 2544 | 3.30% |
| 2022-09-08 | 79 | 1.05% | 1166 | 6.78% |
| 2023-02-06 | 75 | 1.00% | 837 | 8.96% |
| 2022-08-21 | 75 | 1.00% | 2503 | 3.00% |
| 2023-02-07 | 75 | 1.00% | 785 | 9.55% |
| 2022-08-23 | 72 | 0.96% | 2062 | 3.49% |
| 2023-03-29 | 70 | 0.93% | 70 | 100.00% |

Monthly distribution:
| month | sessions | pct |
|---|---:|---:|
| 2022-08 | 903 | 12.00% |
| 2022-09 | 1144 | 15.21% |
| 2022-10 | 624 | 8.30% |
| 2022-11 | 493 | 6.55% |
| 2022-12 | 397 | 5.28% |
| 2023-01 | 1314 | 17.47% |
| 2023-02 | 1930 | 25.66% |
| 2023-03 | 717 | 9.53% |

## Provisional Gate
This cluster behaves like a structurally valid minimal/no-op session type rather than a pure terminal churn artifact.