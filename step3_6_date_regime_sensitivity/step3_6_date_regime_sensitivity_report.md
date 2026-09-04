# Step 3.6 Date-regime Sensitivity

No broad model search was run. This checks the existing minimal cluster and two targeted GMM K5 refits.

## Existing K5 Cluster 4 After Removing 2023-01-31 To 2023-02-07
- cluster: 4
- sessions: 5314
- pct_subset_sessions: 5.509019282604188
- players: 3147
- pct_subset_players: 31.53306613226453
- duration_minutes_median: 0.5833333333333334
- duration_minutes_p95: 7.294999999999981
- no_event_rate_pct: 98.28754234098608
- missing_mode_rate_pct: 98.73917952578095
- first_session_rate_pct: 11.027474595408355
- has_next_session_return_rate_pct: 67.59503199096726
- median_trajectory_position_pct: 75.73027718550107
- trajectory_first_pct: 11.027474595408355
- trajectory_early_pct: 8.449378998870907
- trajectory_middle_pct: 13.455024463680843
- trajectory_late_pct: 16.917576213774936
- trajectory_last_pct: 50.15054572826496
- median_minimal_sessions_per_player: 1.0
- max_minimal_sessions_per_player: 33
- sensitivity: original_labels_excluding_2023_01_31_to_2023_02_07
- subset_sessions: 96460
- subset_players: 9980
- removed_window_sessions: 9358
- removed_window_cluster4_sessions: 2208

Top dates after window removal:
| date | sessions | pct of cluster | cluster share of day |
|---|---:|---:|---:|
| 2022-08-19 | 107 | 2.01% | 4.04% |
| 2022-08-22 | 90 | 1.69% | 3.94% |
| 2022-08-20 | 84 | 1.58% | 3.30% |
| 2022-09-08 | 79 | 1.49% | 6.78% |
| 2022-08-21 | 75 | 1.41% | 3.00% |
| 2022-08-23 | 72 | 1.35% | 3.49% |
| 2023-03-29 | 70 | 1.32% | 100.00% |
| 2022-08-25 | 63 | 1.19% | 3.62% |
| 2022-08-27 | 60 | 1.13% | 3.34% |
| 2022-09-09 | 60 | 1.13% | 5.52% |
| 2022-08-24 | 59 | 1.11% | 3.21% |
| 2022-09-03 | 58 | 1.09% | 4.22% |

## Targeted GMM K5 Refits
| sensitivity | subset sessions | minimal cluster | sessions | pct | players | duration med | no-event | missing-mode | late+last | return | found? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exclude_2023_01_31_to_2023_02_07 | 96460 | 1 | 5318 | 5.51% | 3150 | 0.58 | 98.21% | 98.66% | 67.07% | 67.62% | True |
| pre_2023_01_31_only | 81430 | 3 | 3946 | 4.85% | 2309 | 0.57 | 98.12% | 98.56% | 62.57% | 75.95% | True |

## Gate
The minimal/no-op structure survives date-window removal and reappears under both targeted K5 refits, including the pre-2023-01-31 period.
## Minimal Cluster Purchase Anomalies

- minimal_cluster: 4
- minimal_cluster_sessions: 7522
- purchase_minimal_sessions: 54
- purchase_minimal_sessions_pct: 0.7178941770805637
- purchase_events_in_purchase_minimal_sessions: 1662
- max_purchases_in_one_minimal_session: 317
- players_with_purchase_minimal_sessions: 54
- median_duration_minutes: 2.4749999999999996
- p95_duration_minutes: 44.4249999999999
- no_job_task_event_rate_pct: 90.74074074074075
- missing_mode_rate_pct: 90.74074074074075

Top purchase anomaly sessions:
| session_id | pid | session_index | login_time_utc | duration min | purchases | job/task events | has next session |
|---|---|---:|---|---:|---:|---:|---|
| p705_s0032 | p705 | 32 | 2023-01-31 20:01:25 | 3.35 | 317 | 0 | False |
| p1795_s0017 | p1795 | 17 | 2023-02-01 18:39:35 | 5.10 | 261 | 0 | False |
| p11186_s0063 | p11186 | 63 | 2022-11-16 13:43:51 | 2.90 | 163 | 0 | True |
| p10445_s0026 | p10445 | 26 | 2022-09-02 17:18:20 | 7.32 | 140 | 0 | True |
| p4680_s0004 | p4680 | 4 | 2022-08-28 07:15:48 | 5.02 | 105 | 0 | False |
| p9429_s0016 | p9429 | 16 | 2022-09-10 18:59:35 | 13.60 | 71 | 0 | True |
| p7778_s0002 | p7778 | 2 | 2023-02-01 12:53:57 | 1.43 | 70 | 0 | False |
| p8414_s0010 | p8414 | 10 | 2023-01-31 18:59:07 | 1.72 | 61 | 0 | False |
| p9438_s0011 | p9438 | 11 | 2022-09-13 19:33:36 | 19.25 | 58 | 0 | True |
| p5574_s0004 | p5574 | 4 | 2022-11-05 01:14:12 | 1.52 | 52 | 0 | False |
| p8808_s0027 | p8808 | 27 | 2023-03-03 17:03:32 | 4.27 | 45 | 0 | True |
| p614_s0014 | p614 | 14 | 2023-02-05 22:03:39 | 2.42 | 40 | 0 | False |
| p11159_s0011 | p11159 | 11 | 2023-01-19 06:02:48 | 4.15 | 37 | 0 | True |
| p2707_s0005 | p2707 | 5 | 2023-02-06 09:33:49 | 2.93 | 25 | 0 | True |
| p7961_s0014 | p7961 | 14 | 2023-02-04 15:06:42 | 1.23 | 19 | 0 | False |

These sessions should be flagged as purchase-burst/boundary-risk cases when interpreting the minimal/no-op state.

## External Context For 2023-01-31

- SteamDB records a major PowerWash Simulator update on 2023-01-31: `1.1 Update + Free Tomb Raider Special Pack Out Now!`, build `10314903`.
- The same update notes include the Tomb Raider Special Pack, new Special Jobs access, save/reset-dirt controls, Research Edition Rewards, and a Research Branch issue about popup spam with a later live fix.
- The Scientific Data paper reports a second recruitment wave in January 2023, where the main PWS branch added a menu button inviting players into the research branch.
- This supports the interpretation that 2023-01-31 was a game-version and research-branch regime change. The timing strongly suggests a telemetry/instrumentation regime change associated with the Jan31 v1.1 / Research Edition update.
- Evidence boundary: this does not prove that official notes explicitly introduced `update_current_state`; we should not write that as a confirmed fact.
- Combined with this Step 3.6 sensitivity check, the best current interpretation is: the Jan31 update likely created a short-term spike in minimal/no-op sessions, but the minimal/no-op state itself predates the update and survives when the update window is removed.