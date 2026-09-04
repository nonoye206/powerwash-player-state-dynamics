# Step 1 Data Audit + Session Reconstruction

Source: PowerWash Simulator: Research Edition raw telemetry

## Core table audit

| table | rows | players | Time_utc range | missing pid | missing Time_utc | max column missing | duplicate pid-time-event rows |
|---|---:|---:|---|---:|---:|---:|---:|
| demographics | 11080 | 11080 | n/a to n/a | 0.00% |  | 25.56% | 0 |
| exited_game | 105869 | 10834 | 2022-08-18 13:16:05 to 2023-03-31 06:23:29 | 0.00% | 0.00% | 7.38% | 1 |
| game_saved | 8482454 | 7725 | 2022-08-18 13:08:56 to 2023-02-03 14:59:20 | 0.00% | 0.00% | 100.00% | 0 |
| item_purchased | 306975 | 7737 | 2022-08-18 13:26:37 to 2023-03-29 00:38:56 | 0.00% | 0.00% | 3.16% | 0 |
| job_completed | 155100 | 10105 | 2022-08-18 13:11:53 to 2023-03-29 00:27:32 | 0.00% | 0.00% | 0.11% | 0 |
| job_exited | 240417 | 10478 | 2022-08-18 13:12:00 to 2023-03-29 02:14:02 | 0.00% | 0.00% | 17.09% | 0 |
| job_resumed | 67682 | 7140 | 2022-08-18 13:23:09 to 2023-03-28 20:00:07 | 0.00% | 0.00% | 91.65% | 0 |
| job_started | 176824 | 10489 | 2022-08-18 13:08:21 to 2023-03-29 00:33:56 | 0.00% | 0.00% | 27.52% | 0 |
| mood_reported | 24107 | 5851 | 2022-08-18 13:09:48 to 2023-03-28 21:54:46 | 0.00% | 0.00% | 41.74% | 0 |
| player_logged_in | 113004 | 11080 | 2022-08-18 13:06:56 to 2023-03-31 06:23:01 | 0.00% | 0.00% | 0.00% | 0 |
| study_prompt_answered | 702209 | 9622 | 2022-08-18 13:15:17 to 2023-03-29 00:41:26 | 0.00% | 0.00% | 49.16% | 0 |
| study_reward_claimed | 17126 | 3926 | 2023-01-31 17:13:21 to 2023-03-31 02:06:52 | 0.00% | 0.00% | 64.55% | 0 |
| study_reward_unlocked | 9623 | 2340 | 2023-01-31 17:38:39 to 2023-03-29 14:45:07 | 0.00% | 0.00% | 38.75% | 0 |
| subtask_completed | 14376699 | 10432 | 2022-08-18 13:08:49 to 2023-03-29 00:45:55 | 0.00% | 0.00% | 0.00% | 0 |
| task_completed | 212818 | 10107 | 2022-08-18 13:11:53 to 2023-03-29 00:27:32 | 0.00% | 0.00% | 0.00% | 0 |
| update_current_state | 12710590 | 3939 | 2023-01-31 17:22:14 to 2023-03-29 02:13:59 | 0.00% | 0.00% | 16.32% | 0 |

## Session pairing audit

- players_with_login_or_exit: 11080
- total_logins_with_valid_pid_time: 113004
- total_exits_with_valid_pid_time: 105869
- paired_sessions: 105818
- unmatched_logins: 7186
- unmatched_exits: 51
- overlapping_login_events: 5970
- negative_duration_sessions: 0
- zero_duration_sessions: 4
- sessions_over_12h: 534
- duration_seconds_min: 0.0
- duration_seconds_p50: 3238.0
- duration_seconds_p95: 17117.29999999999
- duration_seconds_max: 85999.0
- abs_delta_vs_exit_current_session_length_minutes_p50: 0.5
- abs_delta_vs_exit_current_session_length_minutes_p95: 0.9666666666666666

Players with fully clean login/exit pairing: 7991/11080 (72.12% if denominator > 0).

## Preliminary judgment
`player_logged_in -> exited_game` is not perfectly one-to-one. Use the greedy within-player chronological pairing as a reconstruction baseline, and explicitly flag/drop unmatched or overlapping cases depending on downstream analysis.

Detailed CSV outputs:
- `table_audit.csv`
- `column_missingness.csv`
- `session_pairing_by_player.csv`
- `reconstructed_sessions.csv`