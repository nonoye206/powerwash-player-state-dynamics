# Step 2 V1 Session Feature Sanity Check

## Feature table
- Rows: 105818
- Players: 10834
- Columns: 23
- Core feature columns: 15

## Event assignment
| table | valid events | assigned | unassigned | assigned pct |
|---|---:|---:|---:|---:|
| job_started | 176824 | 169159 | 7665 | 95.67% |
| job_resumed | 67682 | 64135 | 3547 | 94.76% |
| job_exited | 240417 | 231884 | 8533 | 96.45% |
| job_completed | 155100 | 148865 | 6235 | 95.98% |
| task_completed | 212818 | 204367 | 8451 | 96.03% |
| item_purchased | 306975 | 292563 | 14412 | 95.31% |
| exited_game | 105869 | 105819 | 50 | 99.95% |

## Main checks
- sessions: 105818
- players: 10834
- zero_duration_sessions: 4
- long_over_12h_sessions: 534
- sessions_without_matched_exit_event: 0
- sessions_with_multiple_matched_exit_events: 1
- negative_progression_gain_sessions: 0
- negative_campaign_progression_gain_sessions: 0
- sessions_job_completed_gt_started_plus_resumed: 105
- sessions_job_exited_gt_started_plus_resumed: 235
- max_sessions_per_player: 269
- p95_sessions_per_player: 36.0
- duration_minutes_p50: 53.96666666666667
- duration_minutes_p95: 285.2883333333332
- duration_minutes_max: 1433.3166666666666

## Notes
- `duration_minutes` comes from Step 1 reconstructed login-to-exit duration.
- `progression_gain` is max minus min `LevelProgressionAmount` observed inside the session.
- `campaign_progression_gain` is max minus min `CampaignProgressionAmount` observed inside the session.
- V1 intentionally excludes `subtask_completed`, `update_current_state`, study prompts, and mood reports.