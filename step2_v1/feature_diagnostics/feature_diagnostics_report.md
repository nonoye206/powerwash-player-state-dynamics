# Step 2 V1 Feature Diagnostics

## Recommended Discovery Set
`duration_minutes`, `days_since_previous_session`, `jobs_started`, `jobs_resumed`, `tasks_completed`, `items_purchased`, `job_completion_ratio`, `job_exit_ratio`, `progression_gain`, `mode_diversity`, plus `first_session` and `no_started_or_resumed_job` indicators.

Use transformed numeric inputs for discovery:
- log1p transform duration, gap, event counts, and raw heavy-tailed counts.
- Winsorize heavy-tailed numeric features at P99 before scaling.
- Add indicators for first session and no started/resumed job before filling ratio NaNs.
- Keep `primary_game_mode` and `player_lifetime_days` for post-hoc state interpretation, not the primary distance space.
- Use `purchase_any` rather than raw purchase count for the main V1 clustering.
- Cap `job_completion_ratio` and `job_exit_ratio` to [0, 1]; keep >1 flags in a sidecar file.

## Main Distribution Notes
- `duration_minutes`: missing 0.00%, zero 0.00%, p50 53.97, p95 285.3, p99 566.1, max 1433.
- `days_since_previous_session`: missing 10.24%, zero 0.00%, p50 0.7805, p95 26.72, p99 124.5, max 220.1.
- `player_lifetime_days`: missing 0.00%, zero 10.24%, p50 14, p95 162.2, p99 193.1, max 224.
- `jobs_started`: missing 0.00%, zero 33.35%, p50 1, p95 5, p99 10, max 85.
- `jobs_resumed`: missing 0.00%, zero 44.35%, p50 1, p95 1, p99 2, max 12.
- `jobs_completed`: missing 0.00%, zero 40.63%, p50 1, p95 5, p99 9, max 311.
- `jobs_exited`: missing 0.00%, zero 7.67%, p50 2, p95 6, p99 10, max 83.
- `tasks_completed`: missing 0.00%, zero 38.24%, p50 1, p95 8, p99 14, max 311.
- `items_purchased`: missing 0.00%, zero 69.55%, p50 0, p95 10, p99 53, max 384.
- `job_completion_ratio`: missing 7.11%, zero 33.53%, p50 0.5, p95 1, p99 1, max 72.
- `job_exit_ratio`: missing 7.11%, zero 0.63%, p50 1, p95 1, p99 1, max 3.
- `progression_gain`: missing 7.11%, zero 16.83%, p50 0.8933, p95 1, p99 1, max 1.
- `campaign_progression_gain`: missing 7.11%, zero 60.26%, p50 0, p95 0.1053, p99 0.2105, max 1.
- `mode_diversity`: missing 0.00%, zero 0.00%, p50 1, p95 2, p99 2, max 4.

## High Correlations After Log Transforms
- `log1p_jobs_completed` vs `log1p_tasks_completed`: 0.920
- `log1p_jobs_started` vs `log1p_jobs_exited`: 0.886
- `log1p_jobs_completed` vs `log1p_jobs_exited`: 0.870
- `log1p_jobs_started` vs `log1p_jobs_completed`: 0.843
- `log1p_jobs_completed` vs `job_completion_ratio`: 0.839
- `log1p_jobs_completed` vs `campaign_progression_gain`: 0.817
- `log1p_jobs_exited` vs `log1p_tasks_completed`: 0.816
- `log1p_jobs_exited` vs `progression_gain`: 0.784
- `log1p_jobs_started` vs `log1p_tasks_completed`: 0.780
- `log1p_jobs_exited` vs `campaign_progression_gain`: 0.759
- `log1p_tasks_completed` vs `job_completion_ratio`: 0.752
- `log1p_tasks_completed` vs `campaign_progression_gain`: 0.748
- `log1p_jobs_completed` vs `progression_gain`: 0.718
- `log1p_jobs_started` vs `campaign_progression_gain`: 0.703
- `log1p_tasks_completed` vs `progression_gain`: 0.690

## Player Influence
- players: 10834
- sessions: 105818
- max_sessions_per_player: 269
- p95_sessions_per_player: 36.0
- top_1pct_players_session_share: 0.10844090797406869
- top_5pct_players_session_share: 0.30257612126481315

## Primary Game Mode
- Career: 85388 sessions (80.69%)
- (missing): 7527 sessions (7.11%)
- FreePlay: 6918 sessions (6.54%)
- Special: 5320 sessions (5.03%)
- Challenge: 665 sessions (0.63%)

## Feature Decisions
- `duration_minutes`: use_log1p_winsorized. Core engagement intensity. Very right-skewed, with >12h sessions present.
- `days_since_previous_session`: use_log1p_with_first_session_indicator. Important recency signal. Missing mainly first sessions.
- `player_lifetime_days`: exclude_from_main_discovery_use_for_interpretation. Captures longitudinal position, but may make states reflect lifecycle stage rather than session behavior.
- `jobs_started`: use_log1p_winsorized. Direct activity breadth signal.
- `jobs_resumed`: use_log1p_winsorized. Captures returning to incomplete jobs.
- `jobs_completed`: drop_or_keep_only_for_completion_variant. Highly redundant with job_completion_ratio and campaign/progression signals.
- `jobs_exited`: drop_or_keep_only_for_exit_variant. Highly coupled to starts/resumes; job_exit_ratio is more interpretable.
- `tasks_completed`: use_log1p_winsorized. Best compact signal for within-session work volume.
- `items_purchased`: use_binary_or_log1p_winsorized. Sparse but meaningful behavior; binary purchase_any may be more stable for clustering.
- `job_completion_ratio`: use_with_no_job_indicator. Interpretable outcome quality signal; undefined when no started/resumed job.
- `job_exit_ratio`: use_with_no_job_indicator. Interpretable disengagement/friction signal; undefined when no started/resumed job.
- `progression_gain`: use_winsorized. No negative values and directly captures level progress inside session.
- `campaign_progression_gain`: drop_initially. Sparse/redundant with completion and progression; keep for sensitivity analysis.
- `primary_game_mode`: exclude_from_unsupervised_numeric_state_discovery. Categorical label can dominate distance metrics; use for post-hoc state interpretation.
- `mode_diversity`: use. Compact numeric signal for mode switching.