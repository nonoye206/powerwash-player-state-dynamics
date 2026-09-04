# Step 3.0 and 3.1 PCA

## Step 3.0 Input Validation
- row_count: 105818
- row_count_expected: 105818
- row_count_matches_expected: True
- session_id_unique: True
- session_id_matches_raw: True
- pid_matches_raw: True
- session_index_matches_raw: True
- nan_cells: 0
- inf_cells: 0
- all_transform_checks_passed: True

Transform checks all compare `state_discovery_input_v1.csv` against `session_features_v1.csv`.

- first_session equals missing raw gap: max_abs_diff=0, passed=True
- no_started_or_resumed_job equals jobs_started + jobs_resumed == 0: max_abs_diff=0, passed=True
- log_duration_minutes equals log1p(P99-winsorized duration): max_abs_diff=8.88e-16, passed=True
- log_gap_days equals log1p(P99-winsorized gap, first sessions filled 0): max_abs_diff=8.88e-16, passed=True
- log_jobs_started equals log1p(P99-winsorized jobs_started): max_abs_diff=4.44e-16, passed=True
- log_jobs_resumed equals log1p(P99-winsorized jobs_resumed): max_abs_diff=0, passed=True
- log_tasks_completed equals log1p(P99-winsorized tasks_completed): max_abs_diff=4.44e-16, passed=True
- purchase_any equals items_purchased > 0: max_abs_diff=0, passed=True
- job_completion_ratio_capped equals clipped raw completion ratio: max_abs_diff=0, passed=True
- job_exit_ratio_capped equals clipped raw exit ratio: max_abs_diff=0, passed=True
- progression_gain equals P99-winsorized raw progression gain: max_abs_diff=4.24e-22, passed=True
- mode_diversity equals raw mode diversity: max_abs_diff=0, passed=True

## Step 3.1 Explained Variance
- PC1: 37.83% (cumulative 37.83%)
- PC2: 15.73% (cumulative 53.56%)
- PC3: 10.91% (cumulative 64.47%)
- PC4: 8.23% (cumulative 72.70%)
- PC5: 7.48% (cumulative 80.18%)
- PC6: 5.46% (cumulative 85.64%)
- PC7: 5.18% (cumulative 90.83%)
- PC8: 3.23% (cumulative 94.06%)

## PC Interpretation
- PC1: 37.83% variance.
  Positive: no_started_or_resumed_job=0.309; log_gap_days=0.115; log_jobs_resumed=-0.076; first_session=-0.089
  Negative: log_tasks_completed=-0.390; log_duration_minutes=-0.386; log_jobs_started=-0.376; progression_gain=-0.375
- PC2: 15.73% variance.
  Positive: log_jobs_resumed=0.501; job_exit_ratio_capped=0.438; log_duration_minutes=0.154; log_gap_days=-0.013
  Negative: no_started_or_resumed_job=-0.444; first_session=-0.366; log_jobs_started=-0.286; log_tasks_completed=-0.228
- PC3: 10.91% variance.
  Positive: log_gap_days=0.563; log_jobs_resumed=0.308; no_started_or_resumed_job=0.267; purchase_any=0.235
  Negative: first_session=-0.534; job_exit_ratio_capped=-0.278; log_jobs_started=0.005; log_duration_minutes=0.043
- PC4: 8.23% variance.
  Positive: purchase_any=0.195; job_completion_ratio_capped=0.178; log_gap_days=0.112; progression_gain=0.095
  Negative: mode_diversity=-0.942; log_jobs_started=-0.125; first_session=-0.033; job_exit_ratio_capped=-0.021
- PC5: 7.48% variance.
  Positive: log_gap_days=0.610; job_exit_ratio_capped=0.267; log_jobs_started=0.205; log_duration_minutes=0.106
  Negative: log_jobs_resumed=-0.522; purchase_any=-0.287; progression_gain=-0.257; no_started_or_resumed_job=-0.247

## Standardization Check
- max absolute standardized mean: 2.44e-16
- max standardized std deviation error: 1.11e-16

Generated files include PCA scores, loadings, explained variance, validation tables, and SVG plots.