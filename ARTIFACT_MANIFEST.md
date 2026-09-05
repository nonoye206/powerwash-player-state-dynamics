# Artifact Manifest

This manifest maps the main reported claims to the scripts and generated artifacts that support them. Raw telemetry is not redistributed in this repository.

## Inputs

- Raw zip archive: pass with `scripts/step1_data_audit.py --raw-zip <path>` or set `PWS_RAW_ZIP`.
- Extracted raw CSV directory: pass with `scripts/step2_v1_session_features.py --raw-data-dir <path>` or set `PWS_RAW_DATA_DIR`.
- Repo-relative fallback paths are `./data.zip` and `./data/data/`.

## Reproduction Entry Point

- `run_pipeline.ps1`: runs the full analysis from raw telemetry through README figures.
- Example:

```powershell
.\run_pipeline.ps1 `
  -Python "C:\path\to\python.exe" `
  -RawZip "C:\data\powerwash\data.zip" `
  -RawDataDir "C:\data\powerwash\data\data"
```

## Main Claims And Artifacts

| Claim | Primary script | Primary artifacts |
|---|---|---|
| Session reconstruction yields 105,818 paired sessions | `scripts/step1_data_audit.py` | `step1_audit/session_summary.json`, `step1_audit/reconstructed_sessions.csv` |
| V1 session features align one row per reconstructed session | `scripts/step2_v1_session_features.py`, `scripts/step2_v1_feature_diagnostics.py` | `step2_v1/session_features_v1.csv`, `step2_v1/feature_diagnostics/state_discovery_input_v1.csv` |
| PCA validates the standardized state-discovery input | `scripts/step3_pca_validation.py`, `scripts/export_requested_pca_artifacts.py` | `step3_pca/pca_report.md`, `step3_pca/pca_loadings.csv`, `step3_pca/pca_scree_plot.png` |
| Frozen state representation uses `pc1_pc5_gmm_diag_k5` | `scripts/step3_2_cluster_evaluation.py`, `scripts/step3_3_cluster_profiles.py`, `scripts/step3_4_gmm_posterior_stability.py` | `step3_2_cluster_evaluation/labels_pc1_pc5_gmm_diag_k5.csv`, `step3_3_cluster_profiles/candidate_cluster_profiles_compact.csv` |
| S4 minimal/no-op state is robust to the Jan31-Feb7 update window | `scripts/step3_5_minimal_session_cluster_audit.py`, `scripts/step3_6_date_regime_sensitivity.py` | `step3_5_minimal_session_cluster_audit/`, `step3_6_date_regime_sensitivity/` |
| Player-level sequences produce 94,984 transitions | `scripts/step4_1_state_sequences.py` | `step4_1_state_sequences/state_transition_edges_v1.csv`, `step4_1_state_sequences/transition_matrix_probabilities_5x5.csv` |
| Transition structure is robust to update-window exclusion and player balancing | `scripts/step4_2_transition_robustness.py` | `step4_2_transition_robustness/` |
| Observed disengagement uses right-censored 7d, 14d, and 30d labels | `scripts/step5_disengagement_definition_risk.py` | `step5_disengagement_risk/disengagement_threshold_summary.csv` |
| S4 persistence has elevated 30d risk in the main model but weakens under update-window exclusion | `scripts/step5_2_logistic_disengagement.py` | `step5_2_logistic_disengagement/` |
| Recovery from S4 to S1/S2 has lower adjusted disengagement risk than S4 persistence | `scripts/step5_3_recovery_path_analysis.py`, `scripts/step5_3_adjusted_recovery_path_model.py` | `step5_3_adjusted_recovery_path_model/adjusted_recovery_path_coefficients.csv`, `figures/figure_1_recovery_vs_persistence.png` |
| Behavioral narrowing is not disengagement-specific in V1 after matched comparison | `scripts/step5_4_behavioral_rigidity.py`, `scripts/step5_5_behavioral_narrowing_test.py` | `step5_5_behavioral_narrowing_test/` |
| Recent S4 concentration is associated with lower recoverability | `scripts/step5_6_recovery_threshold_scan.py`, `scripts/step5_7_recovery_spline_model.py` | `step5_6_recovery_threshold_scan/`, `step5_7_recovery_spline_model/recovery_spline_predicted_curve.csv`, `figures/figure_2_recoverability_curve.png` |

## Core Figures

- `figures/figure_1_recovery_vs_persistence.png`
- `figures/figure_2_recoverability_curve.png`

## Version-Control Policy

- Source scripts, documentation, manifest files, and core figures should be committed.
- Raw telemetry archives and extracted raw CSVs should not be committed.
- Large generated intermediate outputs may be regenerated with `run_pipeline.ps1`; commit only the outputs needed for review or reporting.
