# Player Behavior Research

This repository contains a reproducible analysis pipeline for player behavior research, from raw event audit and session reconstruction through session-level feature engineering, PCA validation, behavioral state clustering, stability checks, and player-level state transition analysis.

## Pipeline

- `step1_data_audit.py`: audits source event tables and reconstructs player sessions.
- `step2_v1_session_features.py`: builds session-level features for downstream modeling.
- `step2_v1_feature_diagnostics.py`: produces feature diagnostics, distributions, correlations, and modeling input tables.
- `step3_pca_validation.py`: validates transforms, standardizes features, runs PCA, and exports PCA scores/loadings/plots.
- `step3_2_cluster_evaluation.py`: evaluates KMeans and diagonal-covariance GMM cluster candidates across `k=2..8`.
- `step3_3_cluster_profiles.py`: summarizes selected candidate cluster profiles.
- `step3_4_gmm_posterior_stability.py`: checks GMM posterior confidence and seed/subsample stability.
- `step3_5_minimal_session_cluster_audit.py`: audits the minimal-session cluster and related session patterns.
- `step3_6_date_regime_sensitivity.py`: tests sensitivity to update windows and date regimes.
- `step4_1_state_sequences.py`: builds player-level state sequences and transition matrices.

## Key Outputs

- `step1_audit/`: data audit reports and reconstructed sessions.
- `step2_v1/`: session feature table and feature diagnostics.
- `step3_pca/`: PCA inputs, scores, loadings, explained variance, and plots.
- `step3_2_cluster_evaluation/`: cluster labels and model metrics.
- `step3_3_cluster_profiles/`: selected cluster profile summaries.
- `step3_4_gmm_posterior_stability/`: posterior and stability diagnostics.
- `step3_5_minimal_session_cluster_audit/`: focused audit of minimal-session cluster behavior.
- `step3_6_date_regime_sensitivity/`: date-regime and anomaly sensitivity outputs.
- `step4_1_state_sequences/`: state assignment, player sequence, and transition matrix outputs.

## Current Analysis Snapshot

- Reconstructed sessions: 105,818
- Players in transition analysis: 10,834
- Players with at least two sessions: 8,031
- State transition events: 94,984
- PCA first five components cumulative variance: 80.18%
- Main state model for sequence analysis: `pc1_pc5_gmm_diag_k5`

## Notes

The original source archive referenced by the audit report was located outside this repository. This repository stores derived analysis code and exported artifacts needed to review the completed project state.
