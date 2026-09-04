param(
    [string]$Python = "python",
    [string]$RawZip = "",
    [string]$RawDataDir = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Script,
        [string[]]$Arguments = @()
    )
    Write-Host ""
    Write-Host "==> $Script $($Arguments -join ' ')"
    & $Python $Script @Arguments
}

$step1Args = @()
if ($RawZip) {
    $step1Args += @("--raw-zip", $RawZip)
}

$step2Args = @()
if ($RawDataDir) {
    $step2Args += @("--raw-data-dir", $RawDataDir)
}

Invoke-Step "step1_data_audit.py" $step1Args
Invoke-Step "step2_v1_session_features.py" $step2Args
Invoke-Step "step2_v1_feature_diagnostics.py"
Invoke-Step "step3_pca_validation.py"
Invoke-Step "export_requested_pca_artifacts.py"
Invoke-Step "step3_2_cluster_evaluation.py"
Invoke-Step "step3_3_cluster_profiles.py"
Invoke-Step "step3_4_gmm_posterior_stability.py"
Invoke-Step "step3_5_minimal_session_cluster_audit.py"
Invoke-Step "step3_6_date_regime_sensitivity.py"
Invoke-Step "step4_1_state_sequences.py"
Invoke-Step "step4_2_transition_robustness.py"
Invoke-Step "step5_disengagement_definition_risk.py"
Invoke-Step "step5_2_logistic_disengagement.py"
Invoke-Step "step5_3_recovery_path_analysis.py"
Invoke-Step "step5_3_adjusted_recovery_path_model.py"
Invoke-Step "step5_4_behavioral_rigidity.py"
Invoke-Step "step5_5_behavioral_narrowing_test.py"
Invoke-Step "step5_6_recovery_threshold_scan.py"
Invoke-Step "step5_6_recovery_probability_plots.py"
Invoke-Step "step5_7_recovery_spline_model.py"
Invoke-Step "make_core_figures.py"

Write-Host ""
Write-Host "Pipeline complete."
