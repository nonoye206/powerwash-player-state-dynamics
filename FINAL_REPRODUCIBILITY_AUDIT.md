# Final Reproducibility Audit

Date: 2026-09-04

## Scope

This audit verifies that the repository can be rerun from raw telemetry using parameterized data paths, and that the main reported quantities are reproduced after fixing the Step 5.7 spline prediction preprocessing bug.

## Command

```powershell
.\run_pipeline.ps1 `
  -Python "C:\path\to\python.exe" `
  -RawZip "C:\data\powerwash\data.zip" `
  -RawDataDir "C:\data\powerwash\data\data"
```

## Result

The full pipeline completed successfully.

## Key Checks

| Check | Reproduced value |
|---|---:|
| Paired reconstructed sessions | 105,818 |
| Players in V1 session features | 10,834 |
| PCA rows | 105,818 |
| PCA PC1-PC5 cumulative variance | 80.18% |
| Minimal/no-op S4 sessions in main assignment | 7,522 |
| Player-level transitions | 94,984 |
| Transition robustness update-window-excluded transitions | 86,480 |
| 30d evaluable S4-origin sessions in spline model | 7,093 |
| Players in spline model | 3,730 |
| Recovered S4-origin sessions in spline model | 2,096 |

## Step 5.7 Fix Verification

The spline preprocessing fix changed only the adjusted prediction curve. Model coefficients, p-values, and sample size remained unchanged from the pre-fix run.

| Term | Coef | p-value |
|---|---:|---:|
| density_linear | -3.874973 | 4.92e-10 |
| density_rcs_1 | 8.793591 | 0.0600 |
| density_rcs_2 | -19.016666 | 0.0608 |

Corrected adjusted predicted recovery:

| recent S4 density prev5 | predicted recovery |
|---:|---:|
| 0.0 | 43.6% |
| 0.2 | 27.6% |
| 0.5 | 16.6% |
| 0.8 | 6.8% |
| 1.0 | 3.0% |

## Notes

- Raw telemetry paths are no longer hardcoded in Step 1/2; they can be passed by CLI or environment variables.
- `requirements.txt`, `environment.yml`, `run_pipeline.ps1`, `scripts/`, and `ARTIFACT_MANIFEST.md` document the runnable environment and artifact map.
- Raw telemetry is intentionally excluded from version control.
