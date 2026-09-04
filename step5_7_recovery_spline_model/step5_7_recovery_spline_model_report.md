# Step 5.7 Nonlinear Recovery Probability Model

## Scope

- Model: `Recovery within 30d after S4 ~ restricted cubic spline(recent_S4_density_prev5) + controls`.
- RCS knots: [0.0, 0.2, 0.5, 1.0].
- Controls: pre-S4 gap, player lifetime, prior session count, prior S4 count/share, update-window flag, and month.
- Standard errors are clustered by player.
- Sample: 7,093 30d-evaluable S4 origin sessions from 3,730 players.

## Adjusted Curve Points

| recent S4 density prev5 | predicted recovery | 95% CI |
|---:|---:|---|
| 0.00 | 43.6% | 38.5%-48.8% |
| 0.20 | 27.6% | 22.5%-33.4% |
| 0.50 | 16.6% | 11.2%-24.0% |
| 0.80 | 6.8% | 3.6%-12.6% |
| 1.00 | 3.0% | 1.1%-7.7% |

## Density Terms

| term | OR | 95% CI | p-value |
|---|---:|---|---:|
| density_linear | 0.021 | 0.006-0.070 | 4.92e-10 |
| density_rcs_1 | 6591.861 | 0.690-62950151.913 | 0.06 |
| density_rcs_2 | 0.000 | 0.000-2.371 | 0.0608 |

## Interpretation

- The adjusted curve declines as recent S4 density rises, matching the threshold result around density >= 0.5.
- Because recent S4 density has only a few effective values in a 5-session window, the spline should be read as a smoothed recoverability curve, with observed binned points used as the empirical anchor.

## Outputs

- `recovery_spline_curve_recent_s4_density_prev5_30d.png`
- `recovery_spline_predicted_curve.csv`
- `recovery_spline_observed_bins.csv`
- `recovery_spline_model_coefficients.csv`
- `recovery_spline_model_summary.csv`
