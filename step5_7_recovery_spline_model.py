from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "step5_6_recovery_tipping_point" / "s4_recovery_tipping_point_base.csv"
OUT_DIR = ROOT / "step5_7_recovery_spline_model"

RIDGE = 1e-6
KNOTS = [0.0, 0.2, 0.5, 1.0]
CONTINUOUS_CONTROLS = [
    "log_gap_before_s4_days",
    "log_player_lifetime_days",
    "log_prior_session_count",
    "log_prior_s4_count",
    "prior_s4_share",
]


def normal_p_value(z: float) -> float:
    return erfc(abs(z) / sqrt(2.0))


def rcs_basis(x: pd.Series | np.ndarray, knots=KNOTS) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    k_last = knots[-1]
    k_penult = knots[-2]
    denom = k_last - k_penult

    out = {"density_linear": x}
    for j, k_j in enumerate(knots[:-2], start=1):
        term = (
            np.maximum(x - k_j, 0) ** 3
            - ((k_last - k_j) / denom) * np.maximum(x - k_penult, 0) ** 3
            + ((k_penult - k_j) / denom) * np.maximum(x - k_last, 0) ** 3
        )
        out[f"density_rcs_{j}"] = term
    return pd.DataFrame(out)


def logistic_irls(x_df: pd.DataFrame, y: np.ndarray, groups: pd.Series, max_iter=100, tol=1e-7):
    x = x_df.to_numpy(dtype=float)
    y = y.astype(float)
    p = x.shape[1]
    beta = np.zeros(p)
    penalty = np.eye(p) * RIDGE
    penalty[0, 0] = 0.0
    for iteration in range(1, max_iter + 1):
        eta = np.clip(x @ beta, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-8, None)
        grad = x.T @ (y - mu) - penalty @ beta
        hess = (x.T * w) @ x + penalty
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(x @ beta, -35, 35)
    mu = 1 / (1 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-8, None)
    bread = np.linalg.inv((x.T * w) @ x + penalty)

    score = x * (y - mu)[:, None]
    score_df = pd.DataFrame(score, columns=x_df.columns)
    score_df["pid"] = groups.to_numpy()
    summed = score_df.groupby("pid", sort=False).sum()
    s = summed.to_numpy(dtype=float)
    meat = s.T @ s
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se, cov, mu, iteration


def build_design(
    df: pd.DataFrame,
    spline=True,
    standardization: dict[str, tuple[float, float]] | None = None,
    columns: list[str] | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    if spline:
        density = rcs_basis(df["recent_s4_density_prev5"])
    else:
        density = pd.DataFrame({"density_linear": df["recent_s4_density_prev5"].astype(float).to_numpy()})
    controls = df[
        [
            "log_gap_before_s4_days",
            "log_player_lifetime_days",
            "log_prior_session_count",
            "log_prior_s4_count",
            "prior_s4_share",
            "is_update_window_2023_01_31_to_2023_02_07",
        ]
    ].astype(float).reset_index(drop=True)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=int).reset_index(drop=True)
    x = pd.concat([density.reset_index(drop=True), controls, month_dummies.astype(float)], axis=1)
    if columns is None:
        nunique = x.nunique(dropna=False)
        x = x.loc[:, nunique > 1]
    fitted_standardization = {}
    for col in CONTINUOUS_CONTROLS:
        if col in x.columns:
            if standardization is None:
                mean = x[col].mean()
                sd = x[col].std(ddof=1)
                fitted_standardization[col] = (float(mean), float(sd))
            else:
                mean, sd = standardization[col]
            if sd > 0:
                x[col] = (x[col] - mean) / sd
    x.insert(0, "intercept", 1.0)
    if columns is not None:
        for col in columns:
            if col not in x.columns:
                x[col] = 0.0
        x = x[columns]
    if return_metadata:
        metadata = {
            "standardization": fitted_standardization,
            "columns": list(x.columns),
        }
        return x, metadata
    return x


def model_summary(x: pd.DataFrame, beta: np.ndarray, se: np.ndarray, analysis: str) -> pd.DataFrame:
    rows = []
    for term, coef, term_se in zip(x.columns, beta, se):
        z = coef / term_se if term_se > 0 else np.nan
        rows.append(
            {
                "analysis": analysis,
                "term": term,
                "coef_log_odds": float(coef),
                "cluster_robust_se_by_player": float(term_se),
                "z": float(z) if np.isfinite(z) else np.nan,
                "p_value": float(normal_p_value(z)) if np.isfinite(z) else np.nan,
                "odds_ratio": float(np.exp(coef)),
                "or_ci_low_95": float(np.exp(coef - 1.96 * term_se)),
                "or_ci_high_95": float(np.exp(coef + 1.96 * term_se)),
            }
        )
    return pd.DataFrame(rows)


def predict_curve(
    df: pd.DataFrame,
    x: pd.DataFrame,
    beta: np.ndarray,
    cov: np.ndarray,
    design_metadata: dict[str, object],
) -> pd.DataFrame:
    grid = pd.DataFrame({"recent_s4_density_prev5": np.linspace(0, 1, 101)})
    template = df.copy()
    medians = {
        "log_gap_before_s4_days": df["log_gap_before_s4_days"].median(),
        "log_player_lifetime_days": df["log_player_lifetime_days"].median(),
        "log_prior_session_count": df["log_prior_session_count"].median(),
        "log_prior_s4_count": df["log_prior_s4_count"].median(),
        "prior_s4_share": df["prior_s4_share"].median(),
        "is_update_window_2023_01_31_to_2023_02_07": 0,
    }
    pred = pd.DataFrame({k: [v] * len(grid) for k, v in medians.items()})
    pred["recent_s4_density_prev5"] = grid["recent_s4_density_prev5"]
    modal_month = df["month"].mode().iloc[0]
    pred["month"] = modal_month

    xp = build_design(
        pred,
        spline=True,
        standardization=design_metadata["standardization"],
        columns=design_metadata["columns"],
    )
    xp = xp[x.columns]

    eta = xp.to_numpy(dtype=float) @ beta
    var_eta = np.einsum("ij,jk,ik->i", xp.to_numpy(dtype=float), cov, xp.to_numpy(dtype=float))
    se_eta = np.sqrt(np.maximum(var_eta, 0))
    for name, e in [("fit", eta), ("low", eta - 1.96 * se_eta), ("high", eta + 1.96 * se_eta)]:
        pred[f"predicted_recovery_{name}"] = 1 / (1 + np.exp(-np.clip(e, -35, 35)))
    pred["recent_s4_density_prev5"] = grid["recent_s4_density_prev5"]
    return pred[["recent_s4_density_prev5", "predicted_recovery_fit", "predicted_recovery_low", "predicted_recovery_high"]]


def binned_observed(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["recent_s4_density_prev5", "recovered_to_s1_s2_within_30d", "pid"]].copy()
    work["bin"] = pd.qcut(work["recent_s4_density_prev5"], q=6, duplicates="drop")
    out = (
        work.groupby("bin", observed=True)
        .agg(
            sessions=("recovered_to_s1_s2_within_30d", "size"),
            players=("pid", "nunique"),
            density_median=("recent_s4_density_prev5", "median"),
            density_min=("recent_s4_density_prev5", "min"),
            density_max=("recent_s4_density_prev5", "max"),
            observed_recovery_rate=("recovered_to_s1_s2_within_30d", "mean"),
        )
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def font(size: int, bold=False):
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_plot(curve: pd.DataFrame, bins: pd.DataFrame, out_path: Path):
    width, height = 1200, 760
    bg = (250, 250, 248)
    text = (34, 34, 34)
    muted = (98, 108, 118)
    axis = (64, 64, 64)
    grid = (218, 224, 228)
    line = (31, 96, 128)
    band = (187, 214, 224)
    point = (25, 112, 105)
    red = (184, 60, 60)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "Nonlinear Recovery Probability After S4", font=font(34, True), fill=text)
    draw.text((70, 80), "Restricted cubic spline for recent S4 density, adjusted for ex-ante controls", font=font(21), fill=muted)

    left, right = 115, 1080
    top, bottom = 160, 610
    ymin, ymax = 0.0, 0.60

    def sx(v):
        return left + v * (right - left)

    def sy(v):
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)

    for frac in np.linspace(0, 1, 5):
        y = ymin + frac * (ymax - ymin)
        yy = sy(y)
        draw.line((left, yy, right, yy), fill=grid, width=1)
        draw.text((left - 14, yy), f"{int(y*100)}%", font=font(16), fill=muted, anchor="rm")
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    draw.line((left, top, left, bottom), fill=axis, width=2)

    low_pts = [(sx(r.recent_s4_density_prev5), sy(r.predicted_recovery_low)) for r in curve.itertuples(index=False)]
    high_pts = [(sx(r.recent_s4_density_prev5), sy(r.predicted_recovery_high)) for r in curve.itertuples(index=False)]
    polygon = high_pts + low_pts[::-1]
    draw.polygon(polygon, fill=band)
    fit_pts = [(sx(r.recent_s4_density_prev5), sy(r.predicted_recovery_fit)) for r in curve.itertuples(index=False)]
    draw.line(fit_pts, fill=line, width=5)

    for r in bins.itertuples(index=False):
        x = sx(float(r.density_median))
        y = sy(float(r.observed_recovery_rate))
        radius = 5 + min(11, int(r.sessions / 700))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=point)
        draw.text((x, y - 24), f"{r.observed_recovery_rate*100:.0f}%", font=font(15), fill=text, anchor="ma")
        draw.text((x, bottom + 38), f"n={int(r.sessions)}", font=font(13), fill=muted, anchor="ma")

    tx = sx(0.5)
    draw.line((tx, top, tx, bottom), fill=red, width=2)
    draw.text((tx + 8, top + 12), "density = 0.5", font=font(18), fill=red)

    for tv in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = sx(tv)
        draw.line((xx, bottom, xx, bottom + 7), fill=axis, width=1)
        draw.text((xx, bottom + 12), f"{tv:.2f}", font=font(16), fill=muted, anchor="ma")
    draw.text(((left + right) / 2, bottom + 70), "Recent S4 density in previous 5 sessions", font=font(18), fill=muted, anchor="ma")
    draw.text((left, top - 30), "Predicted recovery probability", font=font(17), fill=muted)
    draw.text((70, 705), "Line: adjusted spline prediction. Band: cluster-robust 95% CI. Points: observed binned recovery rates.", font=font(17), fill=muted)
    img.save(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE_PATH)
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    for col in ["evaluable_30d", "recovered_to_s1_s2_within_30d", "is_update_window_2023_01_31_to_2023_02_07"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(df[col]).astype(bool)
    work = df[df["evaluable_30d"]].copy()
    y = work["recovered_to_s1_s2_within_30d"].astype(int).to_numpy()

    x_spline, spline_metadata = build_design(work, spline=True, return_metadata=True)
    beta_s, se_s, cov_s, mu_s, it_s = logistic_irls(x_spline, y, work["pid"])
    spline_coef = model_summary(x_spline, beta_s, se_s, "rcs_recent_s4_density_prev5")

    x_linear = build_design(work, spline=False)
    beta_l, se_l, cov_l, mu_l, it_l = logistic_irls(x_linear, y, work["pid"])
    linear_coef = model_summary(x_linear, beta_l, se_l, "linear_recent_s4_density_prev5")
    coefs = pd.concat([spline_coef, linear_coef], ignore_index=True)
    coefs.to_csv(OUT_DIR / "recovery_spline_model_coefficients.csv", index=False, encoding="utf-8-sig")

    curve = predict_curve(work, x_spline, beta_s, cov_s, spline_metadata)
    bins = binned_observed(work)
    curve.to_csv(OUT_DIR / "recovery_spline_predicted_curve.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(OUT_DIR / "recovery_spline_observed_bins.csv", index=False, encoding="utf-8-sig")
    draw_plot(curve, bins, OUT_DIR / "recovery_spline_curve_recent_s4_density_prev5_30d.png")

    summary = pd.DataFrame(
        [
            {
                "analysis": "rcs_recent_s4_density_prev5",
                "sessions": int(len(work)),
                "players": int(work["pid"].nunique()),
                "recovered": int(y.sum()),
                "recovery_rate": float(y.mean()),
                "knots": ",".join(str(k) for k in KNOTS),
                "irls_iterations": int(it_s),
                "mean_predicted_recovery": float(mu_s.mean()),
            },
            {
                "analysis": "linear_recent_s4_density_prev5",
                "sessions": int(len(work)),
                "players": int(work["pid"].nunique()),
                "recovered": int(y.sum()),
                "recovery_rate": float(y.mean()),
                "knots": "",
                "irls_iterations": int(it_l),
                "mean_predicted_recovery": float(mu_l.mean()),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "recovery_spline_model_summary.csv", index=False, encoding="utf-8-sig")

    focus_points = curve[curve["recent_s4_density_prev5"].isin([0.0, 0.2, 0.5, 0.8, 1.0])]
    if focus_points.empty:
        focus_points = curve.iloc[[0, 20, 50, 80, 100]]
    lines = [
        "# Step 5.7 Nonlinear Recovery Probability Model",
        "",
        "## Scope",
        "",
        "- Model: `Recovery within 30d after S4 ~ restricted cubic spline(recent_S4_density_prev5) + controls`.",
        f"- RCS knots: {KNOTS}.",
        "- Controls: pre-S4 gap, player lifetime, prior session count, prior S4 count/share, update-window flag, and month.",
        "- Standard errors are clustered by player.",
        f"- Sample: {len(work):,} 30d-evaluable S4 origin sessions from {work['pid'].nunique():,} players.",
        "",
        "## Adjusted Curve Points",
        "",
        "| recent S4 density prev5 | predicted recovery | 95% CI |",
        "|---:|---:|---|",
    ]
    for r in focus_points.itertuples(index=False):
        lines.append(
            f"| {r.recent_s4_density_prev5:.2f} | {r.predicted_recovery_fit*100:.1f}% | "
            f"{r.predicted_recovery_low*100:.1f}%-{r.predicted_recovery_high*100:.1f}% |"
        )
    density_terms = spline_coef[spline_coef["term"].str.startswith("density")]
    lines += [
        "",
        "## Density Terms",
        "",
        "| term | OR | 95% CI | p-value |",
        "|---|---:|---|---:|",
    ]
    for r in density_terms.itertuples(index=False):
        lines.append(
            f"| {r.term} | {r.odds_ratio:.3f} | {r.or_ci_low_95:.3f}-{r.or_ci_high_95:.3f} | {r.p_value:.3g} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The adjusted curve declines as recent S4 density rises, matching the threshold result around density >= 0.5.",
        "- Because recent S4 density has only a few effective values in a 5-session window, the spline should be read as a smoothed recoverability curve, with observed binned points used as the empirical anchor.",
        "",
        "## Outputs",
        "",
        "- `recovery_spline_curve_recent_s4_density_prev5_30d.png`",
        "- `recovery_spline_predicted_curve.csv`",
        "- `recovery_spline_observed_bins.csv`",
        "- `recovery_spline_model_coefficients.csv`",
        "- `recovery_spline_model_summary.csv`",
    ]
    (OUT_DIR / "step5_7_recovery_spline_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "recovery_spline_curve_recent_s4_density_prev5_30d.png")
    print(curve.iloc[[0, 20, 50, 80, 100]].to_string(index=False))


if __name__ == "__main__":
    main()
