from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


LABELS_PATH = Path("step5_disengagement_risk/session_disengagement_labels.csv")
FEATURES_PATH = Path("step2_v1/session_features_v1.csv")
OUT_DIR = Path("step5_2_logistic_disengagement")
THRESHOLDS = [30, 7, 14]
UPDATE_WINDOW_START = pd.Timestamp("2023-01-31")
UPDATE_WINDOW_END = pd.Timestamp("2023-02-07")
RIDGE = 1e-6


def normal_p_value(z):
    return erfc(abs(z) / sqrt(2.0))


def load_model_frame():
    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_csv(
        FEATURES_PATH,
        usecols=[
            "session_id",
            "pid",
            "session_index",
            "days_since_previous_session",
            "player_lifetime_days",
            "job_completion_ratio",
            "job_exit_ratio",
            "primary_game_mode",
            "mode_diversity",
        ],
    )
    df = labels.merge(features, on=["session_id", "pid", "session_index"], how="left", suffixes=("", "_v1"))
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["month"] = df["login_time_utc"].dt.to_period("M").astype(str)
    df["state_id"] = df["state_id"].astype(int)
    df["prev_state_missing"] = df["prev_state_id"].isna().astype(int)
    df["prev_state_filled"] = df["prev_state_id"].fillna(-1).astype(int)
    df["first_session"] = df["session_index"].eq(1).astype(int)
    df["prior_session_count"] = df["session_index"] - 1
    df["prior_s4_count"] = (
        df.groupby("pid")["state_id"].transform(lambda s: s.eq(4).cumsum().shift(1, fill_value=0))
    )
    df["s4_persistence"] = (df["prev_state_filled"].eq(4) & df["state_id"].eq(4)).astype(int)
    df["s4_streak_2plus"] = df["s4_streak_current"].ge(2).astype(int)
    for s in range(1, 5):
        df[f"current_state_s{s}"] = df["state_id"].eq(s).astype(int)
        df[f"prev_state_s{s}"] = df["prev_state_filled"].eq(s).astype(int)
    for a, b in [(1, 4), (2, 4), (3, 4), (4, 1), (4, 2), (4, 3)]:
        df[f"transition_s{a}_to_s{b}"] = (df["prev_state_filled"].eq(a) & df["state_id"].eq(b)).astype(int)
    df["log_duration_minutes"] = np.log1p(df["duration_minutes"].clip(lower=0))
    df["log_gap_days"] = np.log1p(df["days_since_previous_session"].clip(lower=0).fillna(0))
    df["log_player_lifetime_days"] = np.log1p(df["player_lifetime_days"].clip(lower=0).fillna(0))
    df["log_prior_session_count"] = np.log1p(df["prior_session_count"].clip(lower=0))
    df["log_prior_s4_count"] = np.log1p(df["prior_s4_count"].clip(lower=0))
    df["job_completion_ratio_filled"] = df["job_completion_ratio"].clip(lower=0, upper=1).fillna(0)
    df["job_exit_ratio_filled"] = df["job_exit_ratio"].clip(lower=0, upper=1).fillna(0)
    df["mode_diversity"] = df["mode_diversity"].fillna(0)
    df["is_update_window_2023_01_31_to_2023_02_07"] = df[
        "is_update_window_2023_01_31_to_2023_02_07"
    ].astype(bool).astype(int)
    df["is_minimal_cluster_purchase_anomaly"] = df["is_minimal_cluster_purchase_anomaly"].astype(bool).astype(int)
    return df


def design_matrix(df):
    base_cols = [
        "s4_persistence",
        "current_state_s1",
        "current_state_s2",
        "current_state_s3",
        "current_state_s4",
        "prev_state_s1",
        "prev_state_s2",
        "prev_state_s3",
        "prev_state_s4",
        "transition_s1_to_s4",
        "transition_s2_to_s4",
        "transition_s3_to_s4",
        "first_session",
        "log_duration_minutes",
        "log_gap_days",
        "log_player_lifetime_days",
        "log_prior_session_count",
        "log_prior_s4_count",
        "job_completion_ratio_filled",
        "job_exit_ratio_filled",
        "mode_diversity",
        "is_update_window_2023_01_31_to_2023_02_07",
        "is_minimal_cluster_purchase_anomaly",
    ]
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=int)
    x = pd.concat([df[base_cols].astype(float), month_dummies.astype(float)], axis=1)
    # Drop exact zero-variance columns inside each analysis sample.
    nunique = x.nunique(dropna=False)
    x = x.loc[:, nunique > 1]
    # Standardize continuous non-binary controls for numerical stability; binary dummies remain 0/1.
    continuous = [
        "log_duration_minutes",
        "log_gap_days",
        "log_player_lifetime_days",
        "log_prior_session_count",
        "log_prior_s4_count",
        "job_completion_ratio_filled",
        "job_exit_ratio_filled",
        "mode_diversity",
    ]
    for col in continuous:
        if col in x.columns:
            sd = x[col].std(ddof=1)
            if sd > 0:
                x[col] = (x[col] - x[col].mean()) / sd
    x.insert(0, "intercept", 1.0)
    return x


def logistic_irls(x_df, y, groups, max_iter=80, tol=1e-7):
    x = x_df.to_numpy(dtype=float)
    y = y.astype(float)
    n, p = x.shape
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
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = np.clip(x @ beta, -35, 35)
    mu = 1 / (1 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-8, None)
    bread = np.linalg.inv((x.T * w) @ x + penalty)

    score = x * (y - mu)[:, None]
    score_df = pd.DataFrame(score, columns=x_df.columns)
    score_df["pid"] = groups.to_numpy()
    summed = score_df.groupby("pid", sort=False).sum().drop(columns=[], errors="ignore")
    if "pid" in summed.columns:
        summed = summed.drop(columns=["pid"])
    s = summed.to_numpy(dtype=float)
    meat = s.T @ s
    cov_cluster = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov_cluster), 0))

    return beta, se, mu, iteration


def fit_one(df, threshold, analysis_name, exclude_update_window=False):
    work = df[df[f"evaluable_{threshold}d"].astype(bool)].copy()
    if exclude_update_window:
        work = work[
            ~work["login_time_utc"].dt.normalize().between(UPDATE_WINDOW_START, UPDATE_WINDOW_END)
        ].copy()
    y = work[f"observed_disengaged_{threshold}d"].astype(bool).astype(int).to_numpy()
    x = design_matrix(work)
    beta, se, mu, iterations = logistic_irls(x, y, work["pid"])
    rows = []
    for term, coef, term_se in zip(x.columns, beta, se):
        z = coef / term_se if term_se > 0 else np.nan
        rows.append(
            {
                "analysis": analysis_name,
                "threshold_days": threshold,
                "term": term,
                "coef_log_odds": float(coef),
                "cluster_robust_se_by_player": float(term_se),
                "z": float(z),
                "p_value": float(normal_p_value(z)) if np.isfinite(z) else np.nan,
                "adjusted_odds_ratio": float(np.exp(coef)),
                "or_ci_low_95": float(np.exp(coef - 1.96 * term_se)),
                "or_ci_high_95": float(np.exp(coef + 1.96 * term_se)),
            }
        )
    summary = {
        "analysis": analysis_name,
        "threshold_days": threshold,
        "sessions": int(len(work)),
        "players": int(work["pid"].nunique()),
        "events": int(y.sum()),
        "event_rate_pct": float(y.mean() * 100),
        "predictors_including_intercept": int(x.shape[1]),
        "irls_iterations": int(iterations),
        "mean_predicted_risk_pct": float(mu.mean() * 100),
        "s4_persistence_sessions": int(work["s4_persistence"].sum()),
        "s4_persistence_events": int(work.loc[work["s4_persistence"].eq(1), f"observed_disengaged_{threshold}d"].sum()),
        "s4_persistence_unadjusted_risk_pct": float(
            work.loc[work["s4_persistence"].eq(1), f"observed_disengaged_{threshold}d"].mean() * 100
        ),
    }
    dummy_export_cols = [
        "session_id",
        "pid",
        "session_index",
        f"observed_disengaged_{threshold}d",
        *[c for c in x.columns if c != "intercept"],
    ]
    dummies = pd.concat([work[["session_id", "pid", "session_index", f"observed_disengaged_{threshold}d"]], x.drop(columns=["intercept"])], axis=1)
    return pd.DataFrame(rows), summary, dummies


def write_report(summary_df, coef_df):
    focus = coef_df[coef_df["term"].eq("s4_persistence")].copy()
    lines = ["# Step 5.2 Logistic Regression Disengagement Risk", ""]
    lines.append("Outcome: observed Delta-day disengagement among evaluable, non-censored sessions.")
    lines.append("Focal predictor: `s4_persistence = previous state S4 and current state S4`.")
    lines.append("Standard errors are player-cluster robust. Continuous controls are standardized inside each analysis sample.")
    lines.append("")
    lines.append("## Analysis Samples")
    lines.append("| analysis | threshold | sessions | players | events | event rate | S4 persistence sessions | S4 persistence unadjusted risk |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_df.itertuples(index=False):
        lines.append(
            f"| {r.analysis} | {r.threshold_days}d | {r.sessions} | {r.players} | {r.events} | "
            f"{r.event_rate_pct:.2f}% | {r.s4_persistence_sessions} | {r.s4_persistence_unadjusted_risk_pct:.2f}% |"
        )
    lines.append("")
    lines.append("## Adjusted S4 Persistence Effect")
    lines.append("| analysis | threshold | adjusted OR | 95% CI | p-value |")
    lines.append("|---|---:|---:|---|---:|")
    for r in focus.itertuples(index=False):
        lines.append(
            f"| {r.analysis} | {r.threshold_days}d | {r.adjusted_odds_ratio:.3f} | "
            f"{r.or_ci_low_95:.3f}-{r.or_ci_high_95:.3f} | {r.p_value:.3g} |"
        )
    lines.append("")
    main = focus[(focus["analysis"] == "main_30d") & (focus["threshold_days"] == 30)].iloc[0]
    exclude = focus[focus["analysis"].eq("exclude_update_window_30d")].iloc[0]
    sens7 = focus[focus["analysis"].eq("sensitivity_7d")].iloc[0]
    sens14 = focus[focus["analysis"].eq("sensitivity_14d")].iloc[0]
    lines.append("## Answer")
    if (
        main.adjusted_odds_ratio > 1
        and main.p_value < 0.05
        and exclude.adjusted_odds_ratio > 1
        and exclude.p_value < 0.05
        and sens7.adjusted_odds_ratio > 1
        and sens14.adjusted_odds_ratio > 1
    ):
        lines.append(
            "Yes. S4 persistence remains significantly associated with higher disengagement risk after controls, and the direction is stable in 7d, 14d, and exclude-update-window sensitivity checks."
        )
    else:
        lines.append(
            "Not conclusively. S4 persistence does not remain consistently positive and significant across the adjusted robustness checks."
        )
    lines.append("")
    lines.append("## Controls")
    lines.append(
        "Models include current-state dummies, previous-state dummies, S1/S2/S3-to-S4 transition dummies, lifecycle/recency/history controls, month dummies, update-window flag, and minimal-purchase-anomaly flag where estimable. S4-to-S1/S2 recovery paths are left to descriptive transition analysis, not included in the focal persistence regression."
    )
    (OUT_DIR / "step5_2_logistic_disengagement_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_model_frame()
    analyses = [
        ("main_30d", 30, False),
        ("sensitivity_7d", 7, False),
        ("sensitivity_14d", 14, False),
        ("exclude_update_window_30d", 30, True),
    ]
    coef_frames = []
    summaries = []
    for name, threshold, exclude in analyses:
        print(f"fit {name}", flush=True)
        coef, summary, dummies = fit_one(df, threshold, name, exclude_update_window=exclude)
        coef_frames.append(coef)
        summaries.append(summary)
        dummies.to_csv(OUT_DIR / f"{name}_model_frame_dummies.csv", index=False, encoding="utf-8-sig")
    coef_df = pd.concat(coef_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    coef_df.to_csv(OUT_DIR / "logistic_coefficients_adjusted_odds_ratios.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "logistic_model_sample_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary_df, coef_df)
    focus = coef_df[coef_df["term"].eq("s4_persistence")][
        ["analysis", "threshold_days", "adjusted_odds_ratio", "or_ci_low_95", "or_ci_high_95", "p_value"]
    ]
    print(focus.to_string(index=False))
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
