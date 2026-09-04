from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "step5_disengagement_risk" / "session_disengagement_labels.csv"
FEATURES_PATH = ROOT / "step2_v1" / "session_features_v1.csv"
OUT_DIR = ROOT / "step5_3_adjusted_recovery_path_model"

UPDATE_WINDOW_START = pd.Timestamp("2023-01-31")
UPDATE_WINDOW_END = pd.Timestamp("2023-02-07")
RIDGE = 1e-6


def normal_p_value(z: float) -> float:
    return erfc(abs(z) / sqrt(2.0))


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def load_frame() -> pd.DataFrame:
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
            "mode_diversity",
        ],
    )
    df = labels.merge(features, on=["session_id", "pid", "session_index"], how="left", suffixes=("", "_v1"))
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["month"] = df["login_time_utc"].dt.to_period("M").astype(str)
    df["state_id"] = df["state_id"].astype(int)
    df["prev_state_id"] = pd.to_numeric(df["prev_state_id"], errors="coerce")

    for col in [
        "is_update_window_2023_01_31_to_2023_02_07",
        "is_minimal_cluster_purchase_anomaly",
    ]:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(df[col]).astype(bool).astype(int)
    for delta in (7, 14, 30):
        for col in [f"evaluable_{delta}d", f"censored_{delta}d", f"observed_disengaged_{delta}d"]:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(df[col]).astype(bool)

    df = df.sort_values(["pid", "session_index"]).copy()
    df["prior_session_count"] = df["session_index"] - 1
    df["prior_s4_count"] = df.groupby("pid")["state_id"].transform(lambda s: s.eq(4).cumsum().shift(1, fill_value=0))
    df["prior_s4_share"] = df["prior_s4_count"] / df["prior_session_count"].replace(0, np.nan)
    df["prior_s4_share"] = df["prior_s4_share"].fillna(0)

    df["log_gap_days"] = np.log1p(df["days_since_previous_session"].clip(lower=0).fillna(0))
    df["log_player_lifetime_days"] = np.log1p(df["player_lifetime_days"].clip(lower=0).fillna(0))
    df["log_prior_session_count"] = np.log1p(df["prior_session_count"].clip(lower=0))
    df["log_prior_s4_count"] = np.log1p(df["prior_s4_count"].clip(lower=0))
    df["log_duration_minutes"] = np.log1p(df["duration_minutes"].clip(lower=0))
    df["job_completion_ratio_filled"] = df["job_completion_ratio"].clip(lower=0, upper=1).fillna(0)
    df["job_exit_ratio_filled"] = df["job_exit_ratio"].clip(lower=0, upper=1).fillna(0)
    df["mode_diversity"] = df["mode_diversity"].fillna(0)

    # Analysis sample: destination sessions after S4, comparing recovery to S1/S2 against S4 persistence.
    df = df[df["prev_state_id"].eq(4) & df["state_id"].isin([1, 2, 4])].copy()
    df["recovery_path"] = df["state_id"].isin([1, 2]).astype(int)
    df["path_label"] = np.where(df["recovery_path"].eq(1), "S4->S1/S2", "S4->S4")
    return df


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "recovery_path",
        "log_gap_days",
        "log_player_lifetime_days",
        "log_prior_session_count",
        "log_prior_s4_count",
        "prior_s4_share",
        "is_update_window_2023_01_31_to_2023_02_07",
    ]
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=int)
    x = pd.concat([df[base_cols].astype(float), month_dummies.astype(float)], axis=1)
    nunique = x.nunique(dropna=False)
    x = x.loc[:, nunique > 1]

    continuous = [
        "log_gap_days",
        "log_player_lifetime_days",
        "log_prior_session_count",
        "log_prior_s4_count",
        "prior_s4_share",
    ]
    for col in continuous:
        if col in x.columns:
            sd = x[col].std(ddof=1)
            if sd > 0:
                x[col] = (x[col] - x[col].mean()) / sd
    x.insert(0, "intercept", 1.0)
    return x


def logistic_irls(x_df: pd.DataFrame, y: np.ndarray, groups: pd.Series, max_iter=100, tol=1e-7):
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
    cov_cluster = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov_cluster), 0))
    return beta, se, mu, iteration


def unadjusted_path_summary(work: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rows = []
    for label, g in work.groupby("path_label"):
        y = g[f"observed_disengaged_{threshold}d"].astype(bool)
        rows.append(
            {
                "path_label": label,
                "sessions": int(len(g)),
                "players": int(g["pid"].nunique()),
                "events": int(y.sum()),
                "risk": float(y.mean()),
                "median_gap_to_destination_days": float(g["days_since_previous_session"].median()),
                "median_duration_minutes": float(g["duration_minutes"].median()),
                "update_window_share": float(g["is_update_window_2023_01_31_to_2023_02_07"].mean()),
                "median_prior_s4_count": float(g["prior_s4_count"].median()),
            }
        )
    return pd.DataFrame(rows)


def fit_one(df: pd.DataFrame, threshold: int, analysis_name: str, exclude_update_window=False):
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
        "event_rate": float(y.mean()),
        "recovery_sessions": int(work["recovery_path"].sum()),
        "persistence_sessions": int(work["recovery_path"].eq(0).sum()),
        "recovery_players": int(work.loc[work["recovery_path"].eq(1), "pid"].nunique()),
        "persistence_players": int(work.loc[work["recovery_path"].eq(0), "pid"].nunique()),
        "predictors_including_intercept": int(x.shape[1]),
        "irls_iterations": int(iterations),
        "mean_predicted_risk": float(mu.mean()),
    }

    unadj = unadjusted_path_summary(work, threshold)
    unadj.insert(0, "analysis", analysis_name)
    unadj.insert(1, "threshold_days", threshold)

    model_frame = pd.concat(
        [
            work[
                [
                    "session_id",
                    "pid",
                    "session_index",
                    "login_time_utc",
                    "state_id",
                    "prev_state_id",
                    "path_label",
                    f"observed_disengaged_{threshold}d",
                ]
            ],
            x.drop(columns=["intercept"]),
        ],
        axis=1,
    )
    return pd.DataFrame(rows), summary, unadj, model_frame


def write_report(summary_df: pd.DataFrame, coef_df: pd.DataFrame, unadj_df: pd.DataFrame) -> None:
    focus = coef_df[coef_df["term"].eq("recovery_path")].copy()
    lines = [
        "# Step 5.3 Adjusted Recovery Path Model",
        "",
        "## Scope",
        "",
        "- Unit of analysis: destination session after an S4 origin.",
        "- Sample: only `S4->S1/S2` recovery and `S4->S4` persistence paths.",
        "- Outcome: observed disengagement after the destination session, with the same right-censoring rule as Step 5.",
        "- Predictor: `recovery_path = 1` for `S4->S1/S2`; reference is `S4->S4`.",
        "- Standard errors are clustered by player.",
        "",
        "## Analysis Samples",
        "",
        "| analysis | threshold | sessions | players | events | event rate | recovery sessions | persistence sessions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_df.itertuples(index=False):
        lines.append(
            f"| {r.analysis} | {r.threshold_days}d | {r.sessions:,} | {r.players:,} | {r.events:,} | "
            f"{pct(r.event_rate)} | {r.recovery_sessions:,} | {r.persistence_sessions:,} |"
        )

    lines += [
        "",
        "## Unadjusted Destination Risk",
        "",
        "| analysis | path | sessions | players | risk | median gap to destination days | update-window share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in unadj_df[unadj_df["threshold_days"].isin([30])].itertuples(index=False):
        lines.append(
            f"| {r.analysis} | {r.path_label} | {r.sessions:,} | {r.players:,} | {pct(r.risk)} | "
            f"{r.median_gap_to_destination_days:.3f} | {pct(r.update_window_share)} |"
        )

    lines += [
        "",
        "## Adjusted Recovery Effect",
        "",
        "| analysis | threshold | adjusted OR for recovery | 95% CI | p-value |",
        "|---|---:|---:|---|---:|",
    ]
    for r in focus.itertuples(index=False):
        lines.append(
            f"| {r.analysis} | {r.threshold_days}d | {r.adjusted_odds_ratio:.3f} | "
            f"{r.or_ci_low_95:.3f}-{r.or_ci_high_95:.3f} | {r.p_value:.3g} |"
        )

    lines += [
        "",
        "## Controls",
        "",
        "Controls include recent gap to the destination session, player lifetime days, prior session count, prior S4 count, prior S4 share, month dummies, and Jan31-Feb7 update-window flag. Current-state, transition, and destination behavior features are intentionally excluded because the predictor itself defines the recovery-vs-persistence contrast and destination behavior can proxy the state label.",
        "",
        "## Answer",
        "",
    ]
    main = focus[focus["analysis"].eq("main_30d")].iloc[0]
    excl = focus[focus["analysis"].eq("exclude_update_window_30d")].iloc[0]
    if main.adjusted_odds_ratio < 1 and main.p_value < 0.05:
        lines.append(
            f"In the main 30d model, recovery to S1/S2 is associated with significantly lower disengagement risk than S4 persistence (adjusted OR {main.adjusted_odds_ratio:.3f}, 95% CI {main.or_ci_low_95:.3f}-{main.or_ci_high_95:.3f}, p={main.p_value:.3g})."
        )
    else:
        lines.append(
            f"In the main 30d model, the adjusted recovery effect is not a clear significant reduction (adjusted OR {main.adjusted_odds_ratio:.3f}, p={main.p_value:.3g})."
        )
    if excl.adjusted_odds_ratio < 1 and excl.p_value < 0.05:
        lines.append(
            "The direction and significance survive excluding the Jan31-Feb7 update window."
        )
    else:
        lines.append(
            "The exclude-update-window robustness check weakens the evidence, so interpret the adjusted recovery effect with that caveat."
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- `adjusted_recovery_path_coefficients.csv`",
        "- `adjusted_recovery_path_model_summary.csv`",
        "- `adjusted_recovery_path_unadjusted_risk.csv`",
        "- `*_model_frame.csv`",
    ]
    (OUT_DIR / "step5_3_adjusted_recovery_path_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    analyses = [
        ("main_30d", 30, False),
        ("sensitivity_7d", 7, False),
        ("sensitivity_14d", 14, False),
        ("exclude_update_window_30d", 30, True),
    ]
    coef_frames = []
    summaries = []
    unadj_frames = []
    for name, threshold, exclude_update_window in analyses:
        print(f"fit {name}", flush=True)
        coef, summary, unadj, model_frame = fit_one(df, threshold, name, exclude_update_window)
        coef_frames.append(coef)
        summaries.append(summary)
        unadj_frames.append(unadj)
        model_frame.to_csv(OUT_DIR / f"{name}_model_frame.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.concat(coef_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    unadj_df = pd.concat(unadj_frames, ignore_index=True)

    coef_df.to_csv(OUT_DIR / "adjusted_recovery_path_coefficients.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "adjusted_recovery_path_model_summary.csv", index=False, encoding="utf-8-sig")
    unadj_df.to_csv(OUT_DIR / "adjusted_recovery_path_unadjusted_risk.csv", index=False, encoding="utf-8-sig")
    write_report(summary_df, coef_df, unadj_df)

    focus = coef_df[coef_df["term"].eq("recovery_path")][
        ["analysis", "threshold_days", "adjusted_odds_ratio", "or_ci_low_95", "or_ci_high_95", "p_value"]
    ]
    print(focus.to_string(index=False))
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
