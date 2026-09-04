from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "step5_disengagement_risk" / "session_disengagement_labels.csv"
FEATURES_PATH = ROOT / "step2_v1" / "session_features_v1.csv"
OUT_DIR = ROOT / "step5_6_recovery_tipping_point"

DELTAS = (30, 14, 7)
UPDATE_WINDOW_START = pd.Timestamp("2023-01-31")
UPDATE_WINDOW_END = pd.Timestamp("2023-02-07")
MIN_GROUP_N = 200
RIDGE = 1e-6


def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(series).astype(bool)


def normal_p_value(z: float) -> float:
    return erfc(abs(z) / sqrt(2.0))


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def safe_or(p_high: float, p_low: float) -> float:
    eps = 1e-9
    oh = (p_high + eps) / (1 - p_high + eps)
    ol = (p_low + eps) / (1 - p_low + eps)
    return float(oh / ol)


def load_s4_frame() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_csv(
        FEATURES_PATH,
        usecols=[
            "session_id",
            "pid",
            "session_index",
            "days_since_previous_session",
            "player_lifetime_days",
        ],
    )
    df = labels.merge(features, on=["session_id", "pid", "session_index"], how="left", suffixes=("", "_feature"))
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["state_id"] = df["state_id"].astype(int)
    df["next_state_id"] = pd.to_numeric(df["next_state_id"], errors="coerce")
    for delta in DELTAS:
        for col in [f"evaluable_{delta}d", f"censored_{delta}d", f"observed_disengaged_{delta}d"]:
            df[col] = boolify(df[col])
    df["is_update_window_2023_01_31_to_2023_02_07"] = boolify(df["is_update_window_2023_01_31_to_2023_02_07"])

    df = df.sort_values(["pid", "session_index"]).copy()
    df["total_observed_sessions"] = df.groupby("pid")["session_index"].transform("max")
    df["observed_lifecycle_position"] = df["session_index"] / df["total_observed_sessions"]
    df["prior_session_count"] = df["session_index"] - 1
    df["prior_s4_count"] = df.groupby("pid")["state_id"].transform(lambda s: s.eq(4).cumsum().shift(1, fill_value=0))
    df["prior_s4_share"] = (df["prior_s4_count"] / df["prior_session_count"].replace(0, np.nan)).fillna(0)
    for w in (3, 5, 10):
        df[f"recent_s4_density_prev{w}"] = (
            df.groupby("pid")["state_id"]
            .transform(lambda s, w=w: s.eq(4).shift(1, fill_value=False).rolling(w, min_periods=1).mean())
            .fillna(0)
        )
    df["gap_before_s4_days"] = df["days_since_previous_session"].fillna(0).clip(lower=0)
    df["log_gap_before_s4_days"] = np.log1p(df["gap_before_s4_days"])
    df["log_player_lifetime_days"] = np.log1p(df["player_lifetime_days"].fillna(0).clip(lower=0))
    df["log_prior_session_count"] = np.log1p(df["prior_session_count"].clip(lower=0))
    df["log_prior_s4_count"] = np.log1p(df["prior_s4_count"].clip(lower=0))
    df["month"] = df["login_time_utc"].dt.to_period("M").astype(str)

    s4 = df[df["state_id"].eq(4)].copy()
    for delta in DELTAS:
        next_gap = pd.to_numeric(s4["next_session_gap_days"], errors="coerce")
        s4[f"recovered_to_s1_s2_within_{delta}d"] = (
            s4["next_state_id"].isin([1, 2])
            & next_gap.notna()
            & next_gap.le(delta)
            & s4[f"evaluable_{delta}d"]
        )
        s4[f"gap_after_s4_{delta}d_capped"] = next_gap.fillna(delta).clip(lower=0, upper=delta)
    return s4


def variable_specs(delta: int) -> list[dict]:
    return [
        {
            "variable": "gap_before_s4_days",
            "label": "gap before S4",
            "timing": "ex_ante",
            "higher_means": "longer pre-S4 gap",
        },
        {
            "variable": f"gap_after_s4_{delta}d_capped",
            "label": f"gap after S4 capped at {delta}d",
            "timing": "outcome_proximal",
            "higher_means": "slower/no return after S4",
        },
        {
            "variable": "recent_s4_density_prev5",
            "label": "recent S4 density prev5",
            "timing": "ex_ante",
            "higher_means": "more recent S4 exposure",
        },
        {
            "variable": "recent_s4_density_prev10",
            "label": "recent S4 density prev10",
            "timing": "ex_ante",
            "higher_means": "more recent S4 exposure",
        },
        {
            "variable": "prior_s4_share",
            "label": "prior S4 share",
            "timing": "ex_ante",
            "higher_means": "more historical S4 exposure",
        },
        {
            "variable": "observed_lifecycle_position",
            "label": "observed lifecycle position",
            "timing": "descriptive_hindsight",
            "higher_means": "later in observed trajectory",
        },
        {
            "variable": "player_lifetime_days",
            "label": "player lifetime days",
            "timing": "ex_ante",
            "higher_means": "later in calendar lifetime",
        },
        {
            "variable": "prior_session_count",
            "label": "prior session count",
            "timing": "ex_ante",
            "higher_means": "later in play history",
        },
    ]


def threshold_scan(df: pd.DataFrame, variable: str, outcome: str, min_group_n=MIN_GROUP_N) -> pd.DataFrame:
    work = df[[variable, outcome, "pid"]].dropna().copy()
    if work.empty or work[variable].nunique() < 2:
        return pd.DataFrame()
    qs = np.arange(0.10, 0.91, 0.05)
    thresholds = sorted(set(float(work[variable].quantile(q)) for q in qs))
    rows = []
    for t in thresholds:
        low = work[work[variable].lt(t)]
        high = work[work[variable].ge(t)]
        if len(low) < min_group_n or len(high) < min_group_n:
            continue
        p_low = float(low[outcome].mean())
        p_high = float(high[outcome].mean())
        rows.append(
            {
                "variable": variable,
                "threshold": t,
                "n_low": int(len(low)),
                "n_high": int(len(high)),
                "players_low": int(low["pid"].nunique()),
                "players_high": int(high["pid"].nunique()),
                "recovery_rate_below_threshold": p_low,
                "recovery_rate_at_or_above_threshold": p_high,
                "collapse_pp": float(p_low - p_high),
                "odds_ratio_high_vs_low": safe_or(p_high, p_low),
            }
        )
    return pd.DataFrame(rows)


def bin_summary(df: pd.DataFrame, variable: str, outcome: str, bins=8) -> pd.DataFrame:
    work = df[[variable, outcome, "pid"]].dropna().copy()
    if work.empty or work[variable].nunique() < 2:
        return pd.DataFrame()
    try:
        work["bin"] = pd.qcut(work[variable], q=bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    out = (
        work.groupby("bin", observed=True)
        .agg(
            sessions=(outcome, "size"),
            players=("pid", "nunique"),
            variable_min=(variable, "min"),
            variable_max=(variable, "max"),
            variable_median=(variable, "median"),
            recovery_rate=(outcome, "mean"),
        )
        .reset_index()
    )
    out["variable"] = variable
    out["bin"] = out["bin"].astype(str)
    return out


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
    return beta, se, mu, iteration


def design_matrix(df: pd.DataFrame, tipping_terms: list[str]) -> pd.DataFrame:
    base_cols = tipping_terms + [
        "log_gap_before_s4_days",
        "log_player_lifetime_days",
        "log_prior_session_count",
        "log_prior_s4_count",
        "prior_s4_share",
        "is_update_window_2023_01_31_to_2023_02_07",
    ]
    x = df[base_cols].astype(float)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=int)
    x = pd.concat([x, month_dummies.astype(float)], axis=1)
    nunique = x.nunique(dropna=False)
    x = x.loc[:, nunique > 1]
    continuous = [
        "log_gap_before_s4_days",
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


def fit_logistic(df: pd.DataFrame, delta: int, thresholds: dict[str, float], analysis_name: str, exclude_update=False):
    work = df[df[f"evaluable_{delta}d"]].copy()
    if exclude_update:
        work = work[
            ~work["login_time_utc"].dt.normalize().between(UPDATE_WINDOW_START, UPDATE_WINDOW_END)
        ].copy()
    outcome = f"recovered_to_s1_s2_within_{delta}d"
    tipping_terms = []
    for var, threshold in thresholds.items():
        term = f"tipping_high_{var}"
        work[term] = work[var].ge(threshold).astype(int)
        tipping_terms.append(term)
    y = work[outcome].astype(int).to_numpy()
    x = design_matrix(work, tipping_terms)
    beta, se, mu, iterations = logistic_irls(x, y, work["pid"])
    rows = []
    for term, coef, term_se in zip(x.columns, beta, se):
        z = coef / term_se if term_se > 0 else np.nan
        rows.append(
            {
                "analysis": analysis_name,
                "threshold_days": delta,
                "term": term,
                "coef_log_odds": float(coef),
                "cluster_robust_se_by_player": float(term_se),
                "z": float(z) if pd.notna(z) else np.nan,
                "p_value": float(normal_p_value(z)) if pd.notna(z) else np.nan,
                "adjusted_odds_ratio_for_recovery": float(np.exp(coef)),
                "or_ci_low_95": float(np.exp(coef - 1.96 * term_se)),
                "or_ci_high_95": float(np.exp(coef + 1.96 * term_se)),
                "n_sessions": int(len(work)),
                "n_players": int(work["pid"].nunique()),
                "n_recovered": int(y.sum()),
                "recovery_rate": float(y.mean()),
                "irls_iterations": int(iterations),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s4 = load_s4_frame()
    s4.to_csv(OUT_DIR / "s4_recovery_tipping_point_base.csv", index=False, encoding="utf-8-sig")

    all_scans = []
    all_bins = []
    best_rows = []
    for delta in DELTAS:
        sample = s4[s4[f"evaluable_{delta}d"]].copy()
        outcome = f"recovered_to_s1_s2_within_{delta}d"
        for spec in variable_specs(delta):
            scan = threshold_scan(sample, spec["variable"], outcome)
            if not scan.empty:
                scan.insert(0, "threshold_days", delta)
                scan["label"] = spec["label"]
                scan["timing"] = spec["timing"]
                scan["higher_means"] = spec["higher_means"]
                all_scans.append(scan)
                best = scan.sort_values("collapse_pp", ascending=False).iloc[0].to_dict()
                best_rows.append(best)
            bins = bin_summary(sample, spec["variable"], outcome)
            if not bins.empty:
                bins.insert(0, "threshold_days", delta)
                bins["label"] = spec["label"]
                bins["timing"] = spec["timing"]
                all_bins.append(bins)

    scans = pd.concat(all_scans, ignore_index=True)
    bins = pd.concat(all_bins, ignore_index=True)
    best = pd.DataFrame(best_rows)
    scans.to_csv(OUT_DIR / "recovery_probability_threshold_scan.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(OUT_DIR / "recovery_probability_binned_trends.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "recovery_probability_best_thresholds.csv", index=False, encoding="utf-8-sig")

    main_best = best[best["threshold_days"].eq(30)].copy()
    # Keep the primary adjusted model focused on ex-ante tipping signals.
    wanted = ["gap_before_s4_days", "recent_s4_density_prev5", "prior_s4_share", "prior_session_count"]
    thresholds = {
        row["variable"]: float(row["threshold"])
        for _, row in main_best[main_best["variable"].isin(wanted)].iterrows()
    }
    coef_frames = [
        fit_logistic(s4, 30, thresholds, "main_30d", exclude_update=False),
        fit_logistic(s4, 14, thresholds, "sensitivity_14d", exclude_update=False),
        fit_logistic(s4, 7, thresholds, "sensitivity_7d", exclude_update=False),
        fit_logistic(s4, 30, thresholds, "exclude_update_window_30d", exclude_update=True),
    ]
    coefs = pd.concat(coef_frames, ignore_index=True)
    coefs.to_csv(OUT_DIR / "recovery_tipping_point_adjusted_logistic.csv", index=False, encoding="utf-8-sig")

    stability_rows = []
    for _, row in main_best.iterrows():
        variable = row["variable"]
        threshold = float(row["threshold"])
        for delta in DELTAS:
            for exclude in [False, True]:
                sample = s4[s4[f"evaluable_{delta}d"]].copy()
                if exclude:
                    sample = sample[
                        ~sample["login_time_utc"].dt.normalize().between(UPDATE_WINDOW_START, UPDATE_WINDOW_END)
                    ].copy()
                if len(sample) < MIN_GROUP_N * 2:
                    continue
                low = sample[sample[variable].lt(threshold)]
                high = sample[sample[variable].ge(threshold)]
                outcome = f"recovered_to_s1_s2_within_{delta}d"
                if len(low) < MIN_GROUP_N or len(high) < MIN_GROUP_N:
                    continue
                p_low = float(low[outcome].mean())
                p_high = float(high[outcome].mean())
                stability_rows.append(
                    {
                        "source_threshold_days": 30,
                        "evaluated_threshold_days": delta,
                        "exclude_update_window": exclude,
                        "variable": variable,
                        "threshold": threshold,
                        "n_low": int(len(low)),
                        "n_high": int(len(high)),
                        "recovery_rate_below_threshold": p_low,
                        "recovery_rate_at_or_above_threshold": p_high,
                        "collapse_pp": p_low - p_high,
                        "odds_ratio_high_vs_low": safe_or(p_high, p_low),
                    }
                )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT_DIR / "recovery_tipping_point_threshold_stability.csv", index=False, encoding="utf-8-sig")

    focal_terms = [f"tipping_high_{v}" for v in wanted if f"tipping_high_{v}" in set(coefs["term"])]
    focal = coefs[coefs["term"].isin(focal_terms)].copy()
    display_best = main_best.sort_values(["timing", "collapse_pp"], ascending=[True, False])

    lines = [
        "# Step 5.6 Recovery Threshold Scan",
        "",
        "## Scope",
        "",
        "- Unit: S4 origin session.",
        "- Main outcome: recovery to S1/S2 within 30 days after S4, among 30d-evaluable S4 origins.",
        "- Sensitivities: 14d, 7d, and 30d excluding Jan31-Feb7.",
        "- Threshold scan: quantile thresholds from P10 to P90, requiring at least 200 sessions on each side.",
        "- Higher-side difference means recovery probability is lower at or above the threshold than below it.",
        "",
        "## Best 30d Thresholds",
        "",
        "| variable | timing | threshold | recovery below | recovery at/above | difference pp | OR high vs low |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in display_best.itertuples(index=False):
        lines.append(
            f"| {r.variable} | {r.timing} | {r.threshold:.4g} | "
            f"{pct(r.recovery_rate_below_threshold)} | {pct(r.recovery_rate_at_or_above_threshold)} | "
            f"{100*r.collapse_pp:.2f} | {r.odds_ratio_high_vs_low:.3f} |"
        )

    lines += [
        "",
        "## Adjusted Ex-Ante Threshold Model",
        "",
        "The adjusted model includes only ex-ante threshold dummies plus continuous controls for pre-S4 gap, lifetime, prior sessions, prior S4 count/share, month, and update-window. Outcome is recovery to S1/S2; OR below 1 means lower recoverability.",
        "",
        "| analysis | term | adjusted OR for recovery | 95% CI | p-value |",
        "|---|---|---:|---|---:|",
    ]
    for r in focal.itertuples(index=False):
        lines.append(
            f"| {r.analysis} | {r.term} | {r.adjusted_odds_ratio_for_recovery:.3f} | "
            f"{r.or_ci_low_95:.3f}-{r.or_ci_high_95:.3f} | {r.p_value:.3g} |"
        )

    strongest = focal[focal["analysis"].eq("main_30d")].sort_values("adjusted_odds_ratio_for_recovery").head(1)
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if not strongest.empty:
        s = strongest.iloc[0]
        lines.append(
            f"- Strongest adjusted ex-ante threshold in the main model: `{s.term}` with OR {s.adjusted_odds_ratio_for_recovery:.3f} for recovery."
        )
    lines.append(
        "- `gap_after_s4` is outcome-proximal, so it is useful for describing post-S4 return timing but should not be treated as an early warning signal."
    )
    stable_prior = focal[
        focal["term"].str.contains("prior_s4_share|recent_s4_density")
        & focal["adjusted_odds_ratio_for_recovery"].lt(1)
        & focal["p_value"].lt(0.05)
    ]
    if not stable_prior.empty:
        lines.append(
            "- The recoverability signal is strongest when S4 becomes dense in recent/history behavior, supporting a progressive recoverability-decline interpretation better than a simple S4 count."
        )
    else:
        lines.append(
            "- The adjusted evidence does not isolate a uniquely stable ex-ante threshold from S4 density/history alone."
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- `s4_recovery_tipping_point_base.csv`",
        "- `recovery_probability_binned_trends.csv`",
        "- `recovery_probability_threshold_scan.csv`",
        "- `recovery_probability_best_thresholds.csv`",
        "- `recovery_tipping_point_threshold_stability.csv`",
        "- `recovery_tipping_point_adjusted_logistic.csv`",
    ]
    (OUT_DIR / "step5_6_recovery_tipping_point_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(focal[["analysis", "term", "adjusted_odds_ratio_for_recovery", "or_ci_low_95", "or_ci_high_95", "p_value"]].to_string(index=False))
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
