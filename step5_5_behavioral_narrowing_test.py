from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "step5_disengagement_risk" / "session_disengagement_labels.csv"
FEATURES_PATH = ROOT / "step2_v1" / "session_features_v1.csv"
OUT_DIR = ROOT / "step5_5_behavioral_narrowing_test"

WINDOWS = (3, 5)
METRICS = ("state_entropy", "unique_states", "dominant_state_share")
SLOPE_HORIZON_SESSIONS = 10
MIN_POINTS_FOR_SLOPE = 5
RIDGE = 1e-8


def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(series).astype(bool)


def normal_p_value(z: float) -> float:
    return erfc(abs(z) / sqrt(2.0))


def pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{100 * x:.2f}%"


def entropy_bits(states: np.ndarray) -> float:
    if len(states) == 0:
        return np.nan
    _, counts = np.unique(states, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def rolling_metrics_for_player(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("session_index").copy()
    states = g["state_id"].to_numpy(dtype=int)
    out = {
        "session_id": g["session_id"].to_numpy(),
        "pid": g["pid"].to_numpy(),
        "session_index": g["session_index"].to_numpy(),
        "login_time_utc": g["login_time_utc"].to_numpy(),
        "state_id": states,
    }
    for w in WINDOWS:
        entropy_vals, unique_vals, dominant_vals = [], [], []
        for i in range(len(states)):
            window = states[max(0, i - w + 1) : i + 1]
            entropy_vals.append(entropy_bits(window))
            unique_vals.append(int(len(np.unique(window))))
            _, counts = np.unique(window, return_counts=True)
            dominant_vals.append(float(counts.max() / counts.sum()))
        out[f"state_entropy_w{w}"] = entropy_vals
        out[f"unique_states_w{w}"] = unique_vals
        out[f"dominant_state_share_w{w}"] = dominant_vals
    return pd.DataFrame(out)


def slope(y: np.ndarray, lag: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(lag)
    y = y[mask]
    lag = lag[mask]
    if len(y) < MIN_POINTS_FOR_SLOPE or len(np.unique(lag)) < 2:
        return np.nan
    x = lag - lag.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return np.nan
    return float((x * (y - y.mean())).sum() / denom)


def add_anchor_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby("pid", sort=False):
        g = g.sort_values("session_index").reset_index(drop=True)
        for anchor_pos in range(len(g)):
            if anchor_pos + 1 < MIN_POINTS_FOR_SLOPE:
                continue
            start = max(0, anchor_pos - SLOPE_HORIZON_SESSIONS)
            hist = g.iloc[start : anchor_pos + 1].copy()
            lag = (g.loc[anchor_pos, "session_index"] - hist["session_index"]).to_numpy(dtype=float)
            rec = {
                "pid": g.loc[anchor_pos, "pid"],
                "anchor_session_id": g.loc[anchor_pos, "session_id"],
                "anchor_session_index": int(g.loc[anchor_pos, "session_index"]),
                "anchor_login_time_utc": g.loc[anchor_pos, "login_time_utc"],
                "anchor_state_id": int(g.loc[anchor_pos, "state_id"]),
                "slope_points": int(len(hist)),
            }
            for w in WINDOWS:
                for metric in METRICS:
                    raw = slope(hist[f"{metric}_w{w}"].to_numpy(dtype=float), lag)
                    # Positive narrowing slope means lower diversity / higher concentration as lag approaches 0.
                    if metric in {"state_entropy", "unique_states"}:
                        rec[f"narrowing_slope_{metric}_w{w}"] = raw
                    else:
                        rec[f"narrowing_slope_{metric}_w{w}"] = -raw
            rows.append(rec)
    return pd.DataFrame(rows)


def session_index_bin(s: pd.Series) -> pd.Series:
    bins = [0, 5, 10, 20, 50, 100, np.inf]
    labels = ["5-10", "6-10", "11-20", "21-50", "51-100", "100+"]
    # First bin label includes anchors >=5 because slopes require history.
    return pd.cut(s, bins=bins, labels=labels, right=True)


def build_matched_sample(anchor_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = anchor_frame[
        anchor_frame["is_last_observed_session"]
        & anchor_frame["evaluable_30d"]
        & anchor_frame["observed_disengaged_30d"]
    ].copy()
    controls = anchor_frame[
        anchor_frame["evaluable_30d"]
        & ~anchor_frame["observed_disengaged_30d"]
        & ~anchor_frame["is_last_observed_session"]
    ].copy()

    slope_cols = [c for c in anchor_frame.columns if c.startswith("narrowing_slope_")]
    cases = cases.dropna(subset=slope_cols)
    controls = controls.dropna(subset=slope_cols)

    for part in (cases, controls):
        part["anchor_month"] = pd.to_datetime(part["anchor_login_time_utc"]).dt.to_period("M").astype(str)
        part["anchor_index_bin"] = session_index_bin(part["anchor_session_index"]).astype(str)
        part["match_stratum"] = part["anchor_month"] + "|" + part["anchor_index_bin"]

    cases = cases.sort_values(["match_stratum", "anchor_session_index", "pid", "anchor_session_id"]).copy()
    controls = controls.sort_values(["match_stratum", "anchor_session_index", "pid", "anchor_session_id"]).copy()
    matched_records = []

    for stratum, case_group in cases.groupby("match_stratum", sort=False):
        control_group = controls[controls["match_stratum"].eq(stratum)].copy()
        if control_group.empty:
            continue
        control_records = control_group.to_dict("records")
        pointer = 0
        for case in case_group.to_dict("records"):
            chosen = None
            while pointer < len(control_records):
                candidate = control_records[pointer]
                pointer += 1
                if candidate["pid"] != case["pid"]:
                    chosen = candidate
                    break
            if chosen is None:
                break
            matched_records.append(
                {
                    "match_id": len(matched_records) + 1,
                    "case_pid": case["pid"],
                    "case_anchor_session_id": case["anchor_session_id"],
                    "control_pid": chosen["pid"],
                    "control_anchor_session_id": chosen["anchor_session_id"],
                    "case_anchor_month": case["anchor_month"],
                    "control_anchor_month": chosen["anchor_month"],
                    "case_anchor_index": int(case["anchor_session_index"]),
                    "control_anchor_index": int(chosen["anchor_session_index"]),
                    "case_anchor_state_id": int(case["anchor_state_id"]),
                    "control_anchor_state_id": int(chosen["anchor_state_id"]),
                    "same_month": True,
                    "same_index_bin": True,
                    "used_fallback": False,
                }
            )

    matches = pd.DataFrame(matched_records)
    case_rows = cases.merge(matches[["match_id", "case_anchor_session_id"]], left_on="anchor_session_id", right_on="case_anchor_session_id", how="inner")
    control_rows = controls.merge(matches[["match_id", "control_anchor_session_id"]], left_on="anchor_session_id", right_on="control_anchor_session_id", how="inner")
    case_rows["terminal_case"] = 1
    control_rows["terminal_case"] = 0
    matched = pd.concat([case_rows, control_rows], ignore_index=True, sort=False)
    return matched, matches


def cluster_robust_ols(x_df: pd.DataFrame, y: np.ndarray, clusters: pd.Series):
    x = x_df.to_numpy(dtype=float)
    y = y.astype(float)
    p = x.shape[1]
    penalty = np.eye(p) * RIDGE
    penalty[0, 0] = 0.0
    xtx = x.T @ x + penalty
    beta = np.linalg.solve(xtx, x.T @ y)
    residual = y - x @ beta
    bread = np.linalg.inv(xtx)
    score = x * residual[:, None]
    score_df = pd.DataFrame(score, columns=x_df.columns)
    score_df["cluster"] = clusters.to_numpy()
    summed = score_df.groupby("cluster", sort=False).sum()
    s = summed.to_numpy(dtype=float)
    meat = s.T @ s
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se


def design_matrix(matched: pd.DataFrame) -> pd.DataFrame:
    x = matched[
        [
            "terminal_case",
            "log_anchor_session_index",
            "log_player_lifetime_days",
            "log_prior_session_count",
            "log_prior_s4_count",
            "prior_s4_share",
        ]
    ].astype(float)
    month_dummies = pd.get_dummies(matched["anchor_month"], prefix="month", drop_first=True, dtype=int)
    state_dummies = pd.get_dummies(matched["anchor_state_id"], prefix="anchor_state", drop_first=True, dtype=int)
    x = pd.concat([x, month_dummies.astype(float), state_dummies.astype(float)], axis=1)
    nunique = x.nunique(dropna=False)
    x = x.loc[:, nunique > 1]
    continuous = [
        "log_anchor_session_index",
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


def paired_comparisons(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        for metric in METRICS:
            col = f"narrowing_slope_{metric}_w{w}"
            wide = matched.pivot(index="match_id", columns="terminal_case", values=col).dropna()
            wide.columns = ["control", "case"]
            diff = wide["case"] - wide["control"]
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
            t = diff.mean() / se if se and se > 0 else np.nan
            rows.append(
                {
                    "window": w,
                    "metric": metric,
                    "matched_pairs": int(len(diff)),
                    "case_mean": float(wide["case"].mean()),
                    "control_mean": float(wide["control"].mean()),
                    "case_minus_control": float(diff.mean()),
                    "paired_se": float(se) if pd.notna(se) else np.nan,
                    "z_or_t": float(t) if pd.notna(t) else np.nan,
                    "p_value_normal_approx": float(normal_p_value(t)) if pd.notna(t) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def adjusted_ols(matched: pd.DataFrame) -> pd.DataFrame:
    x = design_matrix(matched)
    rows = []
    for w in WINDOWS:
        for metric in METRICS:
            col = f"narrowing_slope_{metric}_w{w}"
            work = matched.dropna(subset=[col]).copy()
            xw = x.loc[work.index]
            beta, se = cluster_robust_ols(xw, work[col].to_numpy(dtype=float), work["match_id"])
            for term, coef, term_se in zip(xw.columns, beta, se):
                z = coef / term_se if term_se > 0 else np.nan
                rows.append(
                    {
                        "window": w,
                        "metric": metric,
                        "term": term,
                        "coef": float(coef),
                        "cluster_robust_se_by_match": float(term_se),
                        "z": float(z) if pd.notna(z) else np.nan,
                        "p_value": float(normal_p_value(z)) if pd.notna(z) else np.nan,
                        "n_rows": int(len(work)),
                        "n_pairs": int(work["match_id"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_csv(
        FEATURES_PATH,
        usecols=[
            "session_id",
            "pid",
            "session_index",
            "player_lifetime_days",
        ],
    )
    df = labels.merge(features, on=["session_id", "pid", "session_index"], how="left", suffixes=("", "_feature"))
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["state_id"] = df["state_id"].astype(int)
    for col in ["evaluable_30d", "observed_disengaged_30d", "is_last_observed_session"]:
        df[col] = boolify(df[col])
    df = df.sort_values(["pid", "session_index"]).copy()
    df["prior_session_count"] = df["session_index"] - 1
    df["prior_s4_count"] = df.groupby("pid")["state_id"].transform(lambda s: s.eq(4).cumsum().shift(1, fill_value=0))
    df["prior_s4_share"] = (df["prior_s4_count"] / df["prior_session_count"].replace(0, np.nan)).fillna(0)
    df["log_anchor_session_index"] = np.log1p(df["session_index"])
    df["log_player_lifetime_days"] = np.log1p(df["player_lifetime_days"].clip(lower=0).fillna(0))
    df["log_prior_session_count"] = np.log1p(df["prior_session_count"].clip(lower=0))
    df["log_prior_s4_count"] = np.log1p(df["prior_s4_count"].clip(lower=0))

    rolling = pd.concat([rolling_metrics_for_player(g) for _, g in df.groupby("pid", sort=False)], ignore_index=True)
    slopes = add_anchor_slopes(rolling)
    anchor_frame = df.merge(slopes, left_on=["pid", "session_id", "session_index"], right_on=["pid", "anchor_session_id", "anchor_session_index"], how="inner")
    anchor_frame["anchor_month"] = pd.to_datetime(anchor_frame["login_time_utc"]).dt.to_period("M").astype(str)
    anchor_frame["anchor_login_time_utc"] = anchor_frame["login_time_utc"]
    anchor_frame["anchor_state_id"] = anchor_frame["state_id"]

    matched, matches = build_matched_sample(anchor_frame)
    matched.to_csv(OUT_DIR / "behavioral_narrowing_matched_anchor_slopes.csv", index=False, encoding="utf-8-sig")
    matches.to_csv(OUT_DIR / "behavioral_narrowing_matched_pairs.csv", index=False, encoding="utf-8-sig")

    pair_comp = paired_comparisons(matched)
    ols = adjusted_ols(matched)
    pair_comp.to_csv(OUT_DIR / "behavioral_narrowing_paired_comparison.csv", index=False, encoding="utf-8-sig")
    ols.to_csv(OUT_DIR / "behavioral_narrowing_adjusted_ols.csv", index=False, encoding="utf-8-sig")

    focus = ols[ols["term"].eq("terminal_case")].copy()
    match_quality = {
        "matched_pairs": int(matches["match_id"].nunique()),
        "same_month_rate": float(matches["same_month"].mean()),
        "same_index_bin_rate": float(matches["same_index_bin"].mean()),
        "fallback_rate": float(matches["used_fallback"].mean()),
        "median_abs_session_index_difference": float((matches["case_anchor_index"] - matches["control_anchor_index"]).abs().median()),
    }
    pd.DataFrame([match_quality]).to_csv(OUT_DIR / "behavioral_narrowing_match_quality.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Step 5.5 Behavioral Narrowing Test",
        "",
        "## Scope",
        "",
        "- Question: is state-space narrowing specific to terminal disengagement, or a generic lifecycle pattern?",
        "- Cases: terminal 30-day observed disengagement anchors.",
        "- Controls: active pseudo-anchors where the player returned within 30 days.",
        "- Matching: one active control per terminal case where possible, excluding same-player controls, matched primarily by anchor month and session-index bin.",
        f"- Slope horizon: previous {SLOPE_HORIZON_SESSIONS} sessions plus anchor, requiring at least {MIN_POINTS_FOR_SLOPE} points.",
        "- Narrowing slope is oriented so positive values mean narrowing: entropy/unique states decline toward anchor, dominant-state share rises toward anchor.",
        "",
        "## Match Quality",
        "",
        f"- Matched pairs: {match_quality['matched_pairs']:,}",
        f"- Same month: {pct(match_quality['same_month_rate'])}",
        f"- Same session-index bin: {pct(match_quality['same_index_bin_rate'])}",
        f"- Fallback matching: {pct(match_quality['fallback_rate'])}",
        f"- Median absolute session-index difference: {match_quality['median_abs_session_index_difference']:.0f}",
        "",
        "## Paired Comparison",
        "",
        "| window | metric | case mean | active-control mean | case - control | p-value |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in pair_comp.itertuples(index=False):
        lines.append(
            f"| {r.window} | {r.metric} | {r.case_mean:.4f} | {r.control_mean:.4f} | "
            f"{r.case_minus_control:.4f} | {r.p_value_normal_approx:.3g} |"
        )
    lines += [
        "",
        "## Adjusted Comparison",
        "",
        "Adjusted OLS controls for anchor session index, player lifetime days, prior session count, prior S4 count, prior S4 share, anchor month, and anchor state. Standard errors are clustered by matched pair.",
        "",
        "| window | metric | terminal-case coefficient | SE | p-value |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in focus.itertuples(index=False):
        lines.append(
            f"| {r.window} | {r.metric} | {r.coef:.4f} | {r.cluster_robust_se_by_match:.4f} | {r.p_value:.3g} |"
        )

    sig_positive = focus[(focus["metric"].isin(["state_entropy", "unique_states"])) & (focus["coef"].gt(0)) & (focus["p_value"].lt(0.05))]
    dom = focus[(focus["metric"].eq("dominant_state_share")) & (focus["coef"].gt(0)) & (focus["p_value"].lt(0.05))]
    lines += [
        "",
        "## Answer",
        "",
    ]
    if len(sig_positive) >= 2 and len(dom) >= 1:
        lines.append(
            "The matched/adjusted evidence supports disengagement-specific behavioral narrowing: terminal cases show steeper narrowing than active controls across diversity and concentration metrics."
        )
    else:
        lines.append(
            "The matched/adjusted evidence is mixed; narrowing cannot yet be claimed as clearly disengagement-specific across all metrics."
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `behavioral_narrowing_matched_anchor_slopes.csv`",
        "- `behavioral_narrowing_matched_pairs.csv`",
        "- `behavioral_narrowing_match_quality.csv`",
        "- `behavioral_narrowing_paired_comparison.csv`",
        "- `behavioral_narrowing_adjusted_ols.csv`",
    ]
    (OUT_DIR / "step5_5_behavioral_narrowing_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(focus[["window", "metric", "coef", "cluster_robust_se_by_match", "p_value"]].to_string(index=False))
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
