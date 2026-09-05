from pathlib import Path

import numpy as np
import pandas as pd


ASSIGNMENTS_PATH = Path("step4_1_state_sequences/session_state_assignments_v1.csv")
FEATURES_PATH = Path("step2_v1/session_features_v1.csv")
OUT_DIR = Path("step5_disengagement_risk")
THRESHOLDS = [7, 14, 30]
N_STATES = 5


def load_sessions():
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    features = pd.read_csv(
        FEATURES_PATH,
        usecols=[
            "session_id",
            "pid",
            "session_index",
            "jobs_started",
            "jobs_resumed",
            "tasks_completed",
            "items_purchased",
            "duration_minutes",
            "progression_gain",
        ],
    )
    df = assignments.merge(features, on=["session_id", "pid", "session_index"], how="left", suffixes=("", "_feature"))
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["exit_time_utc"] = pd.to_datetime(df["exit_time_utc"])
    df = df.sort_values(["pid", "session_index", "login_time_utc"]).reset_index(drop=True)
    df["next_login_time_utc"] = df.groupby("pid")["login_time_utc"].shift(-1)
    df["next_session_id"] = df.groupby("pid")["session_id"].shift(-1)
    df["next_state_id"] = df.groupby("pid")["state_id"].shift(-1)
    df["prev_state_id"] = df.groupby("pid")["state_id"].shift(1)
    df["next_session_gap_days"] = (df["next_login_time_utc"] - df["exit_time_utc"]).dt.total_seconds() / 86400.0
    df["is_last_observed_session"] = df["next_session_id"].isna()
    df["data_cutoff_utc"] = df["exit_time_utc"].max()
    df["days_until_data_cutoff"] = (df["data_cutoff_utc"] - df["exit_time_utc"]).dt.total_seconds() / 86400.0
    df["state_pair_prev_current"] = df["prev_state_id"].apply(lambda x: "start" if pd.isna(x) else f"S{int(x)}") + "->S" + df["state_id"].astype(int).astype(str)
    df["state_pair_current_next"] = "S" + df["state_id"].astype(int).astype(str) + "->" + df["next_state_id"].apply(
        lambda x: "end" if pd.isna(x) else f"S{int(x)}"
    )
    df["s4_streak_current"] = current_state_streak(df, state=4)
    return df


def current_state_streak(df, state):
    streaks = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("pid", sort=False).groups.items():
        count = 0
        for pos in idx:
            if int(df.at[pos, "state_id"]) == state:
                count += 1
            else:
                count = 0
            streaks[pos] = count
    return streaks


def add_disengagement_labels(df):
    for days in THRESHOLDS:
        has_return_within = df["next_session_gap_days"].le(days)
        enough_followup = df["days_until_data_cutoff"].ge(days)
        evaluable = has_return_within | enough_followup
        disengaged = evaluable & ~has_return_within
        censored = ~evaluable
        df[f"evaluable_{days}d"] = evaluable
        df[f"censored_{days}d"] = censored
        df[f"observed_disengaged_{days}d"] = disengaged
    return df


def gap_distribution(df):
    gaps = df["next_session_gap_days"].dropna()
    rows = [
        {
            "sessions_with_next_session": int(gaps.size),
            "sessions_without_next_session": int(df["next_session_gap_days"].isna().sum()),
            "min": float(gaps.min()) if gaps.size else np.nan,
            "p25": float(gaps.quantile(0.25)) if gaps.size else np.nan,
            "p50": float(gaps.quantile(0.50)) if gaps.size else np.nan,
            "p75": float(gaps.quantile(0.75)) if gaps.size else np.nan,
            "p90": float(gaps.quantile(0.90)) if gaps.size else np.nan,
            "p95": float(gaps.quantile(0.95)) if gaps.size else np.nan,
            "p99": float(gaps.quantile(0.99)) if gaps.size else np.nan,
            "max": float(gaps.max()) if gaps.size else np.nan,
        }
    ]
    bins = [-np.inf, 0, 1 / 24, 1, 3, 7, 14, 30, 60, 120, np.inf]
    labels = ["<=0d", "0-1h", "1h-1d", "1-3d", "3-7d", "7-14d", "14-30d", "30-60d", "60-120d", ">120d"]
    hist = pd.cut(gaps, bins=bins, labels=labels).value_counts().sort_index().rename_axis("gap_bin").reset_index(name="sessions")
    hist["pct_of_sessions_with_next"] = hist["sessions"] / gaps.size * 100 if gaps.size else 0
    return pd.DataFrame(rows), hist


def threshold_summary(df):
    rows = []
    for days in THRESHOLDS:
        evaluable = df[f"evaluable_{days}d"]
        rows.append(
            {
                "threshold_days": days,
                "sessions": int(len(df)),
                "evaluable_sessions": int(evaluable.sum()),
                "censored_sessions": int(df[f"censored_{days}d"].sum()),
                "censored_pct": float(df[f"censored_{days}d"].mean() * 100),
                "observed_disengaged_sessions": int(df[f"observed_disengaged_{days}d"].sum()),
                "observed_disengagement_rate_evaluable_pct": float(df.loc[evaluable, f"observed_disengaged_{days}d"].mean() * 100),
                "returned_within_threshold_sessions": int((evaluable & ~df[f"observed_disengaged_{days}d"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def risk_by_group(df, group_cols, min_n=30):
    rows = []
    for days in THRESHOLDS:
        evaluable_col = f"evaluable_{days}d"
        label_col = f"observed_disengaged_{days}d"
        base_rate = df.loc[df[evaluable_col], label_col].mean()
        for keys, group in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            evaluable = group[evaluable_col]
            n_eval = int(evaluable.sum())
            if n_eval < min_n:
                continue
            rate = float(group.loc[evaluable, label_col].mean() * 100)
            row = {
                "threshold_days": days,
                "evaluable_sessions": n_eval,
                "observed_disengaged_sessions": int(group.loc[evaluable, label_col].sum()),
                "risk_pct": rate,
                "risk_lift_vs_overall": float((rate / (base_rate * 100)) if base_rate > 0 else np.nan),
            }
            for col, key in zip(group_cols, keys):
                row[col] = key
            rows.append(row)
    return pd.DataFrame(rows)


def transition_specific_risk(df):
    # Risk after the destination session of an observed transition.
    edges = df[df["next_state_id"].notna()][["pid", "session_id", "next_session_id", "state_id", "next_state_id"]].copy()
    dest = df[
        [
            "session_id",
            "state_id",
            "s4_streak_current",
            *[f"evaluable_{d}d" for d in THRESHOLDS],
            *[f"observed_disengaged_{d}d" for d in THRESHOLDS],
        ]
    ].copy()
    edges = edges.merge(dest, left_on="next_session_id", right_on="session_id", how="left", suffixes=("_from", "_dest"))
    edges["transition_pair"] = "S" + edges["state_id_from"].astype(int).astype(str) + "->S" + edges["next_state_id"].astype(int).astype(str)
    return risk_by_group(edges, ["transition_pair"], min_n=30)


def s4_streak_risk(df):
    work = df.copy()
    work["s4_streak_bucket"] = pd.cut(
        work["s4_streak_current"],
        bins=[-1, 0, 1, 2, 3, np.inf],
        labels=["non-S4", "S4x1", "S4x2", "S4x3", "S4x4plus"],
    )
    return risk_by_group(work, ["s4_streak_bucket"], min_n=30)


def early_warning_comparison(df):
    rows = []
    for days in THRESHOLDS:
        evaluable = df[f"evaluable_{days}d"]
        label = f"observed_disengaged_{days}d"
        base = df.loc[evaluable, label].mean() * 100
        s4 = df.loc[evaluable & df["state_id"].eq(4), label].mean() * 100
        s4_after_s4 = df.loc[evaluable & df["state_id"].eq(4) & df["prev_state_id"].eq(4), label].mean() * 100
        s4_streak2plus = df.loc[evaluable & df["s4_streak_current"].ge(2), label].mean() * 100
        s2_to_s4 = df.loc[evaluable & df["prev_state_id"].eq(2) & df["state_id"].eq(4), label].mean() * 100
        s1_to_s4 = df.loc[evaluable & df["prev_state_id"].eq(1) & df["state_id"].eq(4), label].mean() * 100
        rows.append(
            {
                "threshold_days": days,
                "overall_risk_pct": float(base),
                "current_s4_risk_pct": float(s4),
                "prev_s4_current_s4_risk_pct": float(s4_after_s4),
                "s4_streak_2plus_risk_pct": float(s4_streak2plus),
                "s1_to_s4_risk_pct": float(s1_to_s4),
                "s2_to_s4_risk_pct": float(s2_to_s4),
                "current_s4_lift": float(s4 / base) if base else np.nan,
                "s4_streak_2plus_lift": float(s4_streak2plus / base) if base else np.nan,
                "s2_to_s4_minus_s1_to_s4_pctpt": float(s2_to_s4 - s1_to_s4),
            }
        )
    return pd.DataFrame(rows)


def write_report(gap_summary, gap_hist, thresh, state_risk, trans_risk, streak_risk, early):
    lines = ["# Step 5 Disengagement Definition and Risk Dynamics", ""]
    lines.append("This step defines observed disengagement with right-censoring. It does not fit a predictive model.")
    lines.append("")
    lines.append("## Next-session Gap Distribution")
    row = gap_summary.iloc[0]
    for col in gap_summary.columns:
        lines.append(f"- {col}: {row[col]}")
    lines.append("")
    lines.append("Gap histogram:")
    lines.append("| gap bin | sessions | pct of sessions with next |")
    lines.append("|---|---:|---:|")
    for r in gap_hist.itertuples(index=False):
        lines.append(f"| {r.gap_bin} | {r.sessions} | {r.pct_of_sessions_with_next:.2f}% |")
    lines.append("")
    lines.append("## Candidate Disengagement Thresholds")
    lines.append("| threshold | evaluable | censored | censored % | observed disengaged | risk among evaluable |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in thresh.itertuples(index=False):
        lines.append(
            f"| {r.threshold_days}d | {r.evaluable_sessions} | {r.censored_sessions} | {r.censored_pct:.2f}% | "
            f"{r.observed_disengaged_sessions} | {r.observed_disengagement_rate_evaluable_pct:.2f}% |"
        )
    lines.append("")
    lines.append("## Risk By Current State")
    lines.append("| threshold | state | evaluable | disengaged | risk | lift |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in state_risk.sort_values(["threshold_days", "state_id"]).itertuples(index=False):
        lines.append(
            f"| {r.threshold_days}d | S{int(r.state_id)} | {r.evaluable_sessions} | {r.observed_disengaged_sessions} | "
            f"{r.risk_pct:.2f}% | {r.risk_lift_vs_overall:.2f} |"
        )
    lines.append("")
    lines.append("## Focused Risk Comparisons")
    lines.append("| threshold | overall | S4 | S4 after S4 | S4 streak >=2 | S1->S4 | S2->S4 | S2->S4 minus S1->S4 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in early.itertuples(index=False):
        lines.append(
            f"| {r.threshold_days}d | {r.overall_risk_pct:.2f}% | {r.current_s4_risk_pct:.2f}% | "
            f"{r.prev_s4_current_s4_risk_pct:.2f}% | {r.s4_streak_2plus_risk_pct:.2f}% | "
            f"{r.s1_to_s4_risk_pct:.2f}% | {r.s2_to_s4_risk_pct:.2f}% | {r.s2_to_s4_minus_s1_to_s4_pctpt:.2f}pp |"
        )
    lines.append("")
    lines.append("## Selected Transition-pair Risks")
    focus_pairs = ["S4->S4", "S4->S1", "S4->S2", "S1->S4", "S2->S4"]
    focus = trans_risk[trans_risk["transition_pair"].isin(focus_pairs)].sort_values(["threshold_days", "transition_pair"])
    lines.append("| threshold | transition pair | evaluable destination sessions | risk | lift |")
    lines.append("|---:|---|---:|---:|---:|")
    for r in focus.itertuples(index=False):
        lines.append(
            f"| {r.threshold_days}d | {r.transition_pair} | {r.evaluable_sessions} | {r.risk_pct:.2f}% | {r.risk_lift_vs_overall:.2f} |"
        )
    lines.append("")
    lines.append("## S4 Streak Risk")
    lines.append("| threshold | streak bucket | evaluable | risk | lift |")
    lines.append("|---:|---|---:|---:|---:|")
    for r in streak_risk.sort_values(["threshold_days", "s4_streak_bucket"]).itertuples(index=False):
        lines.append(
            f"| {r.threshold_days}d | {r.s4_streak_bucket} | {r.evaluable_sessions} | {r.risk_pct:.2f}% | {r.risk_lift_vs_overall:.2f} |"
        )
    lines.append("")
    lines.append("## Definition Recommendation")
    lines.append(
        "Use 30-day observed disengagement as the main definition if the goal is durable non-return; keep 7-day and 14-day labels as sensitivity checks."
    )
    lines.append(
        "A session is evaluable for Delta-day disengagement if the player returned within Delta days or the session has at least Delta days of observation before the dataset cutoff. Otherwise it is right-censored."
    )
    (OUT_DIR / "step5_disengagement_definition_risk_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = add_disengagement_labels(load_sessions())
    gap_summary, gap_hist = gap_distribution(df)
    thresh = threshold_summary(df)
    state_risk = risk_by_group(df, ["state_id"], min_n=30)
    trans_risk = transition_specific_risk(df)
    streak = s4_streak_risk(df)
    early = early_warning_comparison(df)

    df.to_csv(OUT_DIR / "session_disengagement_labels.csv", index=False, encoding="utf-8-sig")
    gap_summary.to_csv(OUT_DIR / "next_session_gap_summary.csv", index=False, encoding="utf-8-sig")
    gap_hist.to_csv(OUT_DIR / "next_session_gap_histogram.csv", index=False, encoding="utf-8-sig")
    thresh.to_csv(OUT_DIR / "disengagement_threshold_summary.csv", index=False, encoding="utf-8-sig")
    state_risk.to_csv(OUT_DIR / "risk_by_current_state.csv", index=False, encoding="utf-8-sig")
    trans_risk.to_csv(OUT_DIR / "risk_by_transition_pair_destination.csv", index=False, encoding="utf-8-sig")
    streak.to_csv(OUT_DIR / "risk_by_s4_streak.csv", index=False, encoding="utf-8-sig")
    early.to_csv(OUT_DIR / "focused_risk_comparisons.csv", index=False, encoding="utf-8-sig")
    write_report(gap_summary, gap_hist, thresh, state_risk, trans_risk, streak, early)
    print(
        f"sessions={len(df)} cutoff={df['data_cutoff_utc'].iloc[0]} "
        f"labels={THRESHOLDS} out={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
