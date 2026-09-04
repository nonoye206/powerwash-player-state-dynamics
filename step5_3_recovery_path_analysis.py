from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "step5_disengagement_risk" / "session_disengagement_labels.csv"
OUT = ROOT / "step5_3_recovery_path_analysis"

DELTAS = (7, 14, 30)
RECOVERY_STATES = {1, 2}


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def state_label(value) -> str:
    if pd.isna(value):
        return "end"
    return f"S{int(value)}"


def broad_s4_path(next_state) -> str:
    if pd.isna(next_state):
        return "S4->end"
    next_state = int(next_state)
    if next_state in RECOVERY_STATES:
        return "S4->S1/S2"
    return f"S4->S{next_state}"


def exact_s4_path(next_state) -> str:
    if pd.isna(next_state):
        return "S4->end"
    return f"S4->S{int(next_state)}"


def summarize_numeric(group: pd.Series) -> dict:
    s = pd.to_numeric(group, errors="coerce").dropna()
    if s.empty:
        return {
            "n_non_missing": 0,
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "n_non_missing": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def outcome_summary(df: pd.DataFrame, group_col: str, thresholds=DELTAS) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        base = {
            group_col: key,
            "sessions": int(len(g)),
            "players": int(g["pid"].nunique()),
            "share_of_rows": float(len(g) / len(df)) if len(df) else np.nan,
            "median_duration_minutes": float(g["duration_minutes"].median()) if len(g) else np.nan,
            "update_window_share": float(g["is_update_window_2023_01_31_to_2023_02_07"].mean()) if len(g) else np.nan,
            "purchase_anomaly_share": float(g["is_minimal_cluster_purchase_anomaly"].mean()) if len(g) else np.nan,
        }
        if "next_session_gap_days" in g.columns:
            base["median_next_gap_days"] = float(pd.to_numeric(g["next_session_gap_days"], errors="coerce").median())
        for delta in thresholds:
            evaluable = g[f"evaluable_{delta}d"].astype(bool)
            disengaged = g.loc[evaluable, f"observed_disengaged_{delta}d"].astype(bool)
            base[f"evaluable_{delta}d"] = int(evaluable.sum())
            base[f"censored_{delta}d"] = int(g[f"censored_{delta}d"].astype(bool).sum())
            base[f"disengaged_{delta}d"] = int(disengaged.sum())
            base[f"risk_{delta}d"] = float(disengaged.mean()) if len(disengaged) else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def add_forward_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pid", "session_index"]).copy()
    for offset in (1, 2, 3):
        df[f"state_t_plus_{offset}"] = df.groupby("pid")["state_id"].shift(-offset)
        df[f"session_t_plus_{offset}"] = df.groupby("pid")["session_id"].shift(-offset)
        df[f"login_t_plus_{offset}"] = df.groupby("pid")["login_time_utc"].shift(-offset)
    df["path_t_to_t1_exact"] = df["next_state_id"].map(exact_s4_path)
    df["path_t_to_t1_broad"] = df["next_state_id"].map(broad_s4_path)
    return df


def classify_recovery_followup(row: pd.Series) -> str:
    nxt = row["state_t_plus_2"]
    if pd.isna(nxt):
        return "end_after_recovery_session"
    nxt = int(nxt)
    if nxt in RECOVERY_STATES:
        return "stays_in_S1/S2"
    if nxt == 4:
        return "returns_to_S4"
    return f"moves_to_S{nxt}"


def compute_time_to_next_s4_after_recovery(df: pd.DataFrame, s4_rows: pd.DataFrame) -> pd.DataFrame:
    recovery_rows = s4_rows[s4_rows["next_state_id"].isin(list(RECOVERY_STATES))].copy()
    records = []
    grouped = {pid: g.reset_index(drop=True) for pid, g in df.groupby("pid", sort=False)}

    for _, row in recovery_rows.iterrows():
        player = grouped[row["pid"]]
        current_pos = int(row["session_index"]) - 1
        dest_pos = current_pos + 1
        dest = player.iloc[dest_pos] if dest_pos < len(player) else None
        subsequent = player.iloc[dest_pos + 1 :].copy()
        next_s4 = subsequent[subsequent["state_id"].eq(4)].head(1)

        rec = {
            "origin_session_id": row["session_id"],
            "pid": row["pid"],
            "origin_session_index": int(row["session_index"]),
            "destination_session_id": row["next_session_id"],
            "destination_state_id": int(row["next_state_id"]),
            "destination_path_exact": exact_s4_path(row["next_state_id"]),
            "observed_sessions_after_destination": int(max(len(player) - dest_pos - 1, 0)),
            "ever_returns_to_s4_after_recovery": bool(not next_s4.empty),
            "sessions_until_next_s4_after_destination": np.nan,
            "days_until_next_s4_after_destination": np.nan,
        }
        if dest is not None and not next_s4.empty:
            found = next_s4.iloc[0]
            rec["sessions_until_next_s4_after_destination"] = int(found["session_index"] - dest["session_index"])
            rec["days_until_next_s4_after_destination"] = (
                pd.to_datetime(found["login_time_utc"]) - pd.to_datetime(dest["login_time_utc"])
            ).total_seconds() / 86400.0
        records.append(rec)

    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])

    bool_cols = [
        "is_update_window_2023_01_31_to_2023_02_07",
        "is_minimal_cluster_purchase_anomaly",
        "is_last_observed_session",
    ]
    for delta in DELTAS:
        bool_cols += [f"evaluable_{delta}d", f"censored_{delta}d", f"observed_disengaged_{delta}d"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(df[col]).astype(bool)

    df = add_forward_states(df)
    s4 = df[df["state_id"].eq(4)].copy()

    s4["path_exact"] = s4["next_state_id"].map(exact_s4_path)
    s4["path_broad"] = s4["next_state_id"].map(broad_s4_path)
    s4["trajectory_position_pct"] = s4["session_index"] / s4.groupby("pid")["session_index"].transform("max")

    path_distribution = (
        s4.groupby(["path_broad", "path_exact"], dropna=False)
        .agg(
            sessions=("session_id", "size"),
            players=("pid", "nunique"),
            median_duration_minutes=("duration_minutes", "median"),
            median_next_gap_days=("next_session_gap_days", "median"),
            median_trajectory_position_pct=("trajectory_position_pct", "median"),
            update_window_sessions=("is_update_window_2023_01_31_to_2023_02_07", "sum"),
            purchase_anomaly_sessions=("is_minimal_cluster_purchase_anomaly", "sum"),
        )
        .reset_index()
    )
    path_distribution["share_of_s4_sessions"] = path_distribution["sessions"] / len(s4)
    path_distribution["update_window_share"] = path_distribution["update_window_sessions"] / path_distribution["sessions"]
    path_distribution["purchase_anomaly_share"] = path_distribution["purchase_anomaly_sessions"] / path_distribution["sessions"]
    path_distribution = path_distribution.sort_values(["sessions", "path_exact"], ascending=[False, True])
    path_distribution.to_csv(OUT / "s4_next_path_distribution.csv", index=False)

    gap_rows = []
    for key, g in s4.groupby("path_broad"):
        rec = {"path_broad": key, "sessions": int(len(g))}
        rec.update({f"next_gap_days_{k}": v for k, v in summarize_numeric(g["next_session_gap_days"]).items()})
        gap_rows.append(rec)
    pd.DataFrame(gap_rows).to_csv(OUT / "s4_next_gap_summary.csv", index=False)

    current_risk_broad = outcome_summary(s4, "path_broad")
    current_risk_exact = outcome_summary(s4, "path_exact")
    current_risk_broad.to_csv(OUT / "s4_current_session_risk_by_broad_path.csv", index=False)
    current_risk_exact.to_csv(OUT / "s4_current_session_risk_by_exact_path.csv", index=False)

    destination = df[df["prev_state_id"].eq(4)].copy()
    destination["destination_path_exact"] = destination["state_id"].map(lambda x: f"S4->S{int(x)}")
    destination["destination_path_broad"] = destination["state_id"].map(lambda x: "S4->S1/S2" if int(x) in RECOVERY_STATES else f"S4->S{int(x)}")
    destination_risk_broad = outcome_summary(destination, "destination_path_broad")
    destination_risk_exact = outcome_summary(destination, "destination_path_exact")
    destination_risk_broad.to_csv(OUT / "s4_destination_session_risk_by_broad_path.csv", index=False)
    destination_risk_exact.to_csv(OUT / "s4_destination_session_risk_by_exact_path.csv", index=False)

    recovery_origin = s4[s4["next_state_id"].isin(list(RECOVERY_STATES))].copy()
    recovery_origin["after_recovery_next_step"] = recovery_origin.apply(classify_recovery_followup, axis=1)
    sustained = (
        recovery_origin.groupby(["path_exact", "after_recovery_next_step"])
        .agg(
            origin_s4_sessions=("session_id", "size"),
            players=("pid", "nunique"),
            median_gap_to_recovery_days=("next_session_gap_days", "median"),
        )
        .reset_index()
    )
    sustained["share_within_exact_recovery_path"] = sustained["origin_s4_sessions"] / sustained.groupby("path_exact")["origin_s4_sessions"].transform("sum")
    sustained["share_of_all_s4_recovery_origins"] = sustained["origin_s4_sessions"] / len(recovery_origin)
    sustained.to_csv(OUT / "s4_recovery_next_step_distribution.csv", index=False)

    time_to_s4 = compute_time_to_next_s4_after_recovery(df, s4)
    time_to_s4.to_csv(OUT / "s4_recovery_time_to_next_s4_detail.csv", index=False)
    time_summary = []
    for key, g in time_to_s4.groupby("destination_path_exact"):
        rec = {
            "destination_path_exact": key,
            "recovery_origins": int(len(g)),
            "players": int(g["pid"].nunique()),
            "ever_returns_to_s4_after_recovery": int(g["ever_returns_to_s4_after_recovery"].sum()),
            "ever_returns_to_s4_rate": float(g["ever_returns_to_s4_after_recovery"].mean()) if len(g) else np.nan,
        }
        rec.update({f"sessions_until_next_s4_{k}": v for k, v in summarize_numeric(g["sessions_until_next_s4_after_destination"]).items()})
        rec.update({f"days_until_next_s4_{k}": v for k, v in summarize_numeric(g["days_until_next_s4_after_destination"]).items()})
        time_summary.append(rec)
    pd.DataFrame(time_summary).to_csv(OUT / "s4_recovery_time_to_next_s4_summary.csv", index=False)

    player_summary = (
        s4.groupby("pid")
        .agg(
            s4_sessions=("session_id", "size"),
            s4_recovery_to_s1_s2=("next_state_id", lambda x: int(pd.Series(x).isin(list(RECOVERY_STATES)).sum())),
            s4_persistence=("next_state_id", lambda x: int(pd.Series(x).eq(4).sum())),
            s4_terminal=("next_state_id", lambda x: int(pd.Series(x).isna().sum())),
            first_s4_index=("session_index", "min"),
            last_s4_index=("session_index", "max"),
        )
        .reset_index()
    )
    player_total = df.groupby("pid")["session_index"].max().rename("total_sessions").reset_index()
    player_summary = player_summary.merge(player_total, on="pid", how="left")
    player_summary["has_recovery_to_s1_s2"] = player_summary["s4_recovery_to_s1_s2"] > 0
    player_summary["has_s4_persistence"] = player_summary["s4_persistence"] > 0
    player_summary["last_s4_position_pct"] = player_summary["last_s4_index"] / player_summary["total_sessions"]
    player_summary.to_csv(OUT / "s4_recovery_player_summary.csv", index=False)

    # Compact topline table for reading and later paper notes.
    def get_row(table: pd.DataFrame, key_col: str, key: str) -> pd.Series:
        match = table[table[key_col].eq(key)]
        return match.iloc[0] if len(match) else pd.Series(dtype=object)

    broad_dist = path_distribution.groupby("path_broad", as_index=False).agg(
        sessions=("sessions", "sum"),
        players=("players", "sum"),
        share_of_s4_sessions=("share_of_s4_sessions", "sum"),
    )
    report_lines = [
        "# Step 5.3 Recovery Path Analysis",
        "",
        "## Scope",
        "",
        f"- Input: `{INPUT.relative_to(ROOT)}`",
        f"- S4 origin sessions analyzed: {len(s4):,}",
        f"- Players with at least one S4 session: {s4['pid'].nunique():,}",
        "- Recovery is defined descriptively as the next observed session after S4 being S1 or S2.",
        "- Outcomes remain observed disengagement labels with right-censoring from Step 5.",
        "",
        "## Immediate path after S4",
        "",
    ]
    for _, row in broad_dist.sort_values("sessions", ascending=False).iterrows():
        report_lines.append(
            f"- {row['path_broad']}: {int(row['sessions']):,} sessions ({pct(row['share_of_s4_sessions'])} of S4 origins)"
        )

    report_lines += [
        "",
        "## Risk from the S4 origin session",
        "",
    ]
    for key in ["S4->S1/S2", "S4->S4", "S4->end", "S4->S3", "S4->S0"]:
        row = get_row(current_risk_broad, "path_broad", key)
        if not row.empty:
            report_lines.append(
                f"- {key}: 30d risk {pct(row['risk_30d'])}, 14d {pct(row['risk_14d'])}, 7d {pct(row['risk_7d'])}; "
                f"evaluable 30d n={int(row['evaluable_30d']):,}."
            )

    report_lines += [
        "",
        "## Risk after the destination session",
        "",
    ]
    for key in ["S4->S1/S2", "S4->S4", "S4->S3", "S4->S0"]:
        row = get_row(destination_risk_broad, "destination_path_broad", key)
        if not row.empty:
            report_lines.append(
                f"- {key}: 30d risk after destination {pct(row['risk_30d'])}, "
                f"14d {pct(row['risk_14d'])}, 7d {pct(row['risk_7d'])}; destination sessions n={int(row['sessions']):,}."
            )

    recovery_follow = (
        recovery_origin.groupby("after_recovery_next_step")
        .agg(sessions=("session_id", "size"), players=("pid", "nunique"))
        .reset_index()
    )
    recovery_follow["share"] = recovery_follow["sessions"] / len(recovery_origin) if len(recovery_origin) else np.nan

    report_lines += [
        "",
        "## Does recovery persist?",
        "",
        f"- S4 origins with immediate S1/S2 recovery: {len(recovery_origin):,}",
    ]
    for _, row in recovery_follow.sort_values("sessions", ascending=False).iterrows():
        report_lines.append(
            f"- Next step after the recovered S1/S2 session: {row['after_recovery_next_step']} = "
            f"{int(row['sessions']):,} ({pct(row['share'])})."
        )

    time_all = time_to_s4
    report_lines += [
        "",
        "## Later return to S4 after recovery",
        "",
        f"- Among immediate S1/S2 recoveries, later return to S4 was observed in "
        f"{int(time_all['ever_returns_to_s4_after_recovery'].sum()):,}/{len(time_all):,} cases "
        f"({pct(time_all['ever_returns_to_s4_after_recovery'].mean())}).",
    ]
    days = pd.to_numeric(time_all.loc[time_all["ever_returns_to_s4_after_recovery"], "days_until_next_s4_after_destination"], errors="coerce")
    sessions_until = pd.to_numeric(time_all.loc[time_all["ever_returns_to_s4_after_recovery"], "sessions_until_next_s4_after_destination"], errors="coerce")
    if len(days.dropna()):
        report_lines.append(
            f"- Conditional on returning to S4, median time back was {days.median():.2f} days "
            f"and median {sessions_until.median():.0f} observed sessions after the recovered session."
        )

    player_has_recovery_rate = player_summary["has_recovery_to_s1_s2"].mean()
    report_lines += [
        "",
        "## Player-level footprint",
        "",
        f"- Players with S4 who ever recovered immediately to S1/S2: "
        f"{int(player_summary['has_recovery_to_s1_s2'].sum()):,}/{len(player_summary):,} ({pct(player_has_recovery_rate)}).",
        f"- Median S4 sessions per S4-exposed player: {player_summary['s4_sessions'].median():.0f}; "
        f"P90: {player_summary['s4_sessions'].quantile(0.90):.0f}.",
        "",
        "## Interpretation",
        "",
        "- S4 is not a one-way terminal state. A large share of S4 origins are followed by S1/S2, and those destination sessions have much lower subsequent disengagement risk than S4 persistence.",
        "- Recovery is often real but not always durable: many recovered paths stay in S1/S2 for the next step, while a smaller but visible subset returns to S4 later.",
        "- This supports treating S4 persistence and recovery paths separately in Step 5, instead of interpreting all S4 exposure as equivalent.",
        "",
        "## Outputs",
        "",
        "- `s4_next_path_distribution.csv`",
        "- `s4_next_gap_summary.csv`",
        "- `s4_current_session_risk_by_broad_path.csv`",
        "- `s4_current_session_risk_by_exact_path.csv`",
        "- `s4_destination_session_risk_by_broad_path.csv`",
        "- `s4_destination_session_risk_by_exact_path.csv`",
        "- `s4_recovery_next_step_distribution.csv`",
        "- `s4_recovery_time_to_next_s4_summary.csv`",
        "- `s4_recovery_time_to_next_s4_detail.csv`",
        "- `s4_recovery_player_summary.csv`",
    ]
    (OUT / "step5_3_recovery_path_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
