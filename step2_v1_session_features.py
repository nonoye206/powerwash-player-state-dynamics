import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data") / "data"
STEP1_DIR = Path("step1_audit")
OUT_DIR = Path("step2_v1")

EVENT_TABLES = {
    "job_started": ["pid", "Time_utc", "CurrentGameMode", "CurrentJobName"],
    "job_resumed": ["pid", "Time_utc", "CurrentGameMode", "CurrentJobName", "LevelProgressionAmount"],
    "job_exited": ["pid", "Time_utc", "CurrentGameMode", "CurrentJobName", "LevelProgressionAmount"],
    "job_completed": ["pid", "Time_utc", "CurrentGameMode", "CampaignProgressionAmount"],
    "task_completed": ["pid", "Time_utc", "CurrentGameMode", "LevelProgressionAmount"],
    "item_purchased": ["pid", "Time_utc", "CurrentJobName", "CampaignProgressionAmount"],
    "exited_game": [
        "pid",
        "Time_utc",
        "CurrentGameMode",
        "CurrentJobName",
        "CurrentSessionLength",
        "CampaignProgressionAmount",
        "LevelProgressionAmount",
    ],
}


def read_sessions():
    sessions = pd.read_csv(STEP1_DIR / "reconstructed_sessions.csv")
    sessions["login_time_utc"] = pd.to_datetime(sessions["login_time_utc"])
    sessions["exit_time_utc"] = pd.to_datetime(sessions["exit_time_utc"])
    sessions["session_id"] = sessions["pid"].astype(str) + "_s" + sessions["session_index"].astype(str).str.zfill(4)
    sessions = sessions.sort_values(["pid", "session_index"]).reset_index(drop=True)
    sessions["previous_exit_time_utc"] = sessions.groupby("pid")["exit_time_utc"].shift(1)
    sessions["days_since_previous_session"] = (
        (sessions["login_time_utc"] - sessions["previous_exit_time_utc"]).dt.total_seconds() / 86400.0
    )
    sessions["first_login_time_utc"] = sessions.groupby("pid")["login_time_utc"].transform("min")
    sessions["player_lifetime_days"] = (
        (sessions["login_time_utc"] - sessions["first_login_time_utc"]).dt.total_seconds() / 86400.0
    )
    sessions["is_zero_duration"] = sessions["duration_seconds"].eq(0)
    sessions["is_long_over_12h"] = sessions["duration_seconds"].gt(12 * 3600)
    return sessions


def session_lookup(sessions):
    lookup = {}
    for pid, group in sessions.groupby("pid", sort=False):
        lookup[pid] = {
            "starts": group["login_time_utc"].to_numpy(dtype="datetime64[ns]"),
            "ends": group["exit_time_utc"].to_numpy(dtype="datetime64[ns]"),
            "session_ids": group["session_id"].to_numpy(),
        }
    return lookup


def assign_events_to_sessions(events, lookup):
    session_ids = np.full(len(events), None, dtype=object)
    matched = np.zeros(len(events), dtype=bool)
    for pid, index in events.groupby("pid", sort=False).groups.items():
        spec = lookup.get(pid)
        if spec is None:
            continue
        idx = np.fromiter(index, dtype=np.int64)
        times = events.loc[idx, "event_time_utc"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(spec["starts"], times, side="right") - 1
        valid = pos >= 0
        valid_idx = idx[valid]
        valid_pos = pos[valid]
        in_bounds = times[valid] <= spec["ends"][valid_pos]
        final_idx = valid_idx[in_bounds]
        final_pos = valid_pos[in_bounds]
        session_ids[final_idx] = spec["session_ids"][final_pos]
        matched[final_idx] = True
    assigned = events.loc[matched].copy()
    assigned["session_id"] = session_ids[matched]
    return assigned, int(matched.sum()), int((~matched).sum())


def read_event_table(table):
    path = DATA_DIR / f"{table}.csv"
    usecols = EVENT_TABLES[table]
    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    df["event_time_utc"] = pd.to_datetime(df["Time_utc"], errors="coerce")
    df = df[df["pid"].ne("") & df["event_time_utc"].notna()].copy()
    df["event_table"] = table
    for col in ["LevelProgressionAmount", "CampaignProgressionAmount", "CurrentSessionLength"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def mode_or_blank(values):
    clean = values.dropna().astype(str)
    clean = clean[clean.ne("")]
    if clean.empty:
        return ""
    return clean.mode().iloc[0]


def aggregate_event_counts(assigned_by_table):
    rows = []
    for table, df in assigned_by_table.items():
        rows.append(df.groupby("session_id").size().rename(f"{table}_events"))
    return pd.concat(rows, axis=1).fillna(0).astype(int) if rows else pd.DataFrame()


def aggregate_features(sessions, assigned_by_table):
    features = sessions[
        [
            "session_id",
            "pid",
            "session_index",
            "login_time_utc",
            "exit_time_utc",
            "duration_seconds",
            "duration_minutes",
            "days_since_previous_session",
            "player_lifetime_days",
            "is_zero_duration",
            "is_long_over_12h",
        ]
    ].copy()

    counts = aggregate_event_counts(assigned_by_table)
    features = features.merge(counts, on="session_id", how="left")
    count_cols = [c for c in features.columns if c.endswith("_events")]
    features[count_cols] = features[count_cols].fillna(0).astype(int)

    rename = {
        "job_started_events": "jobs_started",
        "job_resumed_events": "jobs_resumed",
        "job_exited_events": "jobs_exited",
        "job_completed_events": "jobs_completed",
        "task_completed_events": "tasks_completed",
        "item_purchased_events": "items_purchased",
        "exited_game_events": "exit_events_matched",
    }
    features = features.rename(columns=rename)
    for col in rename.values():
        if col not in features:
            features[col] = 0

    started_or_resumed = features["jobs_started"] + features["jobs_resumed"]
    features["job_completion_ratio"] = np.where(started_or_resumed > 0, features["jobs_completed"] / started_or_resumed, np.nan)
    features["job_exit_ratio"] = np.where(started_or_resumed > 0, features["jobs_exited"] / started_or_resumed, np.nan)

    mode_frames = []
    for table in ["job_started", "job_resumed", "job_exited", "job_completed", "exited_game"]:
        df = assigned_by_table.get(table)
        if df is not None and "CurrentGameMode" in df.columns:
            mode_frames.append(df[["session_id", "CurrentGameMode"]])
    modes = pd.concat(mode_frames, ignore_index=True) if mode_frames else pd.DataFrame(columns=["session_id", "CurrentGameMode"])
    mode_features = modes.groupby("session_id").agg(
        primary_game_mode=("CurrentGameMode", mode_or_blank),
        mode_diversity=("CurrentGameMode", lambda s: s.replace("", np.nan).dropna().nunique()),
    )
    features = features.merge(mode_features, on="session_id", how="left")
    features["primary_game_mode"] = features["primary_game_mode"].fillna("")
    features["mode_diversity"] = features["mode_diversity"].fillna(0).astype(int)

    level_frames = []
    for table in ["job_resumed", "job_exited", "task_completed", "exited_game"]:
        df = assigned_by_table.get(table)
        if df is not None and "LevelProgressionAmount" in df.columns:
            level_frames.append(df[["session_id", "LevelProgressionAmount"]])
    levels = pd.concat(level_frames, ignore_index=True).dropna() if level_frames else pd.DataFrame()
    if not levels.empty:
        level_features = levels.groupby("session_id")["LevelProgressionAmount"].agg(["min", "max"]).rename(
            columns={"min": "level_progress_min", "max": "level_progress_max"}
        )
        level_features["progression_gain"] = level_features["level_progress_max"] - level_features["level_progress_min"]
        features = features.merge(level_features[["progression_gain"]], on="session_id", how="left")
    else:
        features["progression_gain"] = np.nan

    campaign_frames = []
    for table in ["job_completed", "item_purchased", "exited_game"]:
        df = assigned_by_table.get(table)
        if df is not None and "CampaignProgressionAmount" in df.columns:
            campaign_frames.append(df[["session_id", "CampaignProgressionAmount"]])
    campaigns = pd.concat(campaign_frames, ignore_index=True).dropna() if campaign_frames else pd.DataFrame()
    if not campaigns.empty:
        campaign_features = campaigns.groupby("session_id")["CampaignProgressionAmount"].agg(["min", "max"]).rename(
            columns={"min": "campaign_progress_min", "max": "campaign_progress_max"}
        )
        campaign_features["campaign_progression_gain"] = (
            campaign_features["campaign_progress_max"] - campaign_features["campaign_progress_min"]
        )
        features = features.merge(campaign_features[["campaign_progression_gain"]], on="session_id", how="left")
    else:
        features["campaign_progression_gain"] = np.nan

    ordered = [
        "session_id",
        "pid",
        "session_index",
        "login_time_utc",
        "exit_time_utc",
        "duration_minutes",
        "days_since_previous_session",
        "player_lifetime_days",
        "jobs_started",
        "jobs_resumed",
        "jobs_completed",
        "jobs_exited",
        "tasks_completed",
        "items_purchased",
        "job_completion_ratio",
        "job_exit_ratio",
        "progression_gain",
        "campaign_progression_gain",
        "primary_game_mode",
        "mode_diversity",
        "exit_events_matched",
        "is_zero_duration",
        "is_long_over_12h",
    ]
    return features[ordered].sort_values(["pid", "session_index"])


def sanity_checks(features, assignment_summary):
    checks = {
        "sessions": int(len(features)),
        "players": int(features["pid"].nunique()),
        "zero_duration_sessions": int(features["is_zero_duration"].sum()),
        "long_over_12h_sessions": int(features["is_long_over_12h"].sum()),
        "sessions_without_matched_exit_event": int(features["exit_events_matched"].eq(0).sum()),
        "sessions_with_multiple_matched_exit_events": int(features["exit_events_matched"].gt(1).sum()),
        "negative_progression_gain_sessions": int(features["progression_gain"].lt(-1e-9).sum()),
        "negative_campaign_progression_gain_sessions": int(features["campaign_progression_gain"].lt(-1e-9).sum()),
        "sessions_job_completed_gt_started_plus_resumed": int(
            features["jobs_completed"].gt(features["jobs_started"] + features["jobs_resumed"]).sum()
        ),
        "sessions_job_exited_gt_started_plus_resumed": int(
            features["jobs_exited"].gt(features["jobs_started"] + features["jobs_resumed"]).sum()
        ),
        "max_sessions_per_player": int(features.groupby("pid").size().max()),
        "p95_sessions_per_player": float(features.groupby("pid").size().quantile(0.95)),
        "duration_minutes_p50": float(features["duration_minutes"].quantile(0.50)),
        "duration_minutes_p95": float(features["duration_minutes"].quantile(0.95)),
        "duration_minutes_max": float(features["duration_minutes"].max()),
    }

    lines = ["# Step 2 V1 Session Feature Sanity Check", ""]
    lines.append("## Feature table")
    lines.append(f"- Rows: {checks['sessions']}")
    lines.append(f"- Players: {checks['players']}")
    lines.append(f"- Columns: {features.shape[1]}")
    lines.append("- Core feature columns: 15")
    lines.append("")
    lines.append("## Event assignment")
    lines.append("| table | valid events | assigned | unassigned | assigned pct |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in assignment_summary:
        pct = 100 * row["assigned_events"] / row["valid_events"] if row["valid_events"] else 0
        lines.append(
            f"| {row['table']} | {row['valid_events']} | {row['assigned_events']} | {row['unassigned_events']} | {pct:.2f}% |"
        )
    lines.append("")
    lines.append("## Main checks")
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `duration_minutes` comes from Step 1 reconstructed login-to-exit duration.")
    lines.append("- `progression_gain` is max minus min `LevelProgressionAmount` observed inside the session.")
    lines.append("- `campaign_progression_gain` is max minus min `CampaignProgressionAmount` observed inside the session.")
    lines.append("- V1 intentionally excludes `subtask_completed`, `update_current_state`, study prompts, and mood reports.")
    (OUT_DIR / "sanity_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "sanity_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return checks


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sessions = read_sessions()
    lookup = session_lookup(sessions)
    assigned_by_table = {}
    assignment_summary = []

    for table in EVENT_TABLES:
        events = read_event_table(table)
        assigned, matched_count, unmatched_count = assign_events_to_sessions(events, lookup)
        assigned_by_table[table] = assigned
        assignment_summary.append(
            {
                "table": table,
                "valid_events": int(len(events)),
                "assigned_events": matched_count,
                "unassigned_events": unmatched_count,
            }
        )

    features = aggregate_features(sessions, assigned_by_table)
    features.to_csv(OUT_DIR / "session_features_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(assignment_summary).to_csv(OUT_DIR / "event_assignment_summary.csv", index=False, encoding="utf-8-sig")
    checks = sanity_checks(features, assignment_summary)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
