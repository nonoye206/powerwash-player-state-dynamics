from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path("step2_v1/session_features_v1.csv")
LABELS_PATH = Path("step3_2_cluster_evaluation/labels_pc1_pc5_gmm_diag_k5.csv")
OUT_DIR = Path("step3_5_minimal_session_cluster_audit")

EVENT_COUNT_COLS = [
    "jobs_started",
    "jobs_resumed",
    "jobs_completed",
    "jobs_exited",
    "tasks_completed",
    "items_purchased",
]


def load_labeled_sessions():
    sessions = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    if not sessions["session_id"].equals(labels["session_id"]):
        raise ValueError("Labels are not aligned with session_features_v1.csv")
    sessions["cluster"] = labels["label"].astype(int)
    sessions["login_time_utc"] = pd.to_datetime(sessions["login_time_utc"])
    sessions["exit_time_utc"] = pd.to_datetime(sessions["exit_time_utc"])
    sessions["login_date_utc"] = sessions["login_time_utc"].dt.date.astype(str)
    sessions["any_job_task_item_event"] = sessions[EVENT_COUNT_COLS].sum(axis=1).gt(0)
    sessions["job_task_item_event_count"] = sessions[EVENT_COUNT_COLS].sum(axis=1)
    sessions["total_sessions_player"] = sessions.groupby("pid")["session_id"].transform("size")
    sessions["trajectory_position_pct"] = np.where(
        sessions["total_sessions_player"].gt(1),
        (sessions["session_index"] - 1) / (sessions["total_sessions_player"] - 1),
        0,
    )
    sessions["prev_cluster"] = sessions.groupby("pid")["cluster"].shift(1)
    sessions["next_cluster"] = sessions.groupby("pid")["cluster"].shift(1 * -1)
    sessions["prev_session_id"] = sessions.groupby("pid")["session_id"].shift(1)
    sessions["next_session_id"] = sessions.groupby("pid")["session_id"].shift(-1)
    sessions["has_previous_session"] = sessions["prev_session_id"].notna()
    sessions["has_next_session"] = sessions["next_session_id"].notna()
    sessions["days_to_next_session"] = sessions.groupby("pid")["login_time_utc"].shift(-1)
    sessions["days_to_next_session"] = (
        (sessions["days_to_next_session"] - sessions["exit_time_utc"]).dt.total_seconds() / 86400.0
    )
    return sessions


def identify_minimal_cluster(sessions):
    cluster_summary = (
        sessions.groupby("cluster")
        .agg(
            size=("session_id", "size"),
            duration_median=("duration_minutes", "median"),
            event_count_median=("job_task_item_event_count", "median"),
            no_event_rate=("any_job_task_item_event", lambda s: 1 - s.mean()),
            missing_mode_rate=("primary_game_mode", lambda s: s.fillna("").eq("").mean()),
            first_session_rate=("session_index", lambda s: np.nan),
        )
        .reset_index()
    )
    cluster_summary["minimal_score"] = (
        cluster_summary["duration_median"].rank(method="min", ascending=True)
        + cluster_summary["event_count_median"].rank(method="min", ascending=True)
        + cluster_summary["no_event_rate"].rank(method="min", ascending=False)
        + cluster_summary["missing_mode_rate"].rank(method="min", ascending=False)
    )
    minimal_cluster = int(cluster_summary.sort_values("minimal_score").iloc[0]["cluster"])
    return minimal_cluster, cluster_summary


def audit_cluster(sessions, cluster):
    target = sessions[sessions["cluster"].eq(cluster)].copy()
    n_all = len(sessions)
    n = len(target)

    per_player_counts = target.groupby("pid").size().rename("minimal_session_count").reset_index()
    per_player_dist = (
        per_player_counts["minimal_session_count"]
        .value_counts()
        .sort_index()
        .rename_axis("minimal_sessions_per_player")
        .reset_index(name="players")
    )
    per_player_dist["pct_of_cluster_players"] = per_player_dist["players"] / len(per_player_counts) * 100

    trajectory_bins = pd.cut(
        target["trajectory_position_pct"],
        bins=[-0.001, 0.0, 0.25, 0.5, 0.75, 1.0],
        labels=["first", "early", "middle", "late", "last"],
        include_lowest=True,
    )
    trajectory_dist = trajectory_bins.value_counts(dropna=False).sort_index().rename_axis("trajectory_bin").reset_index(name="sessions")
    trajectory_dist["pct"] = trajectory_dist["sessions"] / n * 100

    prev_dist = (
        target["prev_cluster"]
        .fillna("none")
        .astype(str)
        .value_counts()
        .rename_axis("prev_cluster")
        .reset_index(name="sessions")
    )
    prev_dist["pct"] = prev_dist["sessions"] / n * 100

    next_dist = (
        target["next_cluster"]
        .fillna("none")
        .astype(str)
        .value_counts()
        .rename_axis("next_cluster")
        .reset_index(name="sessions")
    )
    next_dist["pct"] = next_dist["sessions"] / n * 100

    date_counts = target["login_date_utc"].value_counts().rename_axis("login_date_utc").reset_index(name="sessions")
    date_counts["pct"] = date_counts["sessions"] / n * 100
    daily_all = sessions["login_date_utc"].value_counts().rename_axis("login_date_utc").reset_index(name="all_sessions")
    date_counts = date_counts.merge(daily_all, on="login_date_utc", how="left")
    date_counts["minimal_share_of_day_sessions_pct"] = date_counts["sessions"] / date_counts["all_sessions"] * 100

    date_month = target["login_time_utc"].dt.to_period("M").astype(str).value_counts().rename_axis("month").reset_index(name="sessions")
    date_month["pct"] = date_month["sessions"] / n * 100

    event_presence = []
    for col in EVENT_COUNT_COLS:
        event_presence.append(
            {
                "event_feature": col,
                "sessions_with_event": int(target[col].gt(0).sum()),
                "pct_sessions_with_event": float(target[col].gt(0).mean() * 100),
                "sum_events": int(target[col].sum()),
                "median_events": float(target[col].median()),
                "max_events": int(target[col].max()),
            }
        )
    event_presence = pd.DataFrame(event_presence)

    return_summary = {
        "cluster": cluster,
        "cluster_sessions": int(n),
        "cluster_pct_all_sessions": float(n / n_all * 100),
        "players_with_cluster": int(target["pid"].nunique()),
        "pct_all_players_with_cluster": float(target["pid"].nunique() / sessions["pid"].nunique() * 100),
        "first_session_rate_pct": float(target["session_index"].eq(1).mean() * 100),
        "single_session_player_rate_pct": float(target["total_sessions_player"].eq(1).mean() * 100),
        "median_minimal_sessions_per_player_among_owners": float(per_player_counts["minimal_session_count"].median()),
        "max_minimal_sessions_per_player": int(per_player_counts["minimal_session_count"].max()),
        "median_session_index": float(target["session_index"].median()),
        "median_total_sessions_player": float(target["total_sessions_player"].median()),
        "median_trajectory_position_pct": float(target["trajectory_position_pct"].median() * 100),
        "has_previous_session_rate_pct": float(target["has_previous_session"].mean() * 100),
        "has_next_session_return_rate_pct": float(target["has_next_session"].mean() * 100),
        "median_days_to_next_session": float(target["days_to_next_session"].median()),
        "p75_days_to_next_session": float(target["days_to_next_session"].quantile(0.75)),
        "any_job_task_item_event_rate_pct": float(target["any_job_task_item_event"].mean() * 100),
        "no_job_task_item_event_rate_pct": float((~target["any_job_task_item_event"]).mean() * 100),
        "median_event_count": float(target["job_task_item_event_count"].median()),
        "max_event_count": int(target["job_task_item_event_count"].max()),
        "duration_minutes_median": float(target["duration_minutes"].median()),
        "duration_minutes_p95": float(target["duration_minutes"].quantile(0.95)),
        "top_date": str(date_counts.iloc[0]["login_date_utc"]),
        "top_date_pct": float(date_counts.iloc[0]["pct"]),
        "top_7_dates_pct": float(date_counts.head(7)["sessions"].sum() / n * 100),
        "date_range_start": str(target["login_date_utc"].min()),
        "date_range_end": str(target["login_date_utc"].max()),
    }
    return return_summary, per_player_dist, trajectory_dist, prev_dist, next_dist, date_counts, date_month, event_presence, target


def write_report(summary, cluster_summary, per_player_dist, trajectory_dist, prev_dist, next_dist, date_counts, date_month, event_presence):
    lines = ["# Step 3.5 Minimal-session Cluster Audit", ""]
    lines.append("Model source: `pc1_pc5_gmm_diag_k5` labels from Step 3.2. No models were refit.")
    lines.append(f"Audited cluster: `{summary['cluster']}`")
    lines.append("")
    lines.append("## Minimal Cluster Identification")
    lines.append("| cluster | size | duration median | event count median | no-event rate | missing mode rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in cluster_summary.sort_values("cluster").itertuples(index=False):
        lines.append(
            f"| {row.cluster} | {row.size} | {row.duration_median:.2f} | {row.event_count_median:.2f} | "
            f"{row.no_event_rate * 100:.2f}% | {row.missing_mode_rate * 100:.2f}% |"
        )
    lines.append("")
    lines.append("## Main Audit")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Per-player Occurrence")
    lines.append("| minimal sessions per player | players | pct of owning players |")
    lines.append("|---:|---:|---:|")
    for row in per_player_dist.head(20).itertuples(index=False):
        lines.append(f"| {row.minimal_sessions_per_player} | {row.players} | {row.pct_of_cluster_players:.2f}% |")
    lines.append("")
    lines.append("## Trajectory Position")
    lines.append("| trajectory bin | sessions | pct |")
    lines.append("|---|---:|---:|")
    for row in trajectory_dist.itertuples(index=False):
        lines.append(f"| {row.trajectory_bin} | {row.sessions} | {row.pct:.2f}% |")
    lines.append("")
    lines.append("## Previous Cluster")
    lines.append("| previous cluster | sessions | pct |")
    lines.append("|---|---:|---:|")
    for row in prev_dist.head(10).itertuples(index=False):
        lines.append(f"| {row.prev_cluster} | {row.sessions} | {row.pct:.2f}% |")
    lines.append("")
    lines.append("## Next Cluster")
    lines.append("| next cluster | sessions | pct |")
    lines.append("|---|---:|---:|")
    for row in next_dist.head(10).itertuples(index=False):
        lines.append(f"| {row.next_cluster} | {row.sessions} | {row.pct:.2f}% |")
    lines.append("")
    lines.append("## Job/task/item Events Between Login and Exit")
    lines.append("| event feature | sessions with event | pct sessions with event | total events | median | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in event_presence.itertuples(index=False):
        lines.append(
            f"| {row.event_feature} | {row.sessions_with_event} | {row.pct_sessions_with_event:.2f}% | "
            f"{row.sum_events} | {row.median_events:.2f} | {row.max_events} |"
        )
    lines.append("")
    lines.append("## Date Concentration")
    lines.append(f"- Date range: {summary['date_range_start']} to {summary['date_range_end']}")
    lines.append(f"- Top date: {summary['top_date']} ({summary['top_date_pct']:.2f}% of audited cluster)")
    lines.append(f"- Top 7 dates: {summary['top_7_dates_pct']:.2f}% of audited cluster")
    lines.append("")
    lines.append("Top dates:")
    lines.append("| date | minimal sessions | pct of cluster | all sessions that day | minimal share of day |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in date_counts.head(15).itertuples(index=False):
        lines.append(
            f"| {row.login_date_utc} | {row.sessions} | {row.pct:.2f}% | "
            f"{row.all_sessions} | {row.minimal_share_of_day_sessions_pct:.2f}% |"
        )
    lines.append("")
    lines.append("Monthly distribution:")
    lines.append("| month | sessions | pct |")
    lines.append("|---|---:|---:|")
    for row in date_month.sort_values("month").itertuples(index=False):
        lines.append(f"| {row.month} | {row.sessions} | {row.pct:.2f}% |")
    lines.append("")
    lines.append("## Provisional Gate")
    if summary["no_job_task_item_event_rate_pct"] > 95 and summary["has_next_session_return_rate_pct"] > 50:
        lines.append(
            "This cluster behaves like a structurally valid minimal/no-op session type rather than a pure terminal churn artifact."
        )
    else:
        lines.append(
            "This cluster needs caution before freezing V1 representation; inspect detailed sessions if it is dominated by artifacts or non-returning players."
        )
    (OUT_DIR / "minimal_session_cluster_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sessions = load_labeled_sessions()
    minimal_cluster, cluster_summary = identify_minimal_cluster(sessions)
    (
        summary,
        per_player_dist,
        trajectory_dist,
        prev_dist,
        next_dist,
        date_counts,
        date_month,
        event_presence,
        target,
    ) = audit_cluster(sessions, minimal_cluster)

    cluster_summary.to_csv(OUT_DIR / "minimal_cluster_identification.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(OUT_DIR / "minimal_session_cluster_summary.csv", index=False, encoding="utf-8-sig")
    per_player_dist.to_csv(OUT_DIR / "minimal_session_per_player_distribution.csv", index=False, encoding="utf-8-sig")
    trajectory_dist.to_csv(OUT_DIR / "minimal_session_trajectory_position.csv", index=False, encoding="utf-8-sig")
    prev_dist.to_csv(OUT_DIR / "minimal_session_previous_cluster_distribution.csv", index=False, encoding="utf-8-sig")
    next_dist.to_csv(OUT_DIR / "minimal_session_next_cluster_distribution.csv", index=False, encoding="utf-8-sig")
    date_counts.to_csv(OUT_DIR / "minimal_session_date_distribution.csv", index=False, encoding="utf-8-sig")
    date_month.to_csv(OUT_DIR / "minimal_session_month_distribution.csv", index=False, encoding="utf-8-sig")
    event_presence.to_csv(OUT_DIR / "minimal_session_event_presence.csv", index=False, encoding="utf-8-sig")
    target.to_csv(OUT_DIR / "minimal_session_cluster_sessions.csv", index=False, encoding="utf-8-sig")
    write_report(summary, cluster_summary, per_player_dist, trajectory_dist, prev_dist, next_dist, date_counts, date_month, event_presence)
    print(f"cluster={minimal_cluster} sessions={summary['cluster_sessions']} players={summary['players_with_cluster']} out={OUT_DIR}")


if __name__ == "__main__":
    main()
