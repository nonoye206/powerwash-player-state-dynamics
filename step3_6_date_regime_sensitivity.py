from pathlib import Path

import numpy as np
import pandas as pd

import step3_2_cluster_evaluation as cluster_eval


FEATURES_PATH = Path("step2_v1/session_features_v1.csv")
PCA_SCORES_PATH = Path("step3_pca/pca_session_scores.csv")
ORIGINAL_LABELS_PATH = Path("step3_2_cluster_evaluation/labels_pc1_pc5_gmm_diag_k5.csv")
OUT_DIR = Path("step3_6_date_regime_sensitivity")

PC_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]
EVENT_COUNT_COLS = [
    "jobs_started",
    "jobs_resumed",
    "jobs_completed",
    "jobs_exited",
    "tasks_completed",
    "items_purchased",
]

WINDOW_START = pd.Timestamp("2023-01-31")
WINDOW_END = pd.Timestamp("2023-02-07")
PRE_END = pd.Timestamp("2023-01-30")


def load_base():
    features = pd.read_csv(FEATURES_PATH)
    scores = pd.read_csv(PCA_SCORES_PATH, usecols=["session_id", *PC_FEATURES])
    labels = pd.read_csv(ORIGINAL_LABELS_PATH)
    if not features["session_id"].equals(scores["session_id"]):
        raise ValueError("PCA scores are not aligned with features.")
    if not features["session_id"].equals(labels["session_id"]):
        raise ValueError("Original labels are not aligned with features.")
    df = features.merge(scores, on="session_id", how="left")
    df["original_gmm_k5_cluster"] = labels["label"].astype(int)
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["login_date_utc"] = df["login_time_utc"].dt.normalize()
    df["any_job_task_item_event"] = df[EVENT_COUNT_COLS].sum(axis=1).gt(0)
    df["event_count"] = df[EVENT_COUNT_COLS].sum(axis=1)
    df["total_sessions_player"] = df.groupby("pid")["session_id"].transform("size")
    df["trajectory_position_pct"] = np.where(
        df["total_sessions_player"].gt(1),
        (df["session_index"] - 1) / (df["total_sessions_player"] - 1),
        0,
    )
    df["has_next_session"] = df.groupby("pid")["session_id"].shift(-1).notna()
    return df


def trajectory_bins(series):
    return pd.cut(
        series,
        bins=[-0.001, 0.0, 0.25, 0.5, 0.75, 1.0],
        labels=["first", "early", "middle", "late", "last"],
        include_lowest=True,
    )


def minimal_profile(df, label_col, cluster):
    target = df[df[label_col].eq(cluster)].copy()
    if target.empty:
        return {
            "cluster": int(cluster),
            "sessions": 0,
            "pct_subset_sessions": 0.0,
        }
    traj = trajectory_bins(target["trajectory_position_pct"]).value_counts(normalize=True).to_dict()
    return {
        "cluster": int(cluster),
        "sessions": int(len(target)),
        "pct_subset_sessions": float(len(target) / len(df) * 100),
        "players": int(target["pid"].nunique()),
        "pct_subset_players": float(target["pid"].nunique() / df["pid"].nunique() * 100),
        "duration_minutes_median": float(target["duration_minutes"].median()),
        "duration_minutes_p95": float(target["duration_minutes"].quantile(0.95)),
        "no_event_rate_pct": float((~target["any_job_task_item_event"]).mean() * 100),
        "missing_mode_rate_pct": float(target["primary_game_mode"].fillna("").eq("").mean() * 100),
        "first_session_rate_pct": float(target["session_index"].eq(1).mean() * 100),
        "has_next_session_return_rate_pct": float(target["has_next_session"].mean() * 100),
        "median_trajectory_position_pct": float(target["trajectory_position_pct"].median() * 100),
        "trajectory_first_pct": float(traj.get("first", 0) * 100),
        "trajectory_early_pct": float(traj.get("early", 0) * 100),
        "trajectory_middle_pct": float(traj.get("middle", 0) * 100),
        "trajectory_late_pct": float(traj.get("late", 0) * 100),
        "trajectory_last_pct": float(traj.get("last", 0) * 100),
        "median_minimal_sessions_per_player": float(target.groupby("pid").size().median()),
        "max_minimal_sessions_per_player": int(target.groupby("pid").size().max()),
    }


def all_cluster_profiles(df, label_col):
    rows = []
    for cluster, group in df.groupby(label_col, sort=True):
        rows.append(
            {
                "cluster": int(cluster),
                "size": int(len(group)),
                "pct": float(len(group) / len(df) * 100),
                "players": int(group["pid"].nunique()),
                "duration_median": float(group["duration_minutes"].median()),
                "duration_p95": float(group["duration_minutes"].quantile(0.95)),
                "event_count_median": float(group["event_count"].median()),
                "no_event_rate_pct": float((~group["any_job_task_item_event"]).mean() * 100),
                "missing_mode_rate_pct": float(group["primary_game_mode"].fillna("").eq("").mean() * 100),
                "trajectory_last_pct": float((trajectory_bins(group["trajectory_position_pct"]) == "last").mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def identify_minimal_cluster(profile):
    scored = profile.copy()
    scored["minimal_score"] = (
        scored["duration_median"].rank(method="min", ascending=True)
        + scored["event_count_median"].rank(method="min", ascending=True)
        + scored["no_event_rate_pct"].rank(method="min", ascending=False)
        + scored["missing_mode_rate_pct"].rank(method="min", ascending=False)
    )
    return int(scored.sort_values(["minimal_score", "duration_median"]).iloc[0]["cluster"]), scored


def refit_gmm_k5(subset, name):
    x = subset[PC_FEATURES].to_numpy(dtype=np.float64)
    cluster_eval.GMM_RESTARTS = 8
    cluster_eval.KMEANS_RESTARTS = 8
    cluster_eval.GMM_MAX_ITER = 140
    gmm = cluster_eval.fit_diag_gmm(x, 5, seed=6205)
    labels = gmm["labels"]
    result = subset.copy()
    result[f"{name}_gmm_k5_cluster"] = labels.astype(int)
    profile = all_cluster_profiles(result, f"{name}_gmm_k5_cluster")
    minimal_cluster, scored = identify_minimal_cluster(profile)
    summary = minimal_profile(result, f"{name}_gmm_k5_cluster", minimal_cluster)
    summary.update(
        {
            "sensitivity": name,
            "subset_sessions": int(len(subset)),
            "subset_players": int(subset["pid"].nunique()),
            "bic": float(gmm["bic"]),
            "aic": float(gmm["aic"]),
            "minimal_cluster_found": bool(
                summary["duration_minutes_median"] <= 2.0
                and summary["no_event_rate_pct"] >= 95.0
                and summary["missing_mode_rate_pct"] >= 90.0
            ),
        }
    )
    labels_out = result[["session_id", "pid", "session_index", f"{name}_gmm_k5_cluster"]]
    return summary, scored, labels_out


def date_distribution(df, label_col, cluster):
    target = df[df[label_col].eq(cluster)].copy()
    counts = target["login_date_utc"].dt.strftime("%Y-%m-%d").value_counts().rename_axis("date").reset_index(name="sessions")
    counts["pct_cluster"] = counts["sessions"] / len(target) * 100 if len(target) else 0
    all_counts = df["login_date_utc"].dt.strftime("%Y-%m-%d").value_counts().rename_axis("date").reset_index(name="subset_sessions")
    counts = counts.merge(all_counts, on="date", how="left")
    counts["cluster_share_of_day_pct"] = counts["sessions"] / counts["subset_sessions"] * 100
    return counts


def write_report(original_summary, refit_summaries, original_date_counts):
    lines = ["# Step 3.6 Date-regime Sensitivity", ""]
    lines.append("No broad model search was run. This checks the existing minimal cluster and two targeted GMM K5 refits.")
    lines.append("")
    lines.append("## Existing K5 Cluster 4 After Removing 2023-01-31 To 2023-02-07")
    for key, value in original_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Top dates after window removal:")
    lines.append("| date | sessions | pct of cluster | cluster share of day |")
    lines.append("|---|---:|---:|---:|")
    for row in original_date_counts.head(12).itertuples(index=False):
        lines.append(f"| {row.date} | {row.sessions} | {row.pct_cluster:.2f}% | {row.cluster_share_of_day_pct:.2f}% |")
    lines.append("")
    lines.append("## Targeted GMM K5 Refits")
    lines.append("| sensitivity | subset sessions | minimal cluster | sessions | pct | players | duration med | no-event | missing-mode | late+last | return | found? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for summary in refit_summaries:
        late_last = summary.get("trajectory_late_pct", 0) + summary.get("trajectory_last_pct", 0)
        lines.append(
            f"| {summary['sensitivity']} | {summary['subset_sessions']} | {summary['cluster']} | "
            f"{summary['sessions']} | {summary['pct_subset_sessions']:.2f}% | {summary['players']} | "
            f"{summary['duration_minutes_median']:.2f} | {summary['no_event_rate_pct']:.2f}% | "
            f"{summary['missing_mode_rate_pct']:.2f}% | {late_last:.2f}% | "
            f"{summary['has_next_session_return_rate_pct']:.2f}% | {summary['minimal_cluster_found']} |"
        )
    lines.append("")
    lines.append("## Gate")
    if all(s["minimal_cluster_found"] for s in refit_summaries) and original_summary["sessions"] > 0:
        lines.append(
            "The minimal/no-op structure survives date-window removal and reappears under both targeted K5 refits, including the pre-2023-01-31 period."
        )
    else:
        lines.append(
            "The minimal/no-op structure does not fully survive the sensitivity checks; V1 state representation should remain provisional."
        )
    (OUT_DIR / "step3_6_date_regime_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def purchase_anomaly_audit(df):
    target = df[df["original_gmm_k5_cluster"].eq(4)].copy()
    anomalous = target[target["items_purchased"].gt(0)].copy()
    anomalous = anomalous.sort_values(["items_purchased", "login_time_utc"], ascending=[False, True])
    anomaly_summary = {
        "minimal_cluster": 4,
        "minimal_cluster_sessions": int(len(target)),
        "purchase_minimal_sessions": int(len(anomalous)),
        "purchase_minimal_sessions_pct": float(len(anomalous) / len(target) * 100) if len(target) else 0,
        "purchase_events_in_purchase_minimal_sessions": int(anomalous["items_purchased"].sum()),
        "max_purchases_in_one_minimal_session": int(anomalous["items_purchased"].max()) if len(anomalous) else 0,
        "players_with_purchase_minimal_sessions": int(anomalous["pid"].nunique()),
        "median_duration_minutes": float(anomalous["duration_minutes"].median()) if len(anomalous) else np.nan,
        "p95_duration_minutes": float(anomalous["duration_minutes"].quantile(0.95)) if len(anomalous) else np.nan,
        "no_job_task_event_rate_pct": float(
            anomalous[["jobs_started", "jobs_resumed", "jobs_completed", "jobs_exited", "tasks_completed"]]
            .sum(axis=1)
            .eq(0)
            .mean()
            * 100
        )
        if len(anomalous)
        else np.nan,
        "missing_mode_rate_pct": float(anomalous["primary_game_mode"].fillna("").eq("").mean() * 100) if len(anomalous) else np.nan,
    }

    columns = [
        "session_id",
        "pid",
        "session_index",
        "login_time_utc",
        "exit_time_utc",
        "duration_minutes",
        "days_since_previous_session",
        "trajectory_position_pct",
        "has_next_session",
        "primary_game_mode",
        "items_purchased",
        "jobs_started",
        "jobs_resumed",
        "jobs_completed",
        "jobs_exited",
        "tasks_completed",
    ]
    anomalous[columns].to_csv(OUT_DIR / "minimal_cluster_purchase_anomaly_sessions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([anomaly_summary]).to_csv(
        OUT_DIR / "minimal_cluster_purchase_anomaly_summary.csv", index=False, encoding="utf-8-sig"
    )

    date_dist = (
        anomalous["login_date_utc"]
        .dt.strftime("%Y-%m-%d")
        .value_counts()
        .rename_axis("date")
        .reset_index(name="purchase_minimal_sessions")
    )
    if len(anomalous):
        date_dist["pct_purchase_minimal_sessions"] = date_dist["purchase_minimal_sessions"] / len(anomalous) * 100
        date_dist = date_dist.merge(
            anomalous.groupby(anomalous["login_date_utc"].dt.strftime("%Y-%m-%d"))["items_purchased"]
            .sum()
            .rename_axis("date")
            .reset_index(name="purchase_events"),
            on="date",
            how="left",
        )
    date_dist.to_csv(OUT_DIR / "minimal_cluster_purchase_anomaly_date_distribution.csv", index=False, encoding="utf-8-sig")

    flags = df[["session_id", "pid", "session_index"]].copy()
    flags["is_minimal_cluster_original_k5"] = df["original_gmm_k5_cluster"].eq(4)
    flags["is_minimal_cluster_purchase_anomaly"] = flags["is_minimal_cluster_original_k5"] & df["items_purchased"].gt(0)
    flags["minimal_cluster_purchase_count"] = np.where(flags["is_minimal_cluster_purchase_anomaly"], df["items_purchased"], 0)
    flags.to_csv(OUT_DIR / "minimal_cluster_purchase_anomaly_flags.csv", index=False, encoding="utf-8-sig")

    append_lines = ["", "## Minimal Cluster Purchase Anomalies", ""]
    for key, value in anomaly_summary.items():
        append_lines.append(f"- {key}: {value}")
    append_lines.append("")
    append_lines.append("Top purchase anomaly sessions:")
    append_lines.append("| session_id | pid | session_index | login_time_utc | duration min | purchases | job/task events | has next session |")
    append_lines.append("|---|---|---:|---|---:|---:|---:|---|")
    for row in anomalous.head(15).itertuples(index=False):
        job_task = row.jobs_started + row.jobs_resumed + row.jobs_completed + row.jobs_exited + row.tasks_completed
        append_lines.append(
            f"| {row.session_id} | {row.pid} | {row.session_index} | {row.login_time_utc} | "
            f"{row.duration_minutes:.2f} | {row.items_purchased} | {job_task} | {row.has_next_session} |"
        )
    append_lines.append("")
    append_lines.append(
        "These sessions should be flagged as purchase-burst/boundary-risk cases when interpreting the minimal/no-op state."
    )
    report_path = OUT_DIR / "step3_6_date_regime_sensitivity_report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + "\n".join(append_lines), encoding="utf-8")
    return anomaly_summary


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_base()
    abnormal_mask = df["login_date_utc"].between(WINDOW_START, WINDOW_END)
    no_window = df[~abnormal_mask].copy()
    pre_only = df[df["login_date_utc"].le(PRE_END)].copy()

    original_summary = minimal_profile(no_window, "original_gmm_k5_cluster", 4)
    original_summary.update(
        {
            "sensitivity": "original_labels_excluding_2023_01_31_to_2023_02_07",
            "subset_sessions": int(len(no_window)),
            "subset_players": int(no_window["pid"].nunique()),
            "removed_window_sessions": int(abnormal_mask.sum()),
            "removed_window_cluster4_sessions": int((abnormal_mask & df["original_gmm_k5_cluster"].eq(4)).sum()),
        }
    )
    original_date_counts = date_distribution(no_window, "original_gmm_k5_cluster", 4)

    refit_summaries = []
    refit_profiles = []
    for name, subset in [
        ("exclude_2023_01_31_to_2023_02_07", no_window),
        ("pre_2023_01_31_only", pre_only),
    ]:
        print(f"refit {name}: sessions={len(subset)}", flush=True)
        summary, profile, labels = refit_gmm_k5(subset, name)
        refit_summaries.append(summary)
        profile.insert(0, "sensitivity", name)
        refit_profiles.append(profile)
        labels.to_csv(OUT_DIR / f"{name}_gmm_k5_labels.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([original_summary]).to_csv(
        OUT_DIR / "original_cluster4_after_window_removal_summary.csv", index=False, encoding="utf-8-sig"
    )
    original_date_counts.to_csv(
        OUT_DIR / "original_cluster4_after_window_removal_date_distribution.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(refit_summaries).to_csv(
        OUT_DIR / "targeted_gmm_k5_refit_minimal_cluster_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(refit_profiles, ignore_index=True).to_csv(
        OUT_DIR / "targeted_gmm_k5_refit_all_cluster_profiles.csv", index=False, encoding="utf-8-sig"
    )
    write_report(original_summary, refit_summaries, original_date_counts)
    anomaly_summary = purchase_anomaly_audit(df)
    print(
        f"original_remaining_cluster4={original_summary['sessions']} "
        f"refits={[(s['sensitivity'], s['cluster'], s['sessions'], s['minimal_cluster_found']) for s in refit_summaries]} "
        f"purchase_anomaly_sessions={anomaly_summary['purchase_minimal_sessions']} "
        f"out={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
