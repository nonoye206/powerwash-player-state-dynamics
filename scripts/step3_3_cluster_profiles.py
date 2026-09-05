from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path("step2_v1/session_features_v1.csv")
DISCOVERY_PATH = Path("step2_v1/feature_diagnostics/state_discovery_input_v1.csv")
SIDECAR_PATH = Path("step2_v1/feature_diagnostics/state_discovery_sidecar_v1.csv")
PCA_SCORES_PATH = Path("step3_pca/pca_session_scores.csv")
METRICS_PATH = Path("step3_2_cluster_evaluation/cluster_model_metrics_k2_k8.csv")
LABEL_DIR = Path("step3_2_cluster_evaluation")
OUT_DIR = Path("step3_3_cluster_profiles")

CANDIDATES = [
    ("main_standardized", "kmeans", 8),
    ("pc1_pc5", "gmm_diag", 5),
    ("pc1_pc5", "kmeans", 6),
    ("pc1_pc5", "kmeans", 7),
]

RAW_PROFILE_FEATURES = [
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
    "mode_diversity",
]

DISCOVERY_FEATURES = [
    "first_session",
    "no_started_or_resumed_job",
    "log_duration_minutes",
    "log_gap_days",
    "log_jobs_started",
    "log_jobs_resumed",
    "log_tasks_completed",
    "purchase_any",
    "job_completion_ratio_capped",
    "job_exit_ratio_capped",
    "progression_gain",
    "mode_diversity",
]

PC_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]


def label_path(space, model, k):
    return LABEL_DIR / f"labels_{space}_{model}_k{k}.csv"


def load_base():
    features = pd.read_csv(FEATURES_PATH)
    discovery = pd.read_csv(DISCOVERY_PATH)
    sidecar = pd.read_csv(SIDECAR_PATH)
    pc = pd.read_csv(PCA_SCORES_PATH, usecols=["session_id", *PC_FEATURES])
    for frame_name, frame in [("discovery", discovery), ("sidecar", sidecar)]:
        if not features["session_id"].equals(frame["session_id"]):
            raise ValueError(f"{frame_name} is not aligned with session_features_v1.csv")
    discovery_for_merge = discovery.drop(columns=["pid", "session_index", "progression_gain", "mode_diversity"])
    base = features.merge(discovery_for_merge, on="session_id", how="left")
    base = base.merge(sidecar[["session_id", "job_completion_ratio_gt_1", "job_exit_ratio_gt_1"]], on="session_id", how="left")
    base = base.merge(pc, on="session_id", how="left")
    return base


def profile_candidate(base, space, model, k):
    labels = pd.read_csv(label_path(space, model, k))
    if not base["session_id"].equals(labels["session_id"]):
        raise ValueError(f"Labels not aligned for {space}/{model}/K{k}")
    df = base.copy()
    df["cluster"] = labels["label"].astype(int)
    n = len(df)

    rows = []
    for cluster, group in df.groupby("cluster", sort=True):
        row = {
            "candidate": f"{space}_{model}_k{k}",
            "space": space,
            "model": model,
            "k": k,
            "cluster": int(cluster),
            "size": int(len(group)),
            "pct": float(len(group) / n * 100),
            "players": int(group["pid"].nunique()),
            "primary_mode_top": str(group["primary_game_mode"].fillna("(missing)").replace("", "(missing)").mode().iloc[0]),
            "primary_mode_top_pct": float(
                group["primary_game_mode"].fillna("(missing)").replace("", "(missing)").value_counts(normalize=True).iloc[0] * 100
            ),
            "zero_duration_pct": float(group["is_zero_duration"].mean() * 100),
            "long_over_12h_pct": float(group["is_long_over_12h"].mean() * 100),
            "completion_ratio_gt_1_pct": float(group["job_completion_ratio_gt_1"].mean() * 100),
            "exit_ratio_gt_1_pct": float(group["job_exit_ratio_gt_1"].mean() * 100),
        }
        for col in RAW_PROFILE_FEATURES:
            s = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(s.mean()) if s.notna().any() else np.nan
            row[f"{col}_median"] = float(s.median()) if s.notna().any() else np.nan
        for col in DISCOVERY_FEATURES:
            s = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(s.mean()) if s.notna().any() else np.nan
        for col in PC_FEATURES:
            row[f"{col}_mean"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def mode_mix(base, space, model, k):
    labels = pd.read_csv(label_path(space, model, k))
    df = base[["session_id", "primary_game_mode"]].copy()
    df["cluster"] = labels["label"].astype(int)
    df["primary_game_mode"] = df["primary_game_mode"].fillna("(missing)").replace("", "(missing)")
    counts = (
        df.groupby(["cluster", "primary_game_mode"])
        .size()
        .rename("sessions")
        .reset_index()
        .sort_values(["cluster", "sessions"], ascending=[True, False])
    )
    totals = counts.groupby("cluster")["sessions"].transform("sum")
    counts["pct_within_cluster"] = counts["sessions"] / totals * 100
    counts.insert(0, "candidate", f"{space}_{model}_k{k}")
    counts.insert(1, "space", space)
    counts.insert(2, "model", model)
    counts.insert(3, "k", k)
    return counts


def compact_columns(profile):
    lead_cols = [
        "candidate",
        "cluster",
        "size",
        "pct",
        "players",
        "primary_mode_top",
        "primary_mode_top_pct",
        "duration_minutes_median",
        "days_since_previous_session_median",
        "jobs_started_median",
        "jobs_resumed_median",
        "tasks_completed_median",
        "items_purchased_median",
        "job_completion_ratio_median",
        "job_exit_ratio_median",
        "progression_gain_median",
        "mode_diversity_median",
        "first_session_mean",
        "no_started_or_resumed_job_mean",
        "purchase_any_mean",
        "long_over_12h_pct",
        "completion_ratio_gt_1_pct",
        "exit_ratio_gt_1_pct",
        "PC1_mean",
        "PC2_mean",
        "PC3_mean",
        "PC4_mean",
        "PC5_mean",
    ]
    return profile[lead_cols]


def write_report(metrics, compact):
    lines = ["# Step 3.3 Candidate Cluster Profiles", ""]
    lines.append("No state names are assigned here. Clusters are described only by numeric profiles.")
    lines.append("")
    lines.append("## Candidate Metrics")
    lines.append("| candidate | silhouette | CH | DB | BIC | AIC | min cluster % | max cluster % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in metrics.iterrows():
        candidate = f"{row.space}_{row.model}_k{row.k}"
        subset = compact[compact["candidate"].eq(candidate)]
        bic = "" if pd.isna(row.bic) else f"{row.bic:.1f}"
        aic = "" if pd.isna(row.aic) else f"{row.aic:.1f}"
        lines.append(
            f"| {candidate} | {row.silhouette_sampled:.4f} | {row.calinski_harabasz:.1f} | "
            f"{row.davies_bouldin:.4f} | {bic} | {aic} | {subset['pct'].min():.2f}% | {subset['pct'].max():.2f}% |"
        )
    lines.append("")
    lines.append("## Cluster Profiles")
    for candidate, group in compact.groupby("candidate", sort=False):
        lines.append(f"### {candidate}")
        lines.append("| cluster | size | pct | duration med | gap med | jobs started med | jobs resumed med | tasks med | purchase rate | completion med | exit med | progression med | mode diversity med | top mode |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in group.sort_values("cluster").itertuples(index=False):
            lines.append(
                f"| {row.cluster} | {row.size} | {row.pct:.2f}% | "
                f"{row.duration_minutes_median:.2f} | {row.days_since_previous_session_median:.2f} | "
                f"{row.jobs_started_median:.2f} | {row.jobs_resumed_median:.2f} | {row.tasks_completed_median:.2f} | "
                f"{row.purchase_any_mean * 100:.2f}% | {row.job_completion_ratio_median:.2f} | "
                f"{row.job_exit_ratio_median:.2f} | {row.progression_gain_median:.2f} | "
                f"{row.mode_diversity_median:.2f} | {row.primary_mode_top} ({row.primary_mode_top_pct:.1f}%) |"
            )
        lines.append("")
    lines.append("## Reading Notes")
    lines.append("- `purchase rate` is the mean of `purchase_any`, not raw purchase count.")
    lines.append("- Median gap is blank/NaN for first-session-heavy clusters only in the detailed CSV; Markdown renders numeric medians when available.")
    lines.append("- Use this file to decide which configuration is interpretable enough before assigning state names.")
    (OUT_DIR / "step3_3_cluster_profiles_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    base = load_base()
    all_profiles = []
    all_modes = []
    metrics = pd.read_csv(METRICS_PATH)
    metric_rows = []
    for space, model, k in CANDIDATES:
        profile = profile_candidate(base, space, model, k)
        all_profiles.append(profile)
        all_modes.append(mode_mix(base, space, model, k))
        metric_rows.append(metrics[metrics["space"].eq(space) & metrics["model"].eq(model) & metrics["k"].eq(k)])

    profile_df = pd.concat(all_profiles, ignore_index=True)
    compact = compact_columns(profile_df)
    mode_df = pd.concat(all_modes, ignore_index=True)
    selected_metrics = pd.concat(metric_rows, ignore_index=True)

    profile_df.to_csv(OUT_DIR / "candidate_cluster_profiles_full.csv", index=False, encoding="utf-8-sig")
    compact.to_csv(OUT_DIR / "candidate_cluster_profiles_compact.csv", index=False, encoding="utf-8-sig")
    mode_df.to_csv(OUT_DIR / "candidate_cluster_mode_mix.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(OUT_DIR / "candidate_model_metrics.csv", index=False, encoding="utf-8-sig")
    write_report(selected_metrics, compact)
    print(f"profiles={len(profile_df)} candidates={len(CANDIDATES)} out={OUT_DIR}")


if __name__ == "__main__":
    main()
