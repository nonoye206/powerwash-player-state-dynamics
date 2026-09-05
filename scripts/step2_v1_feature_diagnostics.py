import json
from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("step2_v1/session_features_v1.csv")
OUT_DIR = Path("step2_v1/feature_diagnostics")

CORE_FEATURES = [
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
]

NUMERIC_FEATURES = [f for f in CORE_FEATURES if f != "primary_game_mode"]
EVENT_COUNT_FEATURES = [
    "jobs_started",
    "jobs_resumed",
    "jobs_completed",
    "jobs_exited",
    "tasks_completed",
    "items_purchased",
]


def pct(series, q):
    return float(series.quantile(q / 100.0))


def summarize_numeric(df):
    rows = []
    n = len(df)
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()
        row = {
            "feature": col,
            "non_null": int(non_null.size),
            "missing_pct": float(s.isna().mean() * 100),
            "zero_pct": float(s.eq(0).mean() * 100),
            "negative_count": int(s.lt(0).sum()),
            "mean": float(non_null.mean()) if non_null.size else np.nan,
            "std": float(non_null.std()) if non_null.size else np.nan,
            "min": float(non_null.min()) if non_null.size else np.nan,
            "p01": pct(non_null, 1) if non_null.size else np.nan,
            "p05": pct(non_null, 5) if non_null.size else np.nan,
            "p25": pct(non_null, 25) if non_null.size else np.nan,
            "p50": pct(non_null, 50) if non_null.size else np.nan,
            "p75": pct(non_null, 75) if non_null.size else np.nan,
            "p95": pct(non_null, 95) if non_null.size else np.nan,
            "p99": pct(non_null, 99) if non_null.size else np.nan,
            "max": float(non_null.max()) if non_null.size else np.nan,
            "top_1pct_share_of_sum": np.nan,
            "unique_values": int(non_null.nunique()) if non_null.size else 0,
        }
        total = non_null.sum()
        if non_null.size and total > 0:
            top_n = max(1, int(np.ceil(n * 0.01)))
            row["top_1pct_share_of_sum"] = float(non_null.nlargest(top_n).sum() / total)
        rows.append(row)
    return pd.DataFrame(rows)


def mode_distribution(df):
    total = len(df)
    values = df["primary_game_mode"].fillna("").replace("", "(missing)")
    out = values.value_counts(dropna=False).rename_axis("primary_game_mode").reset_index(name="sessions")
    out["pct"] = out["sessions"] / total * 100
    return out


def correlation_outputs(df):
    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    corr_raw = numeric.corr(method="spearman")

    transformed = numeric.copy()
    for col in [
        "duration_minutes",
        "days_since_previous_session",
        "player_lifetime_days",
        *EVENT_COUNT_FEATURES,
    ]:
        transformed[f"log1p_{col}"] = np.log1p(transformed[col])
    transformed = transformed[
        [
            "log1p_duration_minutes",
            "log1p_days_since_previous_session",
            "log1p_player_lifetime_days",
            "log1p_jobs_started",
            "log1p_jobs_resumed",
            "log1p_jobs_completed",
            "log1p_jobs_exited",
            "log1p_tasks_completed",
            "log1p_items_purchased",
            "job_completion_ratio",
            "job_exit_ratio",
            "progression_gain",
            "campaign_progression_gain",
            "mode_diversity",
        ]
    ]
    corr_transformed = transformed.corr(method="spearman")

    pairs = []
    cols = list(corr_transformed.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr_transformed.loc[left, right]
            if pd.notna(value):
                pairs.append({"feature_a": left, "feature_b": right, "spearman_corr": float(value)})
    high_pairs = pd.DataFrame(pairs).assign(abs_corr=lambda x: x["spearman_corr"].abs())
    high_pairs = high_pairs.sort_values("abs_corr", ascending=False)
    return corr_raw, corr_transformed, high_pairs


def extreme_rows(df):
    specs = {
        "longest_sessions": ("duration_minutes", False),
        "largest_session_gaps": ("days_since_previous_session", False),
        "most_jobs_started": ("jobs_started", False),
        "most_tasks_completed": ("tasks_completed", False),
        "most_items_purchased": ("items_purchased", False),
        "largest_progression_gain": ("progression_gain", False),
        "largest_campaign_progression_gain": ("campaign_progression_gain", False),
    }
    keep = [
        "session_id",
        "pid",
        "session_index",
        "duration_minutes",
        "days_since_previous_session",
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
    ]
    outputs = {}
    for name, (col, ascending) in specs.items():
        outputs[name] = df.sort_values(col, ascending=ascending, na_position="last").head(25)[keep]
    return outputs


def player_influence(df):
    sessions_per_player = df.groupby("pid").size().rename("sessions").sort_values(ascending=False)
    top = sessions_per_player.head(50).reset_index()
    summary = {
        "players": int(sessions_per_player.size),
        "sessions": int(len(df)),
        "max_sessions_per_player": int(sessions_per_player.max()),
        "p95_sessions_per_player": float(sessions_per_player.quantile(0.95)),
        "top_1pct_players_session_share": float(
            sessions_per_player.head(max(1, int(np.ceil(sessions_per_player.size * 0.01)))).sum() / len(df)
        ),
        "top_5pct_players_session_share": float(
            sessions_per_player.head(max(1, int(np.ceil(sessions_per_player.size * 0.05)))).sum() / len(df)
        ),
    }
    return summary, top


def recommended_policy(summary, high_corr):
    decisions = [
        {
            "feature": "duration_minutes",
            "decision": "use_log1p_winsorized",
            "reason": "Core engagement intensity. Very right-skewed, with >12h sessions present.",
        },
        {
            "feature": "days_since_previous_session",
            "decision": "use_log1p_with_first_session_indicator",
            "reason": "Important recency signal. Missing mainly first sessions.",
        },
        {
            "feature": "player_lifetime_days",
            "decision": "exclude_from_main_discovery_use_for_interpretation",
            "reason": "Captures longitudinal position, but may make states reflect lifecycle stage rather than session behavior.",
        },
        {
            "feature": "jobs_started",
            "decision": "use_log1p_winsorized",
            "reason": "Direct activity breadth signal.",
        },
        {
            "feature": "jobs_resumed",
            "decision": "use_log1p_winsorized",
            "reason": "Captures returning to incomplete jobs.",
        },
        {
            "feature": "jobs_completed",
            "decision": "drop_or_keep_only_for_completion_variant",
            "reason": "Highly redundant with job_completion_ratio and campaign/progression signals.",
        },
        {
            "feature": "jobs_exited",
            "decision": "drop_or_keep_only_for_exit_variant",
            "reason": "Highly coupled to starts/resumes; job_exit_ratio is more interpretable.",
        },
        {
            "feature": "tasks_completed",
            "decision": "use_log1p_winsorized",
            "reason": "Best compact signal for within-session work volume.",
        },
        {
            "feature": "items_purchased",
            "decision": "use_binary_or_log1p_winsorized",
            "reason": "Sparse but meaningful behavior; binary purchase_any may be more stable for clustering.",
        },
        {
            "feature": "job_completion_ratio",
            "decision": "use_with_no_job_indicator",
            "reason": "Interpretable outcome quality signal; undefined when no started/resumed job.",
        },
        {
            "feature": "job_exit_ratio",
            "decision": "use_with_no_job_indicator",
            "reason": "Interpretable disengagement/friction signal; undefined when no started/resumed job.",
        },
        {
            "feature": "progression_gain",
            "decision": "use_winsorized",
            "reason": "No negative values and directly captures level progress inside session.",
        },
        {
            "feature": "campaign_progression_gain",
            "decision": "drop_initially",
            "reason": "Sparse/redundant with completion and progression; keep for sensitivity analysis.",
        },
        {
            "feature": "primary_game_mode",
            "decision": "exclude_from_unsupervised_numeric_state_discovery",
            "reason": "Categorical label can dominate distance metrics; use for post-hoc state interpretation.",
        },
        {
            "feature": "mode_diversity",
            "decision": "use",
            "reason": "Compact numeric signal for mode switching.",
        },
    ]
    return pd.DataFrame(decisions)


def make_state_discovery_input(df):
    selected = pd.DataFrame(
        {
            "session_id": df["session_id"],
            "pid": df["pid"],
            "session_index": df["session_index"],
        }
    )

    selected["first_session"] = df["days_since_previous_session"].isna().astype(int)
    selected["no_started_or_resumed_job"] = (
        (df["jobs_started"].fillna(0) + df["jobs_resumed"].fillna(0)).eq(0)
    ).astype(int)

    def log_winsor(col):
        s = pd.to_numeric(df[col], errors="coerce")
        cap = s.quantile(0.99)
        return np.log1p(s.clip(lower=0, upper=cap).fillna(0))

    selected["log_duration_minutes"] = log_winsor("duration_minutes")
    selected["log_gap_days"] = np.log1p(
        pd.to_numeric(df["days_since_previous_session"], errors="coerce")
        .clip(lower=0, upper=df["days_since_previous_session"].quantile(0.99))
        .fillna(0)
    )
    selected["log_jobs_started"] = log_winsor("jobs_started")
    selected["log_jobs_resumed"] = log_winsor("jobs_resumed")
    selected["log_tasks_completed"] = log_winsor("tasks_completed")
    selected["purchase_any"] = pd.to_numeric(df["items_purchased"], errors="coerce").fillna(0).gt(0).astype(int)

    selected["job_completion_ratio_capped"] = (
        pd.to_numeric(df["job_completion_ratio"], errors="coerce").clip(lower=0, upper=1).fillna(0)
    )
    selected["job_exit_ratio_capped"] = (
        pd.to_numeric(df["job_exit_ratio"], errors="coerce").clip(lower=0, upper=1).fillna(0)
    )
    selected["progression_gain"] = (
        pd.to_numeric(df["progression_gain"], errors="coerce")
        .clip(lower=0, upper=df["progression_gain"].quantile(0.99))
        .fillna(0)
    )
    selected["mode_diversity"] = pd.to_numeric(df["mode_diversity"], errors="coerce").fillna(0)

    sidecar = df[
        [
            "session_id",
            "pid",
            "session_index",
            "player_lifetime_days",
            "jobs_completed",
            "jobs_exited",
            "items_purchased",
            "campaign_progression_gain",
            "primary_game_mode",
            "is_zero_duration",
            "is_long_over_12h",
        ]
    ].copy()
    sidecar["job_completion_ratio_gt_1"] = pd.to_numeric(df["job_completion_ratio"], errors="coerce").gt(1)
    sidecar["job_exit_ratio_gt_1"] = pd.to_numeric(df["job_exit_ratio"], errors="coerce").gt(1)
    return selected, sidecar


def write_markdown(summary_df, mode_df, high_corr, player_summary, decisions):
    lines = ["# Step 2 V1 Feature Diagnostics", ""]
    lines.append("## Recommended Discovery Set")
    recommended = decisions[decisions["decision"].str.startswith("use")]
    lines.append(
        "`duration_minutes`, `days_since_previous_session`, `jobs_started`, "
        "`jobs_resumed`, `tasks_completed`, `items_purchased`, "
        "`job_completion_ratio`, `job_exit_ratio`, `progression_gain`, `mode_diversity`, "
        "plus `first_session` and `no_started_or_resumed_job` indicators."
    )
    lines.append("")
    lines.append("Use transformed numeric inputs for discovery:")
    lines.append("- log1p transform duration, gap, event counts, and raw heavy-tailed counts.")
    lines.append("- Winsorize heavy-tailed numeric features at P99 before scaling.")
    lines.append("- Add indicators for first session and no started/resumed job before filling ratio NaNs.")
    lines.append("- Keep `primary_game_mode` and `player_lifetime_days` for post-hoc state interpretation, not the primary distance space.")
    lines.append("- Use `purchase_any` rather than raw purchase count for the main V1 clustering.")
    lines.append("- Cap `job_completion_ratio` and `job_exit_ratio` to [0, 1]; keep >1 flags in a sidecar file.")
    lines.append("")
    lines.append("## Main Distribution Notes")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['feature']}`: missing {row['missing_pct']:.2f}%, zero {row['zero_pct']:.2f}%, "
            f"p50 {row['p50']:.4g}, p95 {row['p95']:.4g}, p99 {row['p99']:.4g}, max {row['max']:.4g}."
        )
    lines.append("")
    lines.append("## High Correlations After Log Transforms")
    for _, row in high_corr.head(15).iterrows():
        lines.append(f"- `{row['feature_a']}` vs `{row['feature_b']}`: {row['spearman_corr']:.3f}")
    lines.append("")
    lines.append("## Player Influence")
    for key, value in player_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Primary Game Mode")
    for _, row in mode_df.head(10).iterrows():
        lines.append(f"- {row['primary_game_mode']}: {int(row['sessions'])} sessions ({row['pct']:.2f}%)")
    lines.append("")
    lines.append("## Feature Decisions")
    for _, row in decisions.iterrows():
        lines.append(f"- `{row['feature']}`: {row['decision']}. {row['reason']}")
    (OUT_DIR / "feature_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)

    summary = summarize_numeric(df)
    mode_df = mode_distribution(df)
    corr_raw, corr_transformed, high_corr = correlation_outputs(df)
    extremes = extreme_rows(df)
    player_summary, top_players = player_influence(df)
    decisions = recommended_policy(summary, high_corr)
    selected, sidecar = make_state_discovery_input(df)

    summary.to_csv(OUT_DIR / "feature_distribution_summary.csv", index=False, encoding="utf-8-sig")
    mode_df.to_csv(OUT_DIR / "primary_game_mode_distribution.csv", index=False, encoding="utf-8-sig")
    corr_raw.to_csv(OUT_DIR / "spearman_correlation_raw.csv", encoding="utf-8-sig")
    corr_transformed.to_csv(OUT_DIR / "spearman_correlation_log_transformed.csv", encoding="utf-8-sig")
    high_corr.to_csv(OUT_DIR / "high_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    top_players.to_csv(OUT_DIR / "top_players_by_session_count.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUT_DIR / "feature_decisions_v1.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "state_discovery_input_v1.csv", index=False, encoding="utf-8-sig")
    sidecar.to_csv(OUT_DIR / "state_discovery_sidecar_v1.csv", index=False, encoding="utf-8-sig")
    for name, frame in extremes.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    write_markdown(summary, mode_df, high_corr, player_summary, decisions)
    print(json.dumps({"rows": int(len(df)), "outputs": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
