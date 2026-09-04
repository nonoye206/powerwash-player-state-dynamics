from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path("step2_v1/session_features_v1.csv")
LABELS_PATH = Path("step3_2_cluster_evaluation/labels_pc1_pc5_gmm_diag_k5.csv")
PURCHASE_FLAGS_PATH = Path("step3_6_date_regime_sensitivity/minimal_cluster_purchase_anomaly_flags.csv")
OUT_DIR = Path("step4_1_state_sequences")

N_STATES = 5
STATE_COL = "state_id"
UPDATE_WINDOW_START = pd.Timestamp("2023-01-31")
UPDATE_WINDOW_END = pd.Timestamp("2023-02-07")


def load_labeled_sessions():
    sessions = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    if not sessions["session_id"].equals(labels["session_id"]):
        raise ValueError("Labels are not aligned with session_features_v1.csv")
    sessions[STATE_COL] = labels["label"].astype(int)
    sessions["login_time_utc"] = pd.to_datetime(sessions["login_time_utc"])
    sessions["exit_time_utc"] = pd.to_datetime(sessions["exit_time_utc"])
    sessions["is_update_window_2023_01_31_to_2023_02_07"] = sessions["login_time_utc"].dt.normalize().between(
        UPDATE_WINDOW_START, UPDATE_WINDOW_END
    )
    if PURCHASE_FLAGS_PATH.exists():
        flags = pd.read_csv(
            PURCHASE_FLAGS_PATH,
            usecols=["session_id", "is_minimal_cluster_purchase_anomaly", "minimal_cluster_purchase_count"],
        )
        sessions = sessions.merge(flags, on="session_id", how="left")
        sessions["is_minimal_cluster_purchase_anomaly"] = sessions["is_minimal_cluster_purchase_anomaly"].fillna(False).astype(bool)
        sessions["minimal_cluster_purchase_count"] = sessions["minimal_cluster_purchase_count"].fillna(0).astype(int)
    else:
        sessions["is_minimal_cluster_purchase_anomaly"] = False
        sessions["minimal_cluster_purchase_count"] = 0
    sessions = sessions.sort_values(["pid", "session_index", "login_time_utc"]).reset_index(drop=True)
    sessions["prev_state_id"] = sessions.groupby("pid")[STATE_COL].shift(1)
    sessions["next_state_id"] = sessions.groupby("pid")[STATE_COL].shift(-1)
    sessions["prev_session_id"] = sessions.groupby("pid")["session_id"].shift(1)
    sessions["next_session_id"] = sessions.groupby("pid")["session_id"].shift(-1)
    sessions["transition_index_from_prev"] = sessions.groupby("pid").cumcount()
    sessions["total_sessions_player"] = sessions.groupby("pid")["session_id"].transform("size")
    return sessions


def build_player_sequences(sessions):
    rows = []
    for pid, group in sessions.groupby("pid", sort=False):
        states = group[STATE_COL].astype(int).tolist()
        session_ids = group["session_id"].astype(str).tolist()
        rows.append(
            {
                "pid": pid,
                "n_sessions": int(len(group)),
                "n_transitions": int(max(len(group) - 1, 0)),
                "first_session_id": session_ids[0],
                "last_session_id": session_ids[-1],
                "first_state": int(states[0]),
                "last_state": int(states[-1]),
                "state_sequence": " ".join(map(str, states)),
                "session_id_sequence": " ".join(session_ids),
                "has_minimal_state": bool((group[STATE_COL] == 4).any()),
                "has_purchase_anomaly": bool(group["is_minimal_cluster_purchase_anomaly"].any()),
                "update_window_sessions": int(group["is_update_window_2023_01_31_to_2023_02_07"].sum()),
            }
        )
    return pd.DataFrame(rows)


def transition_edges(sessions):
    edges = sessions[sessions["next_state_id"].notna()].copy()
    edges["from_state"] = edges[STATE_COL].astype(int)
    edges["to_state"] = edges["next_state_id"].astype(int)
    edges["from_session_id"] = edges["session_id"]
    edges["to_session_id"] = edges["next_session_id"]
    edges["transition_gap_days"] = (
        sessions.groupby("pid")["login_time_utc"].shift(-1).loc[edges.index] - edges["exit_time_utc"]
    ).dt.total_seconds() / 86400.0
    return edges[
        [
            "pid",
            "transition_index_from_prev",
            "from_session_id",
            "to_session_id",
            "from_state",
            "to_state",
            "transition_gap_days",
            "is_update_window_2023_01_31_to_2023_02_07",
            "is_minimal_cluster_purchase_anomaly",
        ]
    ]


def matrix_from_edges(edges):
    counts = np.zeros((N_STATES, N_STATES), dtype=int)
    for from_state, to_state in zip(edges["from_state"], edges["to_state"]):
        counts[int(from_state), int(to_state)] += 1
    count_df = pd.DataFrame(counts, index=[f"from_{i}" for i in range(N_STATES)], columns=[f"to_{i}" for i in range(N_STATES)])
    probs = counts / counts.sum(axis=1, keepdims=True)
    prob_df = pd.DataFrame(probs, index=count_df.index, columns=count_df.columns)
    long_rows = []
    for i in range(N_STATES):
        row_sum = counts[i].sum()
        for j in range(N_STATES):
            long_rows.append(
                {
                    "from_state": i,
                    "to_state": j,
                    "transition_count": int(counts[i, j]),
                    "transition_probability": float(counts[i, j] / row_sum) if row_sum else np.nan,
                }
            )
    return count_df, prob_df, pd.DataFrame(long_rows)


def state_summary(sessions, edges):
    state_counts = sessions[STATE_COL].value_counts().sort_index()
    starts = sessions.groupby("pid").head(1)[STATE_COL].value_counts().sort_index()
    ends = sessions.groupby("pid").tail(1)[STATE_COL].value_counts().sort_index()
    out_counts = edges["from_state"].value_counts().sort_index()
    in_counts = edges["to_state"].value_counts().sort_index()
    rows = []
    for state in range(N_STATES):
        rows.append(
            {
                "state": state,
                "session_count": int(state_counts.get(state, 0)),
                "session_pct": float(state_counts.get(state, 0) / len(sessions) * 100),
                "players_with_state": int(sessions.loc[sessions[STATE_COL].eq(state), "pid"].nunique()),
                "start_sequence_count": int(starts.get(state, 0)),
                "end_sequence_count": int(ends.get(state, 0)),
                "incoming_transition_count": int(in_counts.get(state, 0)),
                "outgoing_transition_count": int(out_counts.get(state, 0)),
                "self_transition_count": int(((edges["from_state"] == state) & (edges["to_state"] == state)).sum()),
                "self_transition_probability": float(
                    ((edges["from_state"] == state) & (edges["to_state"] == state)).sum() / out_counts.get(state, np.nan)
                )
                if out_counts.get(state, 0)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(sessions, player_sequences, edges, count_df, prob_df, state_stats):
    def df_to_md(df):
        headers = [""] + list(df.columns)
        rows = []
        for idx, row in df.iterrows():
            rows.append([idx] + [row[col] for col in df.columns])
        lines = ["| " + " | ".join(map(str, headers)) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)

    lines = ["# Step 4.1 Player-level State Sequences and Transition Matrix", ""]
    lines.append("State source: `pc1_pc5_gmm_diag_k5` labels. State IDs are numeric only; no state names are assigned here.")
    lines.append("")
    lines.append("## Sequence Summary")
    lines.append(f"- sessions: {len(sessions)}")
    lines.append(f"- players: {sessions['pid'].nunique()}")
    lines.append(f"- players with at least 2 sessions: {(player_sequences['n_sessions'] >= 2).sum()}")
    lines.append(f"- transitions: {len(edges)}")
    lines.append(f"- minimal purchase anomaly sessions flagged: {int(sessions['is_minimal_cluster_purchase_anomaly'].sum())}")
    lines.append(f"- update-window sessions flagged: {int(sessions['is_update_window_2023_01_31_to_2023_02_07'].sum())}")
    lines.append("")
    lines.append("## State Summary")
    lines.append("| state | sessions | session % | players | starts | ends | incoming | outgoing | self prob |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in state_stats.itertuples(index=False):
        lines.append(
            f"| {row.state} | {row.session_count} | {row.session_pct:.2f}% | {row.players_with_state} | "
            f"{row.start_sequence_count} | {row.end_sequence_count} | {row.incoming_transition_count} | "
            f"{row.outgoing_transition_count} | {row.self_transition_probability:.4f} |"
        )
    lines.append("")
    lines.append("## Transition Counts")
    lines.append(df_to_md(count_df))
    lines.append("")
    lines.append("## Transition Probabilities")
    lines.append(df_to_md(prob_df.round(4)))
    lines.append("")
    lines.append("## Notes")
    lines.append("- Transition rows are row-normalized: each row sums over the next state conditional on the current state.")
    lines.append("- Players with one session contribute to state counts and start/end counts, but not transitions.")
    lines.append("- Date-window and purchase-anomaly flags are included for Step 4 sensitivity checks.")
    (OUT_DIR / "step4_1_state_sequences_transition_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sessions = load_labeled_sessions()
    player_sequences = build_player_sequences(sessions)
    edges = transition_edges(sessions)
    count_df, prob_df, long_df = matrix_from_edges(edges)
    state_stats = state_summary(sessions, edges)

    session_cols = [
        "session_id",
        "pid",
        "session_index",
        "login_time_utc",
        "exit_time_utc",
        STATE_COL,
        "prev_state_id",
        "next_state_id",
        "duration_minutes",
        "primary_game_mode",
        "is_update_window_2023_01_31_to_2023_02_07",
        "is_minimal_cluster_purchase_anomaly",
        "minimal_cluster_purchase_count",
    ]
    sessions[session_cols].to_csv(OUT_DIR / "session_state_assignments_v1.csv", index=False, encoding="utf-8-sig")
    player_sequences.to_csv(OUT_DIR / "player_state_sequences_v1.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUT_DIR / "state_transition_edges_v1.csv", index=False, encoding="utf-8-sig")
    count_df.to_csv(OUT_DIR / "transition_matrix_counts_5x5.csv", encoding="utf-8-sig")
    prob_df.to_csv(OUT_DIR / "transition_matrix_probabilities_5x5.csv", encoding="utf-8-sig")
    long_df.to_csv(OUT_DIR / "transition_matrix_long_v1.csv", index=False, encoding="utf-8-sig")
    state_stats.to_csv(OUT_DIR / "state_sequence_summary_v1.csv", index=False, encoding="utf-8-sig")
    write_report(sessions, player_sequences, edges, count_df, prob_df, state_stats)
    print(
        f"sessions={len(sessions)} players={sessions['pid'].nunique()} "
        f"transitions={len(edges)} out={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
