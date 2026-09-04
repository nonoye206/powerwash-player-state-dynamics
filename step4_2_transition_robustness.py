from pathlib import Path

import numpy as np
import pandas as pd


ASSIGNMENTS_PATH = Path("step4_1_state_sequences/session_state_assignments_v1.csv")
OUT_DIR = Path("step4_2_transition_robustness")
N_STATES = 5
UPDATE_WINDOW_START = pd.Timestamp("2023-01-31")
UPDATE_WINDOW_END = pd.Timestamp("2023-02-07")


def load_assignments():
    df = pd.read_csv(ASSIGNMENTS_PATH)
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["exit_time_utc"] = pd.to_datetime(df["exit_time_utc"])
    df["state_id"] = df["state_id"].astype(int)
    return df.sort_values(["pid", "session_index", "login_time_utc"]).reset_index(drop=True)


def build_edges(df):
    work = df.sort_values(["pid", "session_index", "login_time_utc"]).copy()
    work["next_state"] = work.groupby("pid")["state_id"].shift(-1)
    work["next_session_id"] = work.groupby("pid")["session_id"].shift(-1)
    work["next_login_time_utc"] = work.groupby("pid")["login_time_utc"].shift(-1)
    edges = work[work["next_state"].notna()].copy()
    edges["from_state"] = edges["state_id"].astype(int)
    edges["to_state"] = edges["next_state"].astype(int)
    edges["transition_gap_days"] = (edges["next_login_time_utc"] - edges["exit_time_utc"]).dt.total_seconds() / 86400.0
    return edges[["pid", "session_id", "next_session_id", "from_state", "to_state", "transition_gap_days"]]


def count_matrix(edges):
    mat = np.zeros((N_STATES, N_STATES), dtype=float)
    for i, j in zip(edges["from_state"], edges["to_state"]):
        mat[int(i), int(j)] += 1
    return mat


def row_normalize(mat):
    denom = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, denom, out=np.full_like(mat, np.nan, dtype=float), where=denom != 0)


def player_balanced_matrix(edges):
    mats = []
    player_rows = []
    for pid, group in edges.groupby("pid", sort=False):
        mat = count_matrix(group)
        probs = row_normalize(mat)
        valid_rows = ~np.isnan(probs).all(axis=1)
        if valid_rows.any():
            mats.append(probs)
            player_rows.append(
                {
                    "pid": pid,
                    "transitions": int(len(group)),
                    "states_with_outgoing": int(valid_rows.sum()),
                }
            )
    stack = np.stack(mats)
    # Average each from-state over players who actually have outgoing transitions from that state.
    balanced = np.nanmean(stack, axis=0)
    return balanced, pd.DataFrame(player_rows)


def matrix_to_df(mat):
    return pd.DataFrame(mat, index=[f"from_{i}" for i in range(N_STATES)], columns=[f"to_{i}" for i in range(N_STATES)])


def matrix_long(name, mat):
    rows = []
    for i in range(N_STATES):
        for j in range(N_STATES):
            rows.append({"matrix": name, "from_state": i, "to_state": j, "value": float(mat[i, j])})
    return pd.DataFrame(rows)


def compare_matrices(mats):
    names = list(mats)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diff = mats[a] - mats[b]
            rows.append(
                {
                    "matrix_a": a,
                    "matrix_b": b,
                    "max_abs_diff": float(np.nanmax(np.abs(diff))),
                    "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                    "frobenius_norm": float(np.sqrt(np.nansum(diff * diff))),
                }
            )
    return pd.DataFrame(rows)


def transition_features(name, prob):
    rows = []
    for state in range(N_STATES):
        row = prob[state].copy()
        top_to = int(np.nanargmax(row))
        rows.append(
            {
                "matrix": name,
                "state": state,
                "self_transition": float(prob[state, state]),
                "top_outgoing_to_state": top_to,
                "top_outgoing_probability": float(prob[state, top_to]),
                "to_s1": float(prob[state, 1]),
                "to_s2": float(prob[state, 2]),
                "to_s1_or_s2": float(prob[state, 1] + prob[state, 2]),
            }
        )
    rel = {
        "matrix": name,
        "s0_to_s1": float(prob[0, 1]),
        "s0_top_is_s1": bool(np.nanargmax(prob[0]) == 1),
        "s1_to_s2": float(prob[1, 2]),
        "s2_to_s1": float(prob[2, 1]),
        "s1_s2_bidirectional_sum": float(prob[1, 2] + prob[2, 1]),
        "s3_to_s1_or_s2": float(prob[3, 1] + prob[3, 2]),
        "s4_self": float(prob[4, 4]),
        "s4_to_s1_or_s2": float(prob[4, 1] + prob[4, 2]),
    }
    return pd.DataFrame(rows), rel


def save_matrix(name, mat, suffix):
    matrix_to_df(mat).to_csv(OUT_DIR / f"{name}_{suffix}.csv", encoding="utf-8-sig")


def write_report(summary_rows, relation_df, comparison_df, matrices):
    lines = ["# Step 4.2 Transition Robustness", ""]
    lines.append("Scope: validate transition structure only. No churn analysis and no state naming.")
    lines.append("")
    lines.append("## Matrix Summary")
    lines.append("| matrix | sessions | players | transitions |")
    lines.append("|---|---:|---:|---:|")
    for row in summary_rows:
        lines.append(f"| {row['matrix']} | {row['sessions']} | {row['players']} | {row['transitions']} |")
    lines.append("")
    lines.append("## Key Relations")
    lines.append("| matrix | S0->S1 | S0 top S1 | S1->S2 | S2->S1 | S1<->S2 sum | S3->S1/S2 | S4 self | S4->S1/S2 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in relation_df.itertuples(index=False):
        lines.append(
            f"| {row.matrix} | {row.s0_to_s1:.4f} | {row.s0_top_is_s1} | {row.s1_to_s2:.4f} | "
            f"{row.s2_to_s1:.4f} | {row.s1_s2_bidirectional_sum:.4f} | {row.s3_to_s1_or_s2:.4f} | "
            f"{row.s4_self:.4f} | {row.s4_to_s1_or_s2:.4f} |"
        )
    lines.append("")
    lines.append("## Matrix Differences")
    lines.append("| matrix A | matrix B | max abs diff | mean abs diff | Frobenius |")
    lines.append("|---|---|---:|---:|---:|")
    for row in comparison_df.itertuples(index=False):
        lines.append(
            f"| {row.matrix_a} | {row.matrix_b} | {row.max_abs_diff:.4f} | {row.mean_abs_diff:.4f} | {row.frobenius_norm:.4f} |"
        )
    lines.append("")
    for name, prob in matrices.items():
        lines.append(f"## {name} Transition Probabilities")
        lines.append("| from | to_0 | to_1 | to_2 | to_3 | to_4 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for i in range(N_STATES):
            lines.append(
                f"| S{i} | {prob[i,0]:.4f} | {prob[i,1]:.4f} | {prob[i,2]:.4f} | {prob[i,3]:.4f} | {prob[i,4]:.4f} |"
            )
        lines.append("")
    lines.append("## Gate")
    if (
        relation_df["s0_top_is_s1"].all()
        and relation_df["s1_s2_bidirectional_sum"].min() > 0.60
        and relation_df["s3_to_s1_or_s2"].min() > 0.60
        and relation_df["s4_to_s1_or_s2"].min() > 0.35
        and relation_df["s4_self"].min() > 0.25
    ):
        lines.append("Observed player-state dynamics are robust to the Jan-31 update window and to unequal player activity.")
    else:
        lines.append("The transition structure changes enough across controls that Step 4 should remain provisional.")
    (OUT_DIR / "step4_2_transition_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_assignments()

    baseline_edges = build_edges(df)
    exclude_window_df = df[
        ~df["login_time_utc"].dt.normalize().between(UPDATE_WINDOW_START, UPDATE_WINDOW_END)
    ].copy()
    exclude_edges = build_edges(exclude_window_df)

    baseline_counts = count_matrix(baseline_edges)
    baseline_prob = row_normalize(baseline_counts)
    exclude_counts = count_matrix(exclude_edges)
    exclude_prob = row_normalize(exclude_counts)
    balanced_prob, player_balanced_contrib = player_balanced_matrix(baseline_edges)

    probs = {
        "main": baseline_prob,
        "exclude_update_window": exclude_prob,
        "player_balanced": balanced_prob,
    }
    counts = {
        "main": baseline_counts,
        "exclude_update_window": exclude_counts,
    }

    summary_rows = [
        {"matrix": "main", "sessions": len(df), "players": df["pid"].nunique(), "transitions": len(baseline_edges)},
        {
            "matrix": "exclude_update_window",
            "sessions": len(exclude_window_df),
            "players": exclude_window_df["pid"].nunique(),
            "transitions": len(exclude_edges),
        },
        {
            "matrix": "player_balanced",
            "sessions": len(df),
            "players": player_balanced_contrib["pid"].nunique(),
            "transitions": len(baseline_edges),
        },
    ]

    all_transition_features = []
    relation_rows = []
    for name, prob in probs.items():
        features, relation = transition_features(name, prob)
        all_transition_features.append(features)
        relation_rows.append(relation)
        save_matrix(name, prob, "transition_probabilities_5x5")
    for name, mat in counts.items():
        save_matrix(name, mat, "transition_counts_5x5")

    relation_df = pd.DataFrame(relation_rows)
    feature_df = pd.concat(all_transition_features, ignore_index=True)
    comparison_df = compare_matrices(probs)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "transition_matrix_summary.csv", index=False, encoding="utf-8-sig")
    relation_df.to_csv(OUT_DIR / "transition_key_relations.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(OUT_DIR / "transition_state_outgoing_features.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUT_DIR / "transition_matrix_differences.csv", index=False, encoding="utf-8-sig")
    player_balanced_contrib.to_csv(OUT_DIR / "player_balanced_transition_contributors.csv", index=False, encoding="utf-8-sig")
    pd.concat([matrix_long(name, mat) for name, mat in probs.items()], ignore_index=True).to_csv(
        OUT_DIR / "transition_probabilities_long.csv", index=False, encoding="utf-8-sig"
    )
    write_report(summary_rows, relation_df, comparison_df, probs)
    print(
        f"main_transitions={len(baseline_edges)} exclude_window_transitions={len(exclude_edges)} "
        f"player_balanced_players={player_balanced_contrib['pid'].nunique()} out={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
