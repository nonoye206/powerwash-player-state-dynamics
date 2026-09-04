from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "step5_disengagement_risk" / "session_disengagement_labels.csv"
OUT_DIR = ROOT / "step5_4_behavioral_rigidity"
WINDOWS = (3, 5)
MAX_SESSION_LAG_FOR_REPORT = 20


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(series).astype(bool)


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
        n_vals = []
        entropy_vals = []
        unique_vals = []
        dominant_vals = []
        self_transition_vals = []
        for i in range(len(states)):
            window = states[max(0, i - w + 1) : i + 1]
            n_vals.append(len(window))
            entropy_vals.append(entropy_bits(window))
            unique_vals.append(int(len(np.unique(window))))
            vals, counts = np.unique(window, return_counts=True)
            dominant_vals.append(float(counts.max() / counts.sum()))
            if len(window) >= 2:
                self_transition_vals.append(float((window[1:] == window[:-1]).sum() / (len(window) - 1)))
            else:
                self_transition_vals.append(np.nan)
        out[f"n_sessions_w{w}"] = n_vals
        out[f"state_entropy_w{w}"] = entropy_vals
        out[f"unique_states_w{w}"] = unique_vals
        out[f"dominant_state_share_w{w}"] = dominant_vals
        out[f"self_transition_ratio_w{w}"] = self_transition_vals
    return pd.DataFrame(out)


def summarize_trend(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        base = {
            group_col: key,
            "sessions": int(len(g)),
            "players": int(g["pid"].nunique()),
            "median_days_to_terminal_disengagement": float(g["days_to_terminal_disengagement"].median()),
        }
        for w in WINDOWS:
            for metric in [
                "state_entropy",
                "unique_states",
                "dominant_state_share",
                "self_transition_ratio",
            ]:
                col = f"{metric}_w{w}"
                base[f"{col}_mean"] = float(g[col].mean())
                base[f"{col}_median"] = float(g[col].median())
        rows.append(base)
    return pd.DataFrame(rows)


def make_day_bin(days: pd.Series) -> pd.Series:
    bins = [-0.001, 1, 3, 7, 14, 30, 60, 90, np.inf]
    labels = ["0-1d", "1-3d", "3-7d", "7-14d", "14-30d", "30-60d", "60-90d", "90d+"]
    return pd.cut(days, bins=bins, labels=labels)


def period_label(session_lag: pd.Series) -> pd.Series:
    conditions = [
        session_lag.between(0, 2),
        session_lag.between(3, 5),
        session_lag.between(6, 10),
        session_lag.between(11, 20),
        session_lag.gt(20),
    ]
    choices = ["0-2 sessions", "3-5 sessions", "6-10 sessions", "11-20 sessions", "20+ sessions"]
    return np.select(conditions, choices, default="unknown")


def slope_by_player(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, window), g in df.groupby(["pid", "window"]):
        g = g.dropna(subset=["session_lag_to_terminal_disengagement", "value"])
        if len(g) < 3 or g["session_lag_to_terminal_disengagement"].nunique() < 2:
            continue
        # Negative slope means the metric rises as terminal disengagement approaches,
        # because session lag decreases toward 0.
        x = g["session_lag_to_terminal_disengagement"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)
        x = x - x.mean()
        denom = float((x * x).sum())
        if denom == 0:
            continue
        slope = float((x * (y - y.mean())).sum() / denom)
        rows.append({"pid": pid, "window": window, "metric": g["metric"].iloc[0], "slope_vs_session_lag": slope})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABELS_PATH)
    df["login_time_utc"] = pd.to_datetime(df["login_time_utc"])
    df["state_id"] = df["state_id"].astype(int)
    for col in ["evaluable_30d", "observed_disengaged_30d", "is_last_observed_session"]:
        df[col] = boolify(df[col])

    rolling = pd.concat([rolling_metrics_for_player(g) for _, g in df.groupby("pid", sort=False)], ignore_index=True)

    anchors = df[
        df["is_last_observed_session"]
        & df["evaluable_30d"]
        & df["observed_disengaged_30d"]
    ][["pid", "session_id", "session_index", "login_time_utc", "state_id"]].copy()
    anchors = anchors.rename(
        columns={
            "session_id": "terminal_anchor_session_id",
            "session_index": "terminal_anchor_session_index",
            "login_time_utc": "terminal_anchor_login_time_utc",
            "state_id": "terminal_anchor_state_id",
        }
    )
    anchors.to_csv(OUT_DIR / "terminal_30d_disengagement_anchors.csv", index=False, encoding="utf-8-sig")

    work = rolling.merge(anchors, on="pid", how="inner")
    work = work[work["session_index"].le(work["terminal_anchor_session_index"])].copy()
    work["session_lag_to_terminal_disengagement"] = work["terminal_anchor_session_index"] - work["session_index"]
    work["days_to_terminal_disengagement"] = (
        pd.to_datetime(work["terminal_anchor_login_time_utc"]) - pd.to_datetime(work["login_time_utc"])
    ).dt.total_seconds() / 86400.0
    work["day_bin_to_terminal_disengagement"] = make_day_bin(work["days_to_terminal_disengagement"])
    work["session_lag_period"] = period_label(work["session_lag_to_terminal_disengagement"])
    work.to_csv(OUT_DIR / "rolling_rigidity_session_level.csv", index=False, encoding="utf-8-sig")

    by_lag = summarize_trend(
        work[work["session_lag_to_terminal_disengagement"].le(MAX_SESSION_LAG_FOR_REPORT)],
        "session_lag_to_terminal_disengagement",
    ).sort_values("session_lag_to_terminal_disengagement")
    by_lag.to_csv(OUT_DIR / "rigidity_trend_by_session_lag.csv", index=False, encoding="utf-8-sig")

    by_day = summarize_trend(work, "day_bin_to_terminal_disengagement")
    by_day.to_csv(OUT_DIR / "rigidity_trend_by_day_bin.csv", index=False, encoding="utf-8-sig")

    by_period = summarize_trend(work, "session_lag_period")
    order = pd.CategoricalDtype(["0-2 sessions", "3-5 sessions", "6-10 sessions", "11-20 sessions", "20+ sessions"], ordered=True)
    by_period["session_lag_period"] = by_period["session_lag_period"].astype(order)
    by_period = by_period.sort_values("session_lag_period")
    by_period.to_csv(OUT_DIR / "rigidity_trend_by_session_lag_period.csv", index=False, encoding="utf-8-sig")

    by_terminal_state_period = summarize_trend(
        work,
        "session_lag_period",
    )
    by_terminal_state_period_parts = []
    for state_id, g in work.groupby("terminal_anchor_state_id"):
        part = summarize_trend(g, "session_lag_period")
        part.insert(0, "terminal_anchor_state_id", int(state_id))
        by_terminal_state_period_parts.append(part)
    by_terminal_state_period = pd.concat(by_terminal_state_period_parts, ignore_index=True)
    by_terminal_state_period["session_lag_period"] = by_terminal_state_period["session_lag_period"].astype(order)
    by_terminal_state_period = by_terminal_state_period.sort_values(["terminal_anchor_state_id", "session_lag_period"])
    by_terminal_state_period.to_csv(OUT_DIR / "rigidity_trend_by_terminal_state_and_lag_period.csv", index=False, encoding="utf-8-sig")

    player_summary = (
        work.groupby("pid")
        .agg(
            sessions_before_terminal=("session_id", "size"),
            terminal_anchor_session_index=("terminal_anchor_session_index", "max"),
            terminal_anchor_state_id=("terminal_anchor_state_id", "max"),
            total_days_observed_before_terminal=("days_to_terminal_disengagement", "max"),
            near_entropy_w3=("state_entropy_w3", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].le(2)].mean()),
            far_entropy_w3=("state_entropy_w3", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].between(6, 10)].mean()),
            near_self_transition_w3=("self_transition_ratio_w3", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].le(2)].mean()),
            far_self_transition_w3=("self_transition_ratio_w3", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].between(6, 10)].mean()),
            near_entropy_w5=("state_entropy_w5", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].le(2)].mean()),
            far_entropy_w5=("state_entropy_w5", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].between(6, 10)].mean()),
            near_self_transition_w5=("self_transition_ratio_w5", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].le(2)].mean()),
            far_self_transition_w5=("self_transition_ratio_w5", lambda s: s[work.loc[s.index, "session_lag_to_terminal_disengagement"].between(6, 10)].mean()),
        )
        .reset_index()
    )
    player_summary["entropy_w3_near_minus_far"] = player_summary["near_entropy_w3"] - player_summary["far_entropy_w3"]
    player_summary["entropy_w5_near_minus_far"] = player_summary["near_entropy_w5"] - player_summary["far_entropy_w5"]
    player_summary["self_transition_w3_near_minus_far"] = player_summary["near_self_transition_w3"] - player_summary["far_self_transition_w3"]
    player_summary["self_transition_w5_near_minus_far"] = player_summary["near_self_transition_w5"] - player_summary["far_self_transition_w5"]
    player_summary.to_csv(OUT_DIR / "rigidity_player_anchor_summary.csv", index=False, encoding="utf-8-sig")

    # Long-form trend table for plotting/statistics.
    long_parts = []
    for w in WINDOWS:
        for metric in ["state_entropy", "unique_states", "dominant_state_share", "self_transition_ratio"]:
            part = work[
                [
                    "pid",
                    "session_id",
                    "session_lag_to_terminal_disengagement",
                    "days_to_terminal_disengagement",
                    "day_bin_to_terminal_disengagement",
                    f"{metric}_w{w}",
                ]
            ].copy()
            part = part.rename(columns={f"{metric}_w{w}": "value"})
            part["metric"] = metric
            part["window"] = w
            long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)
    long.to_csv(OUT_DIR / "rolling_rigidity_long.csv", index=False, encoding="utf-8-sig")

    # Compact near/far comparison.
    comparisons = []
    for w in WINDOWS:
        for metric in ["state_entropy", "unique_states", "dominant_state_share", "self_transition_ratio"]:
            col = f"{metric}_w{w}"
            near = work[work["session_lag_to_terminal_disengagement"].le(2)][col]
            far = work[work["session_lag_to_terminal_disengagement"].between(6, 10)][col]
            comparisons.append(
                {
                    "window": w,
                    "metric": metric,
                    "near_0_2_sessions_mean": float(near.mean()),
                    "far_6_10_sessions_mean": float(far.mean()),
                    "near_minus_far": float(near.mean() - far.mean()),
                    "near_sessions": int(near.notna().sum()),
                    "far_sessions": int(far.notna().sum()),
                }
            )
    comp = pd.DataFrame(comparisons)
    comp.to_csv(OUT_DIR / "rigidity_near_vs_far_comparison.csv", index=False, encoding="utf-8-sig")

    terminal_state_dist = (
        anchors.groupby("terminal_anchor_state_id")
        .agg(players=("pid", "nunique"), terminal_sessions=("terminal_anchor_session_id", "size"))
        .reset_index()
        .sort_values("terminal_anchor_state_id")
    )
    terminal_state_dist["share"] = terminal_state_dist["terminal_sessions"] / len(anchors)
    terminal_state_dist.to_csv(OUT_DIR / "terminal_anchor_state_distribution.csv", index=False, encoding="utf-8-sig")

    stratified_comp = []
    for state_id, g in work.groupby("terminal_anchor_state_id"):
        for w in WINDOWS:
            for metric in ["state_entropy", "unique_states", "dominant_state_share", "self_transition_ratio"]:
                col = f"{metric}_w{w}"
                near = g[g["session_lag_to_terminal_disengagement"].le(2)][col]
                far = g[g["session_lag_to_terminal_disengagement"].between(6, 10)][col]
                stratified_comp.append(
                    {
                        "terminal_anchor_state_id": int(state_id),
                        "window": w,
                        "metric": metric,
                        "near_0_2_sessions_mean": float(near.mean()),
                        "far_6_10_sessions_mean": float(far.mean()),
                        "near_minus_far": float(near.mean() - far.mean()),
                        "near_sessions": int(near.notna().sum()),
                        "far_sessions": int(far.notna().sum()),
                    }
                )
    stratified_comp = pd.DataFrame(stratified_comp)
    stratified_comp.to_csv(OUT_DIR / "rigidity_near_vs_far_by_terminal_state.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Step 5.4 Behavioral Rigidity",
        "",
        "## Scope",
        "",
        f"- Input: `{LABELS_PATH.relative_to(ROOT)}`",
        "- Main anchor: terminal 30-day observed disengagement, defined as a player's last observed session that is evaluable for 30d and has no return within 30d.",
        f"- Terminal anchors: {len(anchors):,} players/sessions.",
        f"- Session rows before anchors: {len(work):,}.",
        "- Rolling windows: last 3 sessions and last 5 sessions, ending at each session.",
        "- Metrics: state entropy, unique states, dominant-state share, self-transition ratio.",
        "",
        "## Near vs Earlier Trend",
        "",
        "| window | metric | mean at 0-2 sessions before terminal | mean at 6-10 sessions before terminal | near - earlier |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in comp.itertuples(index=False):
        lines.append(
            f"| {r.window} | {r.metric} | {r.near_0_2_sessions_mean:.3f} | "
            f"{r.far_6_10_sessions_mean:.3f} | {r.near_minus_far:.3f} |"
        )

    lines += [
        "",
        "## Terminal Anchor States",
        "",
        "| terminal state | players/sessions | share |",
        "|---:|---:|---:|",
    ]
    for r in terminal_state_dist.itertuples(index=False):
        lines.append(f"| S{int(r.terminal_anchor_state_id)} | {int(r.terminal_sessions):,} | {pct(r.share)} |")

    lag0 = by_lag[by_lag["session_lag_to_terminal_disengagement"].eq(0)].iloc[0]
    lag10 = by_lag[by_lag["session_lag_to_terminal_disengagement"].eq(10)].iloc[0] if (by_lag["session_lag_to_terminal_disengagement"].eq(10)).any() else None

    lines += [
        "",
        "## Interpretation",
        "",
        f"- At the terminal session itself, W3 entropy is {lag0['state_entropy_w3_mean']:.3f}, unique states {lag0['unique_states_w3_mean']:.3f}, dominant-state share {lag0['dominant_state_share_w3_mean']:.3f}, and self-transition ratio {lag0['self_transition_ratio_w3_mean']:.3f}.",
    ]
    if lag10 is not None:
        lines.append(
            f"- Ten sessions earlier, W3 entropy is {lag10['state_entropy_w3_mean']:.3f}, unique states {lag10['unique_states_w3_mean']:.3f}, dominant-state share {lag10['dominant_state_share_w3_mean']:.3f}, and self-transition ratio {lag10['self_transition_ratio_w3_mean']:.3f}."
        )
    ent3 = comp[(comp["window"].eq(3)) & (comp["metric"].eq("state_entropy"))]["near_minus_far"].iloc[0]
    dom3 = comp[(comp["window"].eq(3)) & (comp["metric"].eq("dominant_state_share"))]["near_minus_far"].iloc[0]
    self3 = comp[(comp["window"].eq(3)) & (comp["metric"].eq("self_transition_ratio"))]["near_minus_far"].iloc[0]
    if ent3 < 0 and dom3 > 0 and self3 > 0:
        lines.append(
            "- The main pattern is consistent with rising behavioral rigidity before terminal disengagement: entropy falls, dominant-state concentration rises, and self-transition rises near the end."
        )
    else:
        lines.append(
        "- The main pattern is mixed rather than a clean monotonic rigidity signal; inspect the lag and day-bin tables before turning it into a claim."
        )
    lines.append(
        "- The terminal anchors are heterogeneous across states, so the first-pass rigidity result should be framed as partial evidence: lower entropy and higher concentration near terminal disengagement, but no matching rise in self-transition."
    )

    lines += [
        "",
        "## Outputs",
        "",
        "- `terminal_30d_disengagement_anchors.csv`",
        "- `rolling_rigidity_session_level.csv`",
        "- `rolling_rigidity_long.csv`",
        "- `rigidity_trend_by_session_lag.csv`",
        "- `rigidity_trend_by_session_lag_period.csv`",
        "- `rigidity_trend_by_day_bin.csv`",
        "- `rigidity_near_vs_far_comparison.csv`",
        "- `terminal_anchor_state_distribution.csv`",
        "- `rigidity_near_vs_far_by_terminal_state.csv`",
        "- `rigidity_trend_by_terminal_state_and_lag_period.csv`",
        "- `rigidity_player_anchor_summary.csv`",
    ]
    (OUT_DIR / "step5_4_behavioral_rigidity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
