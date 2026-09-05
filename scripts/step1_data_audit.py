import argparse
import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis_config import resolve_raw_zip


ZIP_PATH = Path("data.zip")
OUT_DIR = Path("step1_audit")
DATA_PREFIX = "data/"
MISSING = {"", "NA", "N/A", "NULL", "null", "None", "none", "nan", "NaN"}
CHUNKSIZE = 250_000


def parse_dt(value):
    if value is None:
        return None
    value = value.strip()
    if not value or value in MISSING:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def pct(n, d):
    return None if d == 0 else 100.0 * n / d


def safe_float(value):
    if value is None or value in MISSING:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def csv_members(zf):
    return sorted(
        name
        for name in zf.namelist()
        if name.startswith(DATA_PREFIX) and name.endswith(".csv")
    )


def audit_tables(zf, members):
    summaries = []
    column_missing_rows = []
    login_events = defaultdict(list)
    exit_events = defaultdict(list)
    pid_union = set()

    for member in members:
        table = Path(member).stem
        info = zf.getinfo(member)
        row_count = 0
        players = set()
        event_names = Counter()
        missing_by_col = Counter()
        min_time_utc = max_time_utc = None
        min_time = max_time = None
        bad_time_utc = 0
        bad_time = 0
        duplicate_keys = Counter()
        fields = []

        with zf.open(member, "r") as raw:
            chunks = pd.read_csv(
                raw,
                chunksize=CHUNKSIZE,
                dtype=str,
                keep_default_na=False,
                na_values=[],
                encoding="utf-8-sig",
            )
            for chunk in chunks:
                if not fields:
                    fields = list(chunk.columns)
                row_count += len(chunk)

                normalized = chunk.replace(list(MISSING), "")
                missing_by_col.update(normalized.eq("").sum().to_dict())

                if "pid" in chunk.columns:
                    pid_series = chunk["pid"].astype(str).str.strip()
                    valid_pid = pid_series[(pid_series != "") & (~pid_series.isin(MISSING))]
                    players.update(valid_pid.unique().tolist())
                    pid_union.update(valid_pid.unique().tolist())
                else:
                    pid_series = pd.Series([""] * len(chunk))

                if "EventName" in chunk.columns:
                    event_names.update(chunk["EventName"].astype(str).str.strip().value_counts().to_dict())

                if "Time_utc" in chunk.columns:
                    parsed = pd.to_datetime(chunk["Time_utc"], errors="coerce")
                    valid = parsed.dropna()
                    if not valid.empty:
                        cur_min = valid.min().to_pydatetime()
                        cur_max = valid.max().to_pydatetime()
                        min_time_utc = cur_min if min_time_utc is None else min(min_time_utc, cur_min)
                        max_time_utc = cur_max if max_time_utc is None else max(max_time_utc, cur_max)
                    bad_time_utc += int(parsed.isna().sum())

                if "Time" in chunk.columns:
                    parsed_local = pd.to_datetime(chunk["Time"], errors="coerce")
                    valid_local = parsed_local.dropna()
                    if not valid_local.empty:
                        cur_min = valid_local.min().to_pydatetime()
                        cur_max = valid_local.max().to_pydatetime()
                        min_time = cur_min if min_time is None else min(min_time, cur_min)
                        max_time = cur_max if max_time is None else max(max_time, cur_max)
                    bad_time += int(parsed_local.isna().sum())

                if table in {"player_logged_in", "exited_game"}:
                    event_series = (
                        chunk["EventName"].astype(str).str.strip()
                        if "EventName" in chunk.columns
                        else pd.Series([""] * len(chunk))
                    )
                    time_raw = chunk["Time_utc"].astype(str) if "Time_utc" in chunk.columns else pd.Series([""] * len(chunk))
                    key_df = pd.DataFrame({"pid": pid_series, "time": time_raw, "event": event_series})
                    duplicate_keys.update(
                        {
                            tuple(idx): int(count)
                            for idx, count in key_df.value_counts().items()
                        }
                    )
                    parsed = pd.to_datetime(time_raw, errors="coerce") if "Time_utc" in chunk.columns else pd.Series(pd.NaT, index=chunk.index)
                    valid_mask = (pid_series != "") & (~pid_series.isin(MISSING)) & parsed.notna()
                    lengths = (
                        pd.to_numeric(chunk["CurrentSessionLength"], errors="coerce")
                        if "CurrentSessionLength" in chunk.columns
                        else pd.Series([math.nan] * len(chunk))
                    )
                    for pid, t_value, raw_time, session_len in zip(
                        pid_series[valid_mask],
                        parsed[valid_mask],
                        time_raw[valid_mask],
                        lengths[valid_mask],
                    ):
                        payload = {
                            "time": t_value.to_pydatetime(),
                            "raw_time": raw_time,
                            "current_session_length": None if pd.isna(session_len) else float(session_len),
                        }
                        if table == "player_logged_in":
                            login_events[pid].append(payload)
                        else:
                            exit_events[pid].append(payload)

        duplicate_event_rows = sum(v - 1 for v in duplicate_keys.values() if v > 1)
        max_col_missing = max(missing_by_col.values(), default=0)
        summaries.append(
            {
                "table": table,
                "zip_member": member,
                "compressed_mb": round(info.compress_size / 1024 / 1024, 2),
                "uncompressed_mb": round(info.file_size / 1024 / 1024, 2),
                "rows": row_count,
                "players": len(players),
                "time_utc_min": min_time_utc.isoformat(sep=" ") if min_time_utc else "",
                "time_utc_max": max_time_utc.isoformat(sep=" ") if max_time_utc else "",
                "time_min": min_time.isoformat(sep=" ") if min_time else "",
                "time_max": max_time.isoformat(sep=" ") if max_time else "",
                "missing_pid_pct": pct(missing_by_col.get("pid", 0), row_count),
                "missing_time_utc_pct": pct(missing_by_col.get("Time_utc", 0), row_count) if "Time_utc" in fields else None,
                "bad_time_utc_rows": bad_time_utc,
                "bad_time_rows": bad_time,
                "max_column_missing_pct": pct(max_col_missing, row_count),
                "event_names": json.dumps(event_names.most_common(), ensure_ascii=False),
                "duplicate_pid_time_event_rows": duplicate_event_rows,
            }
        )

        for col, miss in missing_by_col.items():
            column_missing_rows.append(
                {
                    "table": table,
                    "column": col,
                    "missing_rows": miss,
                    "missing_pct": pct(miss, row_count),
                }
            )

    return summaries, column_missing_rows, login_events, exit_events, pid_union


def reconstruct_sessions(login_events, exit_events):
    rows = []
    sessions = []
    all_pids = sorted(set(login_events) | set(exit_events))
    total_logins = total_exits = 0
    unmatched_logins = unmatched_exits = 0
    overlapping_login_events = 0
    negative_duration = 0
    zero_duration = 0
    exact_pairs = 0
    suspicious_long = 0

    for pid in all_pids:
        logins = sorted(login_events.get(pid, []), key=lambda x: x["time"])
        exits = sorted(exit_events.get(pid, []), key=lambda x: x["time"])
        total_logins += len(logins)
        total_exits += len(exits)
        i = j = 0
        pid_sessions = []
        pid_unmatched_logins = 0
        pid_unmatched_exits = 0
        pid_overlaps = 0

        while i < len(logins) or j < len(exits):
            if i >= len(logins):
                pid_unmatched_exits += len(exits) - j
                break
            if j >= len(exits):
                pid_unmatched_logins += len(logins) - i
                break

            login = logins[i]
            exit_ = exits[j]

            if exit_["time"] < login["time"]:
                pid_unmatched_exits += 1
                j += 1
                continue

            if i + 1 < len(logins) and logins[i + 1]["time"] <= exit_["time"]:
                pid_overlaps += 1
                pid_unmatched_logins += 1
                i += 1
                continue

            duration = (exit_["time"] - login["time"]).total_seconds()
            current_session_length_minutes = exit_.get("current_session_length")
            delta_vs_reported = (
                duration / 60.0 - current_session_length_minutes
                if current_session_length_minutes is not None and not math.isnan(current_session_length_minutes)
                else None
            )
            session = {
                "pid": pid,
                "session_index": len(pid_sessions) + 1,
                "login_time_utc": login["time"].isoformat(sep=" "),
                "exit_time_utc": exit_["time"].isoformat(sep=" "),
                "duration_seconds": duration,
                "duration_minutes": duration / 60.0,
                "exit_current_session_length_minutes": current_session_length_minutes,
                "duration_minus_reported_minutes": delta_vs_reported,
            }
            pid_sessions.append(session)
            sessions.append(session)
            if duration < 0:
                negative_duration += 1
            elif duration == 0:
                zero_duration += 1
            if duration > 12 * 3600:
                suspicious_long += 1
            exact_pairs += 1
            i += 1
            j += 1

        unmatched_logins += pid_unmatched_logins
        unmatched_exits += pid_unmatched_exits
        overlapping_login_events += pid_overlaps
        rows.append(
            {
                "pid": pid,
                "logins": len(logins),
                "exits": len(exits),
                "paired_sessions": len(pid_sessions),
                "unmatched_logins": pid_unmatched_logins,
                "unmatched_exits": pid_unmatched_exits,
                "overlapping_login_events": pid_overlaps,
                "first_login_utc": logins[0]["time"].isoformat(sep=" ") if logins else "",
                "last_login_utc": logins[-1]["time"].isoformat(sep=" ") if logins else "",
                "first_exit_utc": exits[0]["time"].isoformat(sep=" ") if exits else "",
                "last_exit_utc": exits[-1]["time"].isoformat(sep=" ") if exits else "",
            }
        )

    durations = [s["duration_seconds"] for s in sessions]
    deltas = [
        abs(s["duration_minus_reported_minutes"])
        for s in sessions
        if s["duration_minus_reported_minutes"] is not None
    ]
    summary = {
        "players_with_login_or_exit": len(all_pids),
        "total_logins_with_valid_pid_time": total_logins,
        "total_exits_with_valid_pid_time": total_exits,
        "paired_sessions": exact_pairs,
        "unmatched_logins": unmatched_logins,
        "unmatched_exits": unmatched_exits,
        "overlapping_login_events": overlapping_login_events,
        "negative_duration_sessions": negative_duration,
        "zero_duration_sessions": zero_duration,
        "sessions_over_12h": suspicious_long,
        "duration_seconds_min": min(durations) if durations else None,
        "duration_seconds_p50": percentile(durations, 50),
        "duration_seconds_p95": percentile(durations, 95),
        "duration_seconds_max": max(durations) if durations else None,
        "abs_delta_vs_exit_current_session_length_minutes_p50": percentile(deltas, 50),
        "abs_delta_vs_exit_current_session_length_minutes_p95": percentile(deltas, 95),
    }
    return summary, rows, sessions


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value):
    return "" if value is None else f"{value:.2f}%"


def write_report(table_summaries, column_missing, session_summary, per_player):
    lines = []
    lines.append("# Step 1 Data Audit + Session Reconstruction")
    lines.append("")
    lines.append("Source: PowerWash Simulator: Research Edition raw telemetry")
    lines.append("")
    lines.append("## Core table audit")
    lines.append("")
    lines.append("| table | rows | players | Time_utc range | missing pid | missing Time_utc | max column missing | duplicate pid-time-event rows |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for s in table_summaries:
        lines.append(
            "| {table} | {rows} | {players} | {start} to {end} | {miss_pid} | {miss_t} | {max_miss} | {dups} |".format(
                table=s["table"],
                rows=s["rows"],
                players=s["players"],
                start=s["time_utc_min"] or "n/a",
                end=s["time_utc_max"] or "n/a",
                miss_pid=fmt_pct(s["missing_pid_pct"]),
                miss_t=fmt_pct(s["missing_time_utc_pct"]),
                max_miss=fmt_pct(s["max_column_missing_pct"]),
                dups=s["duplicate_pid_time_event_rows"],
            )
        )
    lines.append("")
    lines.append("## Session pairing audit")
    lines.append("")
    for key, value in session_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    clean_players = sum(
        1
        for r in per_player
        if r["unmatched_logins"] == 0
        and r["unmatched_exits"] == 0
        and r["overlapping_login_events"] == 0
    )
    lines.append(
        f"Players with fully clean login/exit pairing: {clean_players}/{len(per_player)} "
        f"({100 * clean_players / len(per_player):.2f}% if denominator > 0)."
    )
    lines.append("")
    lines.append("## Preliminary judgment")
    if (
        session_summary["unmatched_logins"] == 0
        and session_summary["unmatched_exits"] == 0
        and session_summary["overlapping_login_events"] == 0
        and session_summary["negative_duration_sessions"] == 0
    ):
        lines.append(
            "`player_logged_in -> exited_game` can be used as a direct one-to-one session definition."
        )
    else:
        lines.append(
            "`player_logged_in -> exited_game` is not perfectly one-to-one. Use the greedy within-player chronological pairing as a reconstruction baseline, and explicitly flag/drop unmatched or overlapping cases depending on downstream analysis."
        )
    lines.append("")
    lines.append("Detailed CSV outputs:")
    lines.append("- `table_audit.csv`")
    lines.append("- `column_missingness.csv`")
    lines.append("- `session_pairing_by_player.csv`")
    lines.append("- `reconstructed_sessions.csv`")
    (OUT_DIR / "step1_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw telemetry tables and reconstruct sessions.")
    parser.add_argument(
        "--raw-zip",
        default=None,
        help="Path to the raw telemetry zip archive. Defaults to PWS_RAW_ZIP or ./data.zip.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for Step 1 audit artifacts.",
    )
    return parser.parse_args()


def main(raw_zip: str | None = None, out_dir: str | None = None):
    global ZIP_PATH, OUT_DIR
    ZIP_PATH = resolve_raw_zip(raw_zip)
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = csv_members(zf)
        table_summaries, column_missing, login_events, exit_events, _ = audit_tables(zf, members)
    session_summary, per_player, sessions = reconstruct_sessions(login_events, exit_events)
    write_csv(OUT_DIR / "table_audit.csv", table_summaries)
    write_csv(OUT_DIR / "column_missingness.csv", column_missing)
    write_csv(OUT_DIR / "session_pairing_by_player.csv", per_player)
    write_csv(OUT_DIR / "reconstructed_sessions.csv", sessions)
    (OUT_DIR / "session_summary.json").write_text(
        json.dumps(session_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(table_summaries, column_missing, session_summary, per_player)
    print(json.dumps(session_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(raw_zip=args.raw_zip, out_dir=args.out_dir)
