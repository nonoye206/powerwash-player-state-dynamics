import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("step2_v1/session_features_v1.csv")
INPUT_PATH = Path("step2_v1/feature_diagnostics/state_discovery_input_v1.csv")
OUT_DIR = Path("step3_pca")

ID_COLS = ["session_id", "pid", "session_index"]
PCA_FEATURES = [
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


def read_data():
    raw = pd.read_csv(RAW_PATH)
    inp = pd.read_csv(INPUT_PATH)
    return raw, inp


def validate_input(raw, inp):
    checks = {}
    expected_rows = 105_818
    checks["row_count"] = int(len(inp))
    checks["row_count_expected"] = expected_rows
    checks["row_count_matches_expected"] = bool(len(inp) == expected_rows)
    checks["session_id_unique"] = bool(inp["session_id"].is_unique)
    checks["session_id_matches_raw"] = bool(inp["session_id"].equals(raw["session_id"]))
    checks["pid_matches_raw"] = bool(inp["pid"].equals(raw["pid"]))
    checks["session_index_matches_raw"] = bool(inp["session_index"].equals(raw["session_index"]))
    checks["nan_cells"] = int(inp[PCA_FEATURES].isna().sum().sum())
    checks["inf_cells"] = int(np.isinf(inp[PCA_FEATURES].to_numpy(dtype=float)).sum())

    transformations = []

    def add_transform_check(name, actual, expected, tolerance=1e-10):
        diff = np.nanmax(np.abs(actual.to_numpy(dtype=float) - expected.to_numpy(dtype=float)))
        transformations.append(
            {
                "check": name,
                "max_abs_diff": float(diff),
                "passed": bool(diff <= tolerance),
            }
        )

    add_transform_check("first_session equals missing raw gap", inp["first_session"], raw["days_since_previous_session"].isna().astype(int))
    add_transform_check(
        "no_started_or_resumed_job equals jobs_started + jobs_resumed == 0",
        inp["no_started_or_resumed_job"],
        (raw["jobs_started"].fillna(0) + raw["jobs_resumed"].fillna(0)).eq(0).astype(int),
    )
    add_transform_check(
        "log_duration_minutes equals log1p(P99-winsorized duration)",
        inp["log_duration_minutes"],
        np.log1p(raw["duration_minutes"].clip(lower=0, upper=raw["duration_minutes"].quantile(0.99)).fillna(0)),
    )
    add_transform_check(
        "log_gap_days equals log1p(P99-winsorized gap, first sessions filled 0)",
        inp["log_gap_days"],
        np.log1p(
            raw["days_since_previous_session"]
            .clip(lower=0, upper=raw["days_since_previous_session"].quantile(0.99))
            .fillna(0)
        ),
    )
    for src, dst in [
        ("jobs_started", "log_jobs_started"),
        ("jobs_resumed", "log_jobs_resumed"),
        ("tasks_completed", "log_tasks_completed"),
    ]:
        add_transform_check(
            f"{dst} equals log1p(P99-winsorized {src})",
            inp[dst],
            np.log1p(raw[src].clip(lower=0, upper=raw[src].quantile(0.99)).fillna(0)),
        )
    add_transform_check("purchase_any equals items_purchased > 0", inp["purchase_any"], raw["items_purchased"].fillna(0).gt(0).astype(int))
    add_transform_check(
        "job_completion_ratio_capped equals clipped raw completion ratio",
        inp["job_completion_ratio_capped"],
        raw["job_completion_ratio"].clip(lower=0, upper=1).fillna(0),
    )
    add_transform_check(
        "job_exit_ratio_capped equals clipped raw exit ratio",
        inp["job_exit_ratio_capped"],
        raw["job_exit_ratio"].clip(lower=0, upper=1).fillna(0),
    )
    add_transform_check(
        "progression_gain equals P99-winsorized raw progression gain",
        inp["progression_gain"],
        raw["progression_gain"].clip(lower=0, upper=raw["progression_gain"].quantile(0.99)).fillna(0),
    )
    add_transform_check("mode_diversity equals raw mode diversity", inp["mode_diversity"], raw["mode_diversity"].fillna(0))

    bounds = []
    for col in PCA_FEATURES:
        s = inp[col]
        bounds.append(
            {
                "feature": col,
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std_sample": float(s.std(ddof=1)),
                "zero_pct": float(s.eq(0).mean() * 100),
            }
        )

    checks["all_transform_checks_passed"] = bool(all(row["passed"] for row in transformations))
    return checks, pd.DataFrame(transformations), pd.DataFrame(bounds)


def standardize(inp):
    x = inp[PCA_FEATURES].astype(float)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=1)
    z = (x - means) / stds
    stats = pd.DataFrame(
        {
            "feature": PCA_FEATURES,
            "raw_mean": means.values,
            "raw_std_sample": stds.values,
            "standardized_mean": z.mean(axis=0).values,
            "standardized_std_sample": z.std(axis=0, ddof=1).values,
        }
    )
    z_out = pd.concat([inp[ID_COLS], z], axis=1)
    return z.to_numpy(dtype=float), z_out, stats


def run_pca(z):
    n = z.shape[0]
    _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    eigenvalues = (singular_values ** 2) / (n - 1)
    explained = eigenvalues / eigenvalues.sum()
    scores = z @ vt.T
    return scores, vt, eigenvalues, explained


def pca_tables(inp, scores, vt, eigenvalues, explained):
    pc_cols = [f"PC{i}" for i in range(1, vt.shape[0] + 1)]
    explained_df = pd.DataFrame(
        {
            "pc": pc_cols,
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": np.cumsum(explained),
        }
    )
    loadings = pd.DataFrame(vt.T, index=PCA_FEATURES, columns=pc_cols).reset_index(names="feature")
    scores_df = pd.concat(
        [
            inp[ID_COLS].reset_index(drop=True),
            pd.DataFrame(scores[:, :6], columns=pc_cols[:6]),
        ],
        axis=1,
    )
    return explained_df, loadings, scores_df


def pc_interpretation(loadings, explained_df):
    rows = []
    for pc in ["PC1", "PC2", "PC3", "PC4", "PC5"]:
        ranked = loadings[["feature", pc]].copy()
        ranked["abs_loading"] = ranked[pc].abs()
        top_pos = ranked.sort_values(pc, ascending=False).head(4)
        top_neg = ranked.sort_values(pc, ascending=True).head(4)
        rows.append(
            {
                "pc": pc,
                "explained_variance_ratio": float(explained_df.loc[explained_df["pc"].eq(pc), "explained_variance_ratio"].iloc[0]),
                "top_positive_loadings": "; ".join(f"{r.feature}={getattr(r, pc):.3f}" for r in top_pos.itertuples(index=False)),
                "top_negative_loadings": "; ".join(f"{r.feature}={getattr(r, pc):.3f}" for r in top_neg.itertuples(index=False)),
            }
        )
    return pd.DataFrame(rows)


def svg_scatter(scores_df, x_col, y_col, out_path, title, n_sample=20000):
    plot = scores_df[[x_col, y_col]].copy()
    if len(plot) > n_sample:
        plot = plot.sample(n_sample, random_state=42)
    x = plot[x_col].to_numpy()
    y = plot[y_col].to_numpy()
    width, height = 960, 720
    ml, mr, mt, mb = 80, 35, 60, 70
    x_lo, x_hi = np.quantile(x, [0.005, 0.995])
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    if x_lo == x_hi:
        x_hi = x_lo + 1
    if y_lo == y_hi:
        y_hi = y_lo + 1

    def sx(v):
        return ml + (np.clip(v, x_lo, x_hi) - x_lo) / (x_hi - x_lo) * (width - ml - mr)

    def sy(v):
        return height - mb - (np.clip(v, y_lo, y_hi) - y_lo) / (y_hi - y_lo) * (height - mt - mb)

    points = "\n".join(
        f'<circle cx="{sx(a):.2f}" cy="{sy(b):.2f}" r="1.25" fill="#2f6f9f" fill-opacity="0.16" />'
        for a, b in zip(x, y)
    )
    x0, y0 = ml, height - mb
    x1, y1 = width - mr, mt
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{ml}" y="34" font-family="Arial, sans-serif" font-size="22" fill="#1f2933">{title}</text>
  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#9aa5b1"/>
  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#9aa5b1"/>
  <text x="{(ml + width - mr) / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#394b59">{x_col}</text>
  <text x="24" y="{(mt + height - mb) / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {(mt + height - mb) / 2:.1f})" font-family="Arial, sans-serif" font-size="15" fill="#394b59">{y_col}</text>
  <text x="{x0}" y="{height - 48}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{x_lo:.2f}</text>
  <text x="{x1}" y="{height - 48}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{x_hi:.2f}</text>
  <text x="{x0 - 8}" y="{y0 + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{y_lo:.2f}</text>
  <text x="{x0 - 8}" y="{y1 + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{y_hi:.2f}</text>
  <g>{points}</g>
</svg>'''
    out_path.write_text(svg, encoding="utf-8")


def svg_scree(explained_df, out_path):
    width, height = 900, 560
    ml, mr, mt, mb = 80, 30, 60, 70
    values = explained_df["explained_variance_ratio"].to_numpy()[:12] * 100
    ymax = math.ceil(values.max() / 5) * 5
    bar_w = (width - ml - mr) / len(values) * 0.72
    gap = (width - ml - mr) / len(values)
    bars = []
    labels = []
    for i, v in enumerate(values):
        x = ml + i * gap + gap * 0.14
        h = v / ymax * (height - mt - mb)
        y = height - mb - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#5b8def"/>')
        labels.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 45}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52616b">PC{i + 1}</text>')
        labels.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#394b59">{v:.1f}%</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{ml}" y="34" font-family="Arial, sans-serif" font-size="22" fill="#1f2933">PCA explained variance</text>
  <line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#9aa5b1"/>
  <line x1="{ml}" y1="{height - mb}" x2="{ml}" y2="{mt}" stroke="#9aa5b1"/>
  <text x="24" y="{(mt + height - mb) / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {(mt + height - mb) / 2:.1f})" font-family="Arial, sans-serif" font-size="15" fill="#394b59">Explained variance %</text>
  <text x="{ml - 8}" y="{height - mb + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52616b">0</text>
  <text x="{ml - 8}" y="{mt + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{ymax}</text>
  <g>{"".join(bars)}</g>
  <g>{"".join(labels)}</g>
</svg>'''
    out_path.write_text(svg, encoding="utf-8")


def write_report(validation, transform_checks, standardization, explained, interpretation):
    lines = ["# Step 3.0 and 3.1 PCA", ""]
    lines.append("## Step 3.0 Input Validation")
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Transform checks all compare `state_discovery_input_v1.csv` against `session_features_v1.csv`.")
    lines.append("")
    for row in transform_checks.itertuples(index=False):
        lines.append(f"- {row.check}: max_abs_diff={row.max_abs_diff:.3g}, passed={row.passed}")
    lines.append("")
    lines.append("## Step 3.1 Explained Variance")
    for row in explained.head(8).itertuples(index=False):
        lines.append(
            f"- {row.pc}: {row.explained_variance_ratio * 100:.2f}% "
            f"(cumulative {row.cumulative_explained_variance * 100:.2f}%)"
        )
    lines.append("")
    lines.append("## PC Interpretation")
    for row in interpretation.itertuples(index=False):
        lines.append(f"- {row.pc}: {row.explained_variance_ratio * 100:.2f}% variance.")
        lines.append(f"  Positive: {row.top_positive_loadings}")
        lines.append(f"  Negative: {row.top_negative_loadings}")
    lines.append("")
    lines.append("## Standardization Check")
    max_abs_mean = standardization["standardized_mean"].abs().max()
    max_std_delta = (standardization["standardized_std_sample"] - 1).abs().max()
    lines.append(f"- max absolute standardized mean: {max_abs_mean:.3g}")
    lines.append(f"- max standardized std deviation error: {max_std_delta:.3g}")
    lines.append("")
    lines.append("Generated files include PCA scores, loadings, explained variance, validation tables, and SVG plots.")
    (OUT_DIR / "step3_0_1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    raw, inp = read_data()
    validation, transform_checks, feature_bounds = validate_input(raw, inp)
    z, standardized_out, standardization = standardize(inp)
    scores, vt, eigenvalues, explained_values = run_pca(z)
    explained, loadings, scores_df = pca_tables(inp, scores, vt, eigenvalues, explained_values)
    interpretation = pc_interpretation(loadings, explained)

    transform_checks.to_csv(OUT_DIR / "step3_0_transform_checks.csv", index=False, encoding="utf-8-sig")
    feature_bounds.to_csv(OUT_DIR / "step3_0_input_feature_summary.csv", index=False, encoding="utf-8-sig")
    standardization.to_csv(OUT_DIR / "step3_0_standardization_stats.csv", index=False, encoding="utf-8-sig")
    standardized_out.to_csv(OUT_DIR / "state_discovery_input_v1_standardized.csv", index=False, encoding="utf-8-sig")
    explained.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(OUT_DIR / "pca_feature_loadings.csv", index=False, encoding="utf-8-sig")
    interpretation.to_csv(OUT_DIR / "pca_pc_interpretation.csv", index=False, encoding="utf-8-sig")
    scores_df.to_csv(OUT_DIR / "pca_session_scores.csv", index=False, encoding="utf-8-sig")

    svg_scree(explained, OUT_DIR / "pca_explained_variance.svg")
    svg_scatter(scores_df, "PC1", "PC2", OUT_DIR / "pca_pc1_pc2.svg", "PCA session scores: PC1 vs PC2")
    svg_scatter(scores_df, "PC1", "PC3", OUT_DIR / "pca_pc1_pc3.svg", "PCA session scores: PC1 vs PC3")

    write_report(validation, transform_checks, standardization, explained, interpretation)
    print(
        json.dumps(
            {
                "rows": int(len(inp)),
                "features": len(PCA_FEATURES),
                "pc1_explained": float(explained_values[0]),
                "pc2_explained": float(explained_values[1]),
                "pc3_explained": float(explained_values[2]),
                "pc5_cumulative": float(explained_values[:5].sum()),
                "all_input_checks_passed": bool(validation["all_transform_checks_passed"] and validation["nan_cells"] == 0 and validation["inf_cells"] == 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
