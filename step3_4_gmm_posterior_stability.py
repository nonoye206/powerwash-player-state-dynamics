import json
from pathlib import Path

import numpy as np
import pandas as pd

import step3_2_cluster_evaluation as cluster_eval


PCA_SCORES_PATH = Path("step3_pca/pca_session_scores.csv")
OUT_DIR = Path("step3_4_gmm_posterior_stability")
PC_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]
K_RANGE = range(4, 9)
SEEDS = [11, 23, 37, 41, 53]
SUBSAMPLE_SEEDS = [101, 103, 107, 109, 113]
SUBSAMPLE_FRAC = 0.80


def load_x():
    df = pd.read_csv(PCA_SCORES_PATH, usecols=["session_id", "pid", "session_index", *PC_FEATURES])
    x = df[PC_FEATURES].to_numpy(dtype=np.float64)
    return df[["session_id", "pid", "session_index"]], x


def logsumexp(a, axis=1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def gmm_posterior(x, gmm):
    n, d = x.shape
    k = len(gmm["weights"])
    log_prob = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        diff = x - gmm["means"][j]
        log_det = np.log(gmm["variances"][j]).sum()
        quad = (diff * diff / gmm["variances"][j]).sum(axis=1)
        log_prob[:, j] = np.log(gmm["weights"][j] + 1e-15) - 0.5 * (d * np.log(2 * np.pi) + log_det + quad)
    log_norm = logsumexp(log_prob, axis=1)
    resp = np.exp(log_prob - log_norm[:, None])
    labels = resp.argmax(axis=1).astype(np.int32)
    return resp, labels, log_norm


def entropy(resp):
    k = resp.shape[1]
    ent = -(resp * np.log(resp + 1e-15)).sum(axis=1)
    return ent / np.log(k)


def adjusted_rand_index(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("Label vectors must be the same length.")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n = len(a)
    contingency = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(contingency, (ai, bi), 1)

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb = comb2(contingency).sum()
    row_comb = comb2(contingency.sum(axis=1)).sum()
    col_comb = comb2(contingency.sum(axis=0)).sum()
    total_comb = comb2(n)
    expected = row_comb * col_comb / total_comb if total_comb else 0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return float((sum_comb - expected) / denom) if denom else 1.0


def posterior_summary(ids, x, k):
    gmm = cluster_eval.fit_diag_gmm(x, k, seed=4200 + k)
    resp, labels, log_norm = gmm_posterior(x, gmm)
    maxp = resp.max(axis=1)
    ent = entropy(resp)
    cluster_rows = []
    for c in range(k):
        mask = labels == c
        cluster_rows.append(
            {
                "k": k,
                "cluster": c,
                "size": int(mask.sum()),
                "pct": float(mask.mean() * 100),
                "mean_max_posterior": float(maxp[mask].mean()),
                "median_max_posterior": float(np.median(maxp[mask])),
                "p10_max_posterior": float(np.quantile(maxp[mask], 0.10)),
                "low_confidence_lt_0_60_pct": float((maxp[mask] < 0.60).mean() * 100),
                "low_confidence_lt_0_70_pct": float((maxp[mask] < 0.70).mean() * 100),
                "mean_normalized_entropy": float(ent[mask].mean()),
            }
        )
    overall = {
        "k": k,
        "bic": float(gmm["bic"]),
        "aic": float(gmm["aic"]),
        "log_likelihood": float(gmm["log_likelihood"]),
        "mean_max_posterior": float(maxp.mean()),
        "median_max_posterior": float(np.median(maxp)),
        "p10_max_posterior": float(np.quantile(maxp, 0.10)),
        "low_confidence_lt_0_60_pct": float((maxp < 0.60).mean() * 100),
        "low_confidence_lt_0_70_pct": float((maxp < 0.70).mean() * 100),
        "mean_normalized_entropy": float(ent.mean()),
        "min_cluster_pct": float(min(row["pct"] for row in cluster_rows)),
        "max_cluster_pct": float(max(row["pct"] for row in cluster_rows)),
    }
    posterior_detail = pd.concat(
        [
            ids,
            pd.DataFrame(
                {
                    "gmm_label": labels,
                    "max_posterior": maxp,
                    "normalized_entropy": ent,
                    "log_likelihood_row": log_norm,
                }
            ),
            pd.DataFrame(resp, columns=[f"posterior_cluster_{i}" for i in range(k)]),
        ],
        axis=1,
    )
    return gmm, labels, overall, pd.DataFrame(cluster_rows), posterior_detail


def seed_stability(x, k):
    labels_by_seed = []
    fit_rows = []
    for seed in SEEDS:
        gmm = cluster_eval.fit_diag_gmm(x, k, seed=seed)
        _, labels, _ = gmm_posterior(x, gmm)
        labels_by_seed.append((seed, labels))
        counts = np.bincount(labels, minlength=k)
        fit_rows.append(
            {
                "k": k,
                "seed": seed,
                "bic": float(gmm["bic"]),
                "aic": float(gmm["aic"]),
                "min_cluster_pct": float(counts.min() / len(labels) * 100),
                "max_cluster_pct": float(counts.max() / len(labels) * 100),
            }
        )
    pair_rows = []
    for i, (seed_a, labels_a) in enumerate(labels_by_seed):
        for seed_b, labels_b in labels_by_seed[i + 1 :]:
            pair_rows.append(
                {
                    "k": k,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "ari": adjusted_rand_index(labels_a, labels_b),
                }
            )
    return pd.DataFrame(fit_rows), pd.DataFrame(pair_rows)


def subsample_stability(x, reference_labels, k):
    n = len(x)
    rows = []
    for seed in SUBSAMPLE_SEEDS:
        rng = np.random.default_rng(seed + k)
        idx = np.sort(rng.choice(n, size=int(n * SUBSAMPLE_FRAC), replace=False))
        gmm = cluster_eval.fit_diag_gmm(x[idx], k, seed=seed)
        _, labels_full, _ = gmm_posterior(x, gmm)
        _, labels_sub, _ = gmm_posterior(x[idx], gmm)
        counts = np.bincount(labels_full, minlength=k)
        rows.append(
            {
                "k": k,
                "subsample_seed": seed,
                "subsample_frac": SUBSAMPLE_FRAC,
                "ari_vs_reference_full": adjusted_rand_index(reference_labels, labels_full),
                "ari_on_subsample_vs_reference": adjusted_rand_index(reference_labels[idx], labels_sub),
                "bic_subsample_fit": float(gmm["bic"]),
                "aic_subsample_fit": float(gmm["aic"]),
                "min_predicted_full_cluster_pct": float(counts.min() / n * 100),
                "max_predicted_full_cluster_pct": float(counts.max() / n * 100),
            }
        )
    return pd.DataFrame(rows)


def write_report(overall, cluster_post, seed_pairs, subsample):
    lines = ["# Step 3.4 GMM Posterior and Stability", ""]
    lines.append("Scope: PC1-PC5 diagonal-covariance GMM, K=4-8. No K is finalized here.")
    lines.append("")
    lines.append("## Posterior Quality")
    lines.append("| K | BIC | AIC | mean max posterior | median max posterior | P10 max posterior | <0.60 | <0.70 | mean entropy | min cluster % | max cluster % |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in overall.sort_values("k").itertuples(index=False):
        lines.append(
            f"| {row.k} | {row.bic:.1f} | {row.aic:.1f} | {row.mean_max_posterior:.4f} | "
            f"{row.median_max_posterior:.4f} | {row.p10_max_posterior:.4f} | "
            f"{row.low_confidence_lt_0_60_pct:.2f}% | {row.low_confidence_lt_0_70_pct:.2f}% | "
            f"{row.mean_normalized_entropy:.4f} | {row.min_cluster_pct:.2f}% | {row.max_cluster_pct:.2f}% |"
        )
    lines.append("")
    lines.append("## Seed Stability")
    seed_summary = seed_pairs.groupby("k")["ari"].agg(["mean", "min", "max"]).reset_index()
    lines.append("| K | mean ARI | min ARI | max ARI |")
    lines.append("|---:|---:|---:|---:|")
    for row in seed_summary.itertuples(index=False):
        lines.append(f"| {row.k} | {row.mean:.4f} | {row.min:.4f} | {row.max:.4f} |")
    lines.append("")
    lines.append("## 80% Subsample Stability")
    sub_summary = subsample.groupby("k")[["ari_vs_reference_full", "ari_on_subsample_vs_reference"]].agg(["mean", "min", "max"])
    lines.append("| K | full mean ARI | full min ARI | subsample mean ARI | subsample min ARI |")
    lines.append("|---:|---:|---:|---:|---:|")
    for k in sub_summary.index:
        lines.append(
            f"| {k} | {sub_summary.loc[k, ('ari_vs_reference_full', 'mean')]:.4f} | "
            f"{sub_summary.loc[k, ('ari_vs_reference_full', 'min')]:.4f} | "
            f"{sub_summary.loc[k, ('ari_on_subsample_vs_reference', 'mean')]:.4f} | "
            f"{sub_summary.loc[k, ('ari_on_subsample_vs_reference', 'min')]:.4f} |"
        )
    lines.append("")
    lines.append("## Reading Notes")
    lines.append("- High mean max posterior and low entropy indicate crisper GMM assignment.")
    lines.append("- ARI is label-invariant; values near 1 mean nearly identical partitions.")
    lines.append("- A K with good BIC but weak ARI or tiny clusters should not be finalized.")
    (OUT_DIR / "step3_4_gmm_posterior_stability_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    ids, x = load_x()
    # Keep stability runtime bounded while preserving deterministic behavior.
    cluster_eval.GMM_RESTARTS = 3
    cluster_eval.KMEANS_RESTARTS = 5
    cluster_eval.GMM_MAX_ITER = 100

    overall_rows = []
    cluster_post_rows = []
    seed_fit_rows = []
    seed_pair_rows = []
    subsample_rows = []

    for k in K_RANGE:
        print(f"posterior K={k}", flush=True)
        _, ref_labels, overall, cluster_post, posterior_detail = posterior_summary(ids, x, k)
        overall_rows.append(overall)
        cluster_post_rows.append(cluster_post)
        posterior_detail.to_csv(OUT_DIR / f"gmm_pc1_pc5_k{k}_posterior.csv", index=False, encoding="utf-8-sig")

        print(f"seed stability K={k}", flush=True)
        seed_fit, seed_pairs = seed_stability(x, k)
        seed_fit_rows.append(seed_fit)
        seed_pair_rows.append(seed_pairs)

        print(f"subsample stability K={k}", flush=True)
        subsample_rows.append(subsample_stability(x, ref_labels, k))

    overall_df = pd.DataFrame(overall_rows)
    cluster_post_df = pd.concat(cluster_post_rows, ignore_index=True)
    seed_fit_df = pd.concat(seed_fit_rows, ignore_index=True)
    seed_pair_df = pd.concat(seed_pair_rows, ignore_index=True)
    subsample_df = pd.concat(subsample_rows, ignore_index=True)

    overall_df.to_csv(OUT_DIR / "gmm_pc1_pc5_k4_k8_posterior_summary.csv", index=False, encoding="utf-8-sig")
    cluster_post_df.to_csv(OUT_DIR / "gmm_pc1_pc5_k4_k8_cluster_posterior_summary.csv", index=False, encoding="utf-8-sig")
    seed_fit_df.to_csv(OUT_DIR / "gmm_pc1_pc5_k4_k8_seed_fit_summary.csv", index=False, encoding="utf-8-sig")
    seed_pair_df.to_csv(OUT_DIR / "gmm_pc1_pc5_k4_k8_seed_stability_ari.csv", index=False, encoding="utf-8-sig")
    subsample_df.to_csv(OUT_DIR / "gmm_pc1_pc5_k4_k8_subsample_stability_ari.csv", index=False, encoding="utf-8-sig")
    write_report(overall_df, cluster_post_df, seed_pair_df, subsample_df)

    print(
        json.dumps(
            {
                "k_values": list(K_RANGE),
                "posterior_rows": int(len(overall_df)),
                "seed_pair_rows": int(len(seed_pair_df)),
                "subsample_rows": int(len(subsample_df)),
                "out": str(OUT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
