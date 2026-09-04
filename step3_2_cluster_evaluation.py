import json
from pathlib import Path

import numpy as np
import pandas as pd


PCA_DIR = Path("step3_pca")
OUT_DIR = Path("step3_2_cluster_evaluation")
MAIN_INPUT = PCA_DIR / "state_discovery_input_v1_standardized.csv"
PCA_INPUT = PCA_DIR / "pca_session_scores.csv"

MAIN_FEATURES = [
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

RANDOM_SEED = 42
K_RANGE = range(2, 9)
KMEANS_RESTARTS = 8
KMEANS_MAX_ITER = 120
GMM_RESTARTS = 5
GMM_MAX_ITER = 120
SILHOUETTE_SAMPLE = 6000


def load_spaces():
    main = pd.read_csv(MAIN_INPUT, usecols=["session_id", "pid", "session_index", *MAIN_FEATURES])
    pca = pd.read_csv(PCA_INPUT, usecols=["session_id", "pid", "session_index", *PC_FEATURES])
    if not main[["session_id", "pid", "session_index"]].equals(pca[["session_id", "pid", "session_index"]]):
        raise ValueError("Main and PCA inputs are not aligned by session_id/pid/session_index.")
    return {
        "main_standardized": (main[["session_id", "pid", "session_index"]], main[MAIN_FEATURES].to_numpy(dtype=np.float64), MAIN_FEATURES),
        "pc1_pc5": (pca[["session_id", "pid", "session_index"]], pca[PC_FEATURES].to_numpy(dtype=np.float64), PC_FEATURES),
    }


def squared_distances(x, centers):
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1)
    return np.maximum(x_norm + c_norm - 2 * x @ centers.T, 0.0)


def init_kmeans_pp(x, k, rng):
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    first = rng.integers(n)
    centers[0] = x[first]
    closest = squared_distances(x, centers[:1]).ravel()
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:
            centers[i] = x[rng.integers(n)]
        else:
            idx = np.searchsorted(np.cumsum(closest), rng.random() * total)
            centers[i] = x[min(idx, n - 1)]
        closest = np.minimum(closest, squared_distances(x, centers[i : i + 1]).ravel())
    return centers


def fit_kmeans(x, k, seed):
    best = None
    for restart in range(KMEANS_RESTARTS):
        rng = np.random.default_rng(seed + restart * 1009 + k)
        centers = init_kmeans_pp(x, k, rng)
        labels = np.full(x.shape[0], -1, dtype=np.int32)
        inertia = np.inf
        for n_iter in range(1, KMEANS_MAX_ITER + 1):
            dist = squared_distances(x, centers)
            new_labels = dist.argmin(axis=1).astype(np.int32)
            new_inertia = float(dist[np.arange(x.shape[0]), new_labels].sum())
            new_centers = np.zeros_like(centers)
            counts = np.bincount(new_labels, minlength=k)
            for j in range(k):
                if counts[j]:
                    new_centers[j] = x[new_labels == j].mean(axis=0)
                else:
                    new_centers[j] = x[rng.integers(x.shape[0])]
            if np.array_equal(new_labels, labels) or abs(inertia - new_inertia) <= 1e-6 * max(1.0, inertia):
                labels, centers, inertia = new_labels, new_centers, new_inertia
                break
            labels, centers, inertia = new_labels, new_centers, new_inertia
        candidate = {"labels": labels, "centers": centers, "inertia": inertia, "iterations": n_iter}
        if best is None or candidate["inertia"] < best["inertia"]:
            best = candidate
    return best


def logsumexp(a, axis=1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def fit_diag_gmm(x, k, seed):
    n, d = x.shape
    best = None
    global_var = x.var(axis=0) + 1e-6
    for restart in range(GMM_RESTARTS):
        km = fit_kmeans(x, k, seed + 7919 * (restart + 1))
        means = km["centers"].copy()
        labels = km["labels"]
        weights = np.bincount(labels, minlength=k).astype(np.float64) / n
        variances = np.vstack(
            [
                x[labels == j].var(axis=0) + 1e-6 if np.any(labels == j) else global_var
                for j in range(k)
            ]
        )
        prev_ll = -np.inf
        for n_iter in range(1, GMM_MAX_ITER + 1):
            log_prob = np.empty((n, k), dtype=np.float64)
            for j in range(k):
                diff = x - means[j]
                log_det = np.log(variances[j]).sum()
                quad = (diff * diff / variances[j]).sum(axis=1)
                log_prob[:, j] = np.log(weights[j] + 1e-15) - 0.5 * (d * np.log(2 * np.pi) + log_det + quad)
            log_norm = logsumexp(log_prob, axis=1)
            ll = float(log_norm.sum())
            resp = np.exp(log_prob - log_norm[:, None])
            nk = resp.sum(axis=0) + 1e-12
            weights = nk / n
            means = (resp.T @ x) / nk[:, None]
            for j in range(k):
                diff = x - means[j]
                variances[j] = (resp[:, j][:, None] * diff * diff).sum(axis=0) / nk[j] + 1e-6
            if abs(ll - prev_ll) <= 1e-5 * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        labels = resp.argmax(axis=1).astype(np.int32)
        params = (k - 1) + k * d + k * d
        bic = -2 * ll + params * np.log(n)
        aic = -2 * ll + 2 * params
        candidate = {
            "labels": labels,
            "means": means,
            "variances": variances,
            "weights": weights,
            "log_likelihood": ll,
            "bic": float(bic),
            "aic": float(aic),
            "iterations": n_iter,
        }
        if best is None or candidate["bic"] < best["bic"]:
            best = candidate
    return best


def calinski_harabasz(x, labels, centers=None):
    n, d = x.shape
    unique = np.unique(labels)
    k = len(unique)
    overall = x.mean(axis=0)
    if centers is None:
        centers = np.vstack([x[labels == c].mean(axis=0) for c in unique])
    between = 0.0
    within = 0.0
    for idx, c in enumerate(unique):
        members = x[labels == c]
        center = centers[idx]
        between += len(members) * np.sum((center - overall) ** 2)
        within += np.sum((members - center) ** 2)
    return float((between / (k - 1)) / (within / (n - k))) if within > 0 and k > 1 else np.nan


def davies_bouldin(x, labels, centers=None):
    unique = np.unique(labels)
    k = len(unique)
    if centers is None:
        centers = np.vstack([x[labels == c].mean(axis=0) for c in unique])
    scatters = np.zeros(k)
    for idx, c in enumerate(unique):
        members = x[labels == c]
        scatters[idx] = np.linalg.norm(members - centers[idx], axis=1).mean() if len(members) else 0
    center_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(center_dist, np.inf)
    ratios = (scatters[:, None] + scatters[None, :]) / center_dist
    return float(np.max(ratios, axis=1).mean())


def sampled_silhouette(x, labels, seed, sample_size=SILHOUETTE_SAMPLE, chunk=500):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    xs = x[idx]
    ls = labels[idx]
    unique = np.unique(ls)
    if len(unique) < 2:
        return np.nan, len(idx)
    result = np.empty(len(idx), dtype=np.float64)
    cluster_indices = {c: np.where(ls == c)[0] for c in unique}
    for start in range(0, len(idx), chunk):
        stop = min(start + chunk, len(idx))
        dist = np.linalg.norm(xs[start:stop, None, :] - xs[None, :, :], axis=2)
        for local_i, global_i in enumerate(range(start, stop)):
            own = ls[global_i]
            own_idx = cluster_indices[own]
            if len(own_idx) <= 1:
                a = 0.0
            else:
                a = dist[local_i, own_idx].sum() / (len(own_idx) - 1)
            b = min(dist[local_i, cluster_indices[c]].mean() for c in unique if c != own)
            denom = max(a, b)
            result[global_i] = 0.0 if denom == 0 else (b - a) / denom
    return float(result.mean()), len(idx)


def cluster_sizes(labels):
    counts = np.bincount(labels)
    n = len(labels)
    rows = []
    for c, count in enumerate(counts):
        if count:
            rows.append({"cluster": int(c), "size": int(count), "pct": float(count / n * 100)})
    return rows


def evaluate_model(x, labels, seed, centers=None):
    silhouette, sample_n = sampled_silhouette(x, labels, seed)
    return {
        "silhouette_sampled": silhouette,
        "silhouette_sample_size": sample_n,
        "calinski_harabasz": calinski_harabasz(x, labels, centers=centers),
        "davies_bouldin": davies_bouldin(x, labels, centers=centers),
    }


def run_all():
    OUT_DIR.mkdir(exist_ok=True)
    spaces = load_spaces()
    metric_rows = []
    size_rows = []

    for space_name, (ids, x, features) in spaces.items():
        print(f"space={space_name} rows={x.shape[0]} features={x.shape[1]}", flush=True)
        for k in K_RANGE:
            print(f"  KMeans k={k}", flush=True)
            km = fit_kmeans(x, k, RANDOM_SEED + k)
            km_metrics = evaluate_model(x, km["labels"], RANDOM_SEED + k, centers=km["centers"])
            metric_rows.append(
                {
                    "space": space_name,
                    "model": "kmeans",
                    "k": k,
                    **km_metrics,
                    "inertia": km["inertia"],
                    "log_likelihood": np.nan,
                    "bic": np.nan,
                    "aic": np.nan,
                    "iterations": km["iterations"],
                }
            )
            for row in cluster_sizes(km["labels"]):
                size_rows.append({"space": space_name, "model": "kmeans", "k": k, **row})
            pd.DataFrame({"session_id": ids["session_id"], "label": km["labels"]}).to_csv(
                OUT_DIR / f"labels_{space_name}_kmeans_k{k}.csv", index=False, encoding="utf-8-sig"
            )

            print(f"  GMM k={k}", flush=True)
            gmm = fit_diag_gmm(x, k, RANDOM_SEED + 1000 + k)
            gmm_centers = np.vstack([x[gmm["labels"] == c].mean(axis=0) for c in range(k)])
            gmm_metrics = evaluate_model(x, gmm["labels"], RANDOM_SEED + 1000 + k, centers=gmm_centers)
            metric_rows.append(
                {
                    "space": space_name,
                    "model": "gmm_diag",
                    "k": k,
                    **gmm_metrics,
                    "inertia": np.nan,
                    "log_likelihood": gmm["log_likelihood"],
                    "bic": gmm["bic"],
                    "aic": gmm["aic"],
                    "iterations": gmm["iterations"],
                }
            )
            for row in cluster_sizes(gmm["labels"]):
                size_rows.append({"space": space_name, "model": "gmm_diag", "k": k, **row})
            pd.DataFrame({"session_id": ids["session_id"], "label": gmm["labels"]}).to_csv(
                OUT_DIR / f"labels_{space_name}_gmm_diag_k{k}.csv", index=False, encoding="utf-8-sig"
            )

    metrics = pd.DataFrame(metric_rows)
    sizes = pd.DataFrame(size_rows)
    metrics.to_csv(OUT_DIR / "cluster_model_metrics_k2_k8.csv", index=False, encoding="utf-8-sig")
    sizes.to_csv(OUT_DIR / "cluster_size_distribution_k2_k8.csv", index=False, encoding="utf-8-sig")
    return metrics, sizes


def write_report(metrics, sizes):
    lines = ["# Step 3.2 Cluster Evaluation", ""]
    lines.append("No state names assigned in this step.")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Main space: original standardized discovery features, 12 dimensions.")
    lines.append("- Control space: PC1-PC5 PCA scores.")
    lines.append("- Models: KMeans and diagonal-covariance GMM, K=2-8.")
    lines.append(f"- Silhouette: exact on a fixed random sample of {SILHOUETTE_SAMPLE:,} sessions; CH/DB and cluster sizes use all sessions.")
    lines.append("")
    lines.append("## Metrics")
    display = metrics.copy()
    for space in ["main_standardized", "pc1_pc5"]:
        lines.append(f"### {space}")
        subset = display[display["space"].eq(space)]
        lines.append("| model | K | silhouette | CH | DB | BIC | AIC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in subset.itertuples(index=False):
            bic = "" if pd.isna(row.bic) else f"{row.bic:.1f}"
            aic = "" if pd.isna(row.aic) else f"{row.aic:.1f}"
            lines.append(
                f"| {row.model} | {row.k} | {row.silhouette_sampled:.4f} | "
                f"{row.calinski_harabasz:.1f} | {row.davies_bouldin:.4f} | {bic} | {aic} |"
            )
        lines.append("")
    lines.append("## Cluster Size Notes")
    for space in ["main_standardized", "pc1_pc5"]:
        for model in ["kmeans", "gmm_diag"]:
            subset = sizes[sizes["space"].eq(space) & sizes["model"].eq(model)]
            min_by_k = subset.groupby("k")["pct"].min()
            max_by_k = subset.groupby("k")["pct"].max()
            lines.append(
                f"- {space} / {model}: min cluster pct by K = "
                + ", ".join(f"K{k}:{min_by_k.loc[k]:.2f}%" for k in min_by_k.index)
            )
            lines.append(
                f"- {space} / {model}: max cluster pct by K = "
                + ", ".join(f"K{k}:{max_by_k.loc[k]:.2f}%" for k in max_by_k.index)
            )
    lines.append("")
    lines.append("## Selection Reminder")
    lines.append("Do not choose only by the best metric. Inspect cluster size balance and downstream interpretability before naming states.")
    (OUT_DIR / "step3_2_cluster_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    metrics, sizes = run_all()
    write_report(metrics, sizes)
    print(
        json.dumps(
            {
                "metric_rows": int(len(metrics)),
                "size_rows": int(len(sizes)),
                "outputs": str(OUT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
