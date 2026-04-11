"""
src/q2.py — Trajectories, Optimisation & Selecting K
=====================================================

Reads  : output/q1/scored_notes.json   (produced by q1.py)
Writes : output/q2/
    cluster_labels.csv          – client_id, cluster  (1-indexed)
    q2_results.pkl              – pickled dict for q3.py
    spaghetti_K{k}.png          – spaghetti plots for K = 2,3,4,5
    plot1_t_star_distributions.png
    plot2_expected_savings.png
    plot3_optimized_vs_baseline.png

Pipeline
--------
(a) Build normalised cumulative trajectories → K-means clustering
(b) Newsvendor audit model → Q* per cluster
(c) K selection by policy distinctiveness
(d) Required plots for final K
(e) Summary table + cluster labels
(f) Case implications (written up in README / report)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCORED_PATH = Path("output/q1/scored_notes.json")
OUT         = Path("output/q2")
OUT.mkdir(parents=True, exist_ok=True)

TMAX = 12

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & BUILD TRAJECTORIES  (part a)
# ─────────────────────────────────────────────────────────────────────────────

def load_scored_notes(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_trajectories(scored_data: list[dict]) -> tuple[list[str], np.ndarray]:
    """
    Convert per-session progress scores into cumulative trajectories.

    Each client gets a 12-element cumulative array anchored at 0 at session 1:
        [0, s1, s1+s2, ..., s1+...+s11]

    Returns
    -------
    client_ids : list[str]
    X_raw      : ndarray of shape (n_clients, 12) — raw cumulative scores
    """
    client_ids, trajs = [], []
    for rec in scored_data:
        cid    = rec["client_id"]
        scores = rec["estimated_trajectory_vector"]          # length 11
        s      = (list(scores) + [0] * TMAX)[:TMAX - 1]     # ensure length 11
        cum    = np.cumsum([0] + s, dtype=float)             # length 12
        client_ids.append(cid)
        trajs.append(cum)
    return client_ids, np.array(trajs)


def normalise_trajectories(X_raw: np.ndarray) -> np.ndarray:
    """Scale each trajectory so its final value = 1 (shape-preserving)."""
    maxes = X_raw[:, -1:].copy()
    maxes[maxes == 0] = 1
    return X_raw / maxes


# ─────────────────────────────────────────────────────────────────────────────
# NEWSVENDOR MODEL  (part b)
# ─────────────────────────────────────────────────────────────────────────────

def compute_t_star(cum: np.ndarray, tmax: int = TMAX) -> int:
    """Earliest session where cumulative progress ≥ 90 % of total."""
    total = cum[-1]
    if total == 0:
        return tmax
    threshold = 0.90 * total
    for t in range(1, tmax + 1):
        if cum[t - 1] >= threshold:
            return t
    return tmax


def expected_savings(t_stars: list[int], Q: int, tmax: int = TMAX) -> float:
    """E[savings](Q) = F_c(Q) × (Tmax − Q)"""
    Fc_Q = np.mean(np.array(t_stars) <= Q)
    return float(Fc_Q * (tmax - Q))


def find_optimal_Q(t_stars: list[int], tmax: int = TMAX) -> tuple[int, float]:
    """Arg-max of E[savings](Q) over Q = 1 … Tmax."""
    savings = [expected_savings(t_stars, Q, tmax) for Q in range(1, tmax + 1)]
    Q_star  = int(np.argmax(savings)) + 1
    return Q_star, savings[Q_star - 1]


# ─────────────────────────────────────────────────────────────────────────────
# FIT & EVALUATE ACROSS K VALUES  (parts b–c)
# ─────────────────────────────────────────────────────────────────────────────

def fit_clusters(X_norm: np.ndarray, X_raw: np.ndarray,
                 K_values: list[int]) -> dict[int, dict]:
    """Fit K-means for each K and compute newsvendor policy per cluster."""
    results = {}
    for K in K_values:
        km     = KMeans(n_clusters=K, random_state=42, n_init=20)
        labels = km.fit_predict(X_norm)
        info   = {}
        for c in range(K):
            mask    = labels == c
            members = X_raw[mask]
            t_stars = [compute_t_star(members[i]) for i in range(len(members))]
            Q_star, max_sav = find_optimal_Q(t_stars)
            info[c] = {
                "size":       int(mask.sum()),
                "members_cum": members,
                "mean_cum":   members.mean(axis=0),
                "t_stars":    t_stars,
                "Q_star":     Q_star,
                "E_savings":  round(max_sav, 4),
                "mean_t_star": round(float(np.mean(t_stars)), 2),
            }
        results[K] = {"labels": labels, "clusters": info}
    return results


def select_K(results_by_K: dict[int, dict]) -> int:
    """
    Select K based on policy distinctiveness.

    Criterion: the clustering that produces the most distinct Q* values
    while keeping clusters large enough to be actionable (≥ 10 clients).
    Ties broken by maximum spread in E[savings] across clusters.
    """
    print("\n=== K Selection ===")
    best_K, best_score = 2, -1
    for K, res in results_by_K.items():
        Q_stars  = [res["clusters"][c]["Q_star"]   for c in range(K)]
        savings  = [res["clusters"][c]["E_savings"] for c in range(K)]
        sizes    = [res["clusters"][c]["size"]      for c in range(K)]
        distinct = len(set(Q_stars))
        spread   = float(np.max(savings) - np.min(savings))
        min_size = min(sizes)
        score    = distinct * spread if min_size >= 8 else 0
        print(f"  K={K}: Q*={Q_stars}  distinct={distinct}  "
              f"savings_spread={spread:.3f}  min_cluster={min_size}  score={score:.3f}")
        if score > best_score:
            best_K, best_score = K, score
    print(f"\n  → Selected K = {best_K}")
    return best_K


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
SESSIONS       = np.arange(1, TMAX + 1)


def plot_spaghetti(results_by_K: dict[int, dict]) -> None:
    """Spaghetti plots for every K evaluated (part a)."""
    for K, res in results_by_K.items():
        fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), sharey=True)
        if K == 1:
            axes = [axes]
        for c in range(K):
            ax   = axes[c]
            info = res["clusters"][c]
            for traj in info["members_cum"]:
                ax.plot(SESSIONS, traj, color=CLUSTER_COLORS[c],
                        alpha=0.22, linewidth=0.9)
            ax.plot(SESSIONS, info["mean_cum"], color="black",
                    linewidth=2.5, label="Cluster mean", zorder=5)
            ax.set_title(f"Cluster {c+1}  (n={info['size']})", fontsize=11)
            ax.set_xlabel("Session")
            if c == 0:
                ax.set_ylabel("Cumulative progress score")
            ax.legend(fontsize=8)
            ax.set_xlim(1, TMAX)
        fig.suptitle(f"Spaghetti Plots — K={K}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT / f"spaghetti_K{K}.png", dpi=140, bbox_inches="tight")
        plt.close()
    print("Spaghetti plots saved.")


def plot_t_star_distributions(clusters: dict, K: int) -> None:
    """Plot 1 — histogram of t* values per cluster."""
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), sharey=False)
    if K == 1:
        axes = [axes]
    for c in range(K):
        info = clusters[c]
        axes[c].hist(info["t_stars"], bins=range(1, TMAX + 2),
                     align="left", color=CLUSTER_COLORS[c],
                     edgecolor="white", rwidth=0.8)
        axes[c].axvline(info["Q_star"], color="black", linestyle="--",
                        linewidth=1.8, label=f"Q*={info['Q_star']}")
        axes[c].set_title(f"Cluster {c+1}  (n={info['size']})", fontsize=11)
        axes[c].set_xlabel("Stopping session (t*)")
        axes[c].set_ylabel("Count")
        axes[c].set_xticks(range(1, TMAX + 1))
        axes[c].legend(fontsize=9)
    fig.suptitle("Plot 1 — t* Distributions by Cluster", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "plot1_t_star_distributions.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Plot 1 saved.")


def plot_expected_savings(clusters: dict, K: int) -> None:
    """Plot 2 — E[savings](Q) curves per cluster with Q* markers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    Q_range = list(range(1, TMAX + 1))
    for c in range(K):
        info      = clusters[c]
        sav_curve = [expected_savings(info["t_stars"], Q) for Q in Q_range]
        ax.plot(Q_range, sav_curve, color=CLUSTER_COLORS[c],
                linewidth=2.2, label=f"Cluster {c+1}  (Q*={info['Q_star']})")
        ax.axvline(info["Q_star"], color=CLUSTER_COLORS[c],
                   linestyle="--", alpha=0.55, linewidth=1.4)
    ax.set_xlabel("Reassessment session Q", fontsize=12)
    ax.set_ylabel("E[sessions saved per child]", fontsize=12)
    ax.set_title("Plot 2 — Expected Sessions Saved vs. Q", fontsize=13, fontweight="bold")
    ax.set_xticks(Q_range)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot2_expected_savings.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Plot 2 saved.")


def plot_optimized_vs_baseline(clusters: dict, K: int) -> None:
    """Plot 3 — optimised Q* vs. mean(t*) baseline, side-by-side bars."""
    opt_sav, base_sav = [], []
    for c in range(K):
        info       = clusters[c]
        t_stars    = info["t_stars"]
        Q_base     = max(1, min(TMAX, int(round(info["mean_t_star"]))))
        base_sav.append(expected_savings(t_stars, Q_base))
        opt_sav.append(info["E_savings"])

    x     = np.arange(K)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    cols = [CLUSTER_COLORS[c] for c in range(K)]
    ax.bar(x - width / 2, opt_sav,  width, label="Optimised Q*",
           color=cols, edgecolor="black", linewidth=0.7)
    ax.bar(x + width / 2, base_sav, width, label="Baseline mean(t*)",
           color=cols, edgecolor="black", linewidth=0.7, alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c+1}" for c in range(K)])
    ax.set_ylabel("E[sessions saved per child]", fontsize=12)
    ax.set_title("Plot 3 — Optimised Q* vs. Mean(t*) Baseline",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Total savings comparison
    total_clients  = sum(clusters[c]["size"] for c in range(K))
    total_opt      = sum(opt_sav[c]  * clusters[c]["size"] for c in range(K))
    total_base     = sum(base_sav[c] * clusters[c]["size"] for c in range(K))
    diff           = total_opt - total_base
    ax.text(0.98, 0.04,
            f"Overall gain vs. baseline: {diff:+.1f} sessions\nacross {total_clients} clients",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.tight_layout()
    plt.savefig(OUT / "plot3_optimized_vs_baseline.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Plot 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE  (part e)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(clusters: dict, K: int) -> None:
    print(f"\n=== Summary Table (K={K}) ===")
    hdr = f"{'Cluster':<10} {'Size':<8} {'Q*':<6} {'E[saved/child]':<18} {'% sessions saved':<18}"
    print(hdr)
    print("-" * len(hdr))
    total_clients, total_saved = 0, 0.0
    for c in range(K):
        info    = clusters[c]
        pct     = round(info["E_savings"] / TMAX * 100, 1)
        total_clients += info["size"]
        total_saved   += info["E_savings"] * info["size"]
        print(f"{c+1:<10} {info['size']:<8} {info['Q_star']:<6} "
              f"{info['E_savings']:<18} {pct:<18}")
    total_E  = round(total_saved / total_clients, 4)
    total_pct = round(total_E / TMAX * 100, 1)
    print("-" * len(hdr))
    print(f"{'Total':<10} {total_clients:<8} {'—':<6} {total_E:<18} {total_pct:<18}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load & build trajectories ────────────────────────────────────────────
    scored     = load_scored_notes(SCORED_PATH)
    client_ids, X_raw  = build_trajectories(scored)
    X_norm     = normalise_trajectories(X_raw)
    print(f"Loaded {len(client_ids)} clients. Trajectory matrix: {X_raw.shape}")

    # ── Fit K-means for K = 2,3,4,5 ─────────────────────────────────────────
    K_values       = [2, 3, 4, 5]
    results_by_K   = fit_clusters(X_norm, X_raw, K_values)

    # ── Spaghetti plots for all K (part a) ───────────────────────────────────
    plot_spaghetti(results_by_K)

    # ── Select final K (part c) ──────────────────────────────────────────────
    FINAL_K   = select_K(results_by_K)
    labels    = results_by_K[FINAL_K]["labels"]
    clusters  = results_by_K[FINAL_K]["clusters"]

    # ── Required plots for final K (part d) ──────────────────────────────────
    plot_t_star_distributions(clusters, FINAL_K)
    plot_expected_savings(clusters, FINAL_K)
    plot_optimized_vs_baseline(clusters, FINAL_K)

    # ── Summary table (part e) ───────────────────────────────────────────────
    print_summary_table(clusters, FINAL_K)

    # ── Export cluster labels CSV ─────────────────────────────────────────────
    labels_df = pd.DataFrame({"client_id": client_ids, "cluster": labels + 1})
    labels_df.to_csv(OUT / "cluster_labels.csv", index=False)
    print(f"\nCluster labels saved → {OUT / 'cluster_labels.csv'}")

    # ── Pickle results for q3.py ──────────────────────────────────────────────
    with open(OUT / "q2_results.pkl", "wb") as f:
        pickle.dump({
            "FINAL_K":    FINAL_K,
            "clusters":   clusters,
            "labels":     labels,
            "client_ids": client_ids,
            "X_raw":      X_raw,
        }, f)
    print(f"Q2 results pickled → {OUT / 'q2_results.pkl'}")


if __name__ == "__main__":
    main()
