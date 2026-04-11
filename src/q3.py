"""
src/q3.py — Predictive Analytics: From Intake Profiles to Expected Service Demand
==================================================================================

Reads  : output/q2/cluster_labels.csv
         output/q2/q2_results.pkl
         data/client_features.csv
         data/waitlist.csv
Writes : output/q3/
    q3a_feature_distributions.png
    q3a_boxplots.png
    q3b_confusion_matrices.png
    q3b_feature_importances.png
    q3c_capacity_savings.png
    waitlist_predictions.csv

Pipeline
--------
(a) Explore intake features by trajectory cluster
(b) Train two classifiers (logistic regression + random forest), 80/20 split
(c) Apply best model to waitlist → estimate capacity demand
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
OUT = Path("output/q3")
OUT.mkdir(parents=True, exist_ok=True)

TMAX = 12


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    features  = pd.read_csv("data/client_features.csv")
    waitlist  = pd.read_csv("data/waitlist.csv")
    labels_df = pd.read_csv("output/q2/cluster_labels.csv")

    with open("output/q2/q2_results.pkl", "rb") as f:
        q2 = pickle.load(f)

    FINAL_K  = q2["FINAL_K"]
    clusters = q2["clusters"]
    return features, waitlist, labels_df, FINAL_K, clusters


def build_cluster_policy(clusters: dict, FINAL_K: int) -> dict:
    """Derive Q*, E[savings], and E[sessions delivered] per cluster (1-indexed)."""
    policy = {}
    for c in range(FINAL_K):
        info    = clusters[c]
        Q_star  = info["Q_star"]
        t_stars = info["t_stars"]
        Fc_Q    = float(np.mean(np.array(t_stars) <= Q_star))
        E_del   = Q_star + (1 - Fc_Q) * (TMAX - Q_star)
        policy[c + 1] = {
            "Q_star":      Q_star,
            "E_savings":   info["E_savings"],
            "E_delivered": round(E_del, 4),
        }
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# PART (a) — FEATURE EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────

def explore_features(hist: pd.DataFrame, FINAL_K: int) -> None:
    """Produce visual summaries of intake features broken down by cluster."""
    clusters_sorted = sorted(hist["cluster"].unique())
    colors          = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"][:FINAL_K]

    # ── Histograms ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for c in clusters_sorted:
        sub = hist[hist["cluster"] == c]
        axes[0, 0].hist(sub["age_years"],       bins=8, alpha=0.55, label=f"Cluster {c}")
        axes[0, 1].hist(sub["complexity_score"],bins=6, alpha=0.55, label=f"Cluster {c}")

    axes[0, 0].set_title("Age at Intake by Cluster");   axes[0, 0].set_xlabel("Age (years)")
    axes[0, 1].set_title("Complexity Score by Cluster");axes[0, 1].set_xlabel("Complexity score")
    for ax in axes[0]:
        ax.set_ylabel("Count"); ax.legend()

    # Gender and referral as grouped bars
    gender_ct = hist.groupby(["cluster", "gender"]).size().unstack(fill_value=0)
    gender_ct.plot(kind="bar", ax=axes[1, 0], color=["#FF9800","#2196F3","#9C27B0"])
    axes[1, 0].set_title("Gender by Cluster")
    axes[1, 0].set_xlabel("Cluster"); axes[1, 0].tick_params(axis="x", rotation=0)

    ref_ct = hist.groupby(["cluster", "referral_reason"]).size().unstack(fill_value=0)
    ref_ct.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Referral Reason by Cluster")
    axes[1, 1].set_xlabel("Cluster"); axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].legend(fontsize=7, loc="upper right")

    fig.suptitle("Q3(a) — Intake Features by Trajectory Cluster",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "q3a_feature_distributions.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── Boxplots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=hist, x="cluster", y="age_years", hue="cluster", ax=axes[0], palette="Set2", legend=False)
    sns.boxplot(data=hist, x="cluster", y="complexity_score", hue="cluster", ax=axes[1], palette="Set2", legend=False)
    axes[0].set_title("Age by Cluster")
    axes[1].set_title("Complexity Score by Cluster")
    fig.suptitle("Q3(a) — Age & Complexity by Cluster", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "q3a_boxplots.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Q3(a) plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# PART (b) — TRAIN CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode gender and referral_reason; keep numeric cols as-is."""
    return pd.get_dummies(
        df[["age_years", "complexity_score", "gender", "referral_reason"]],
        drop_first=False,
    )


def train_and_evaluate(hist: pd.DataFrame) -> tuple[object, pd.Index]:
    """
    Train logistic regression and random forest on historical clients.
    Returns the best model and the training feature columns (for alignment).
    """
    X_full = prepare_features(hist)
    y_full = hist["cluster"].values
    feature_cols = X_full.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train class distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")

    # Model 1: Multinomial Logistic Regression  (parametric)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)

    # Model 2: Random Forest  (non-parametric)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=6)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)

    print(f"\nLogistic Regression accuracy: {lr_acc:.3f}")
    print(f"Random Forest accuracy:        {rf_acc:.3f}")

    # Confusion matrices
    classes = sorted(hist["cluster"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pred, name, acc in [
        (axes[0], lr_pred, "Logistic Regression", lr_acc),
        (axes[1], rf_pred, "Random Forest",       rf_acc),
    ]:
        cm   = confusion_matrix(y_test, pred, labels=classes)
        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAccuracy = {acc:.3f}")
    fig.suptitle("Q3(b) — Confusion Matrices", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "q3b_confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Feature importances (Random Forest)
    fi = pd.Series(rf.feature_importances_,
                   index=X_full.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(fi) * 0.35)))
    fi.plot(kind="barh", ax=ax, color="#2196F3")
    ax.set_title("Q3(b) — Random Forest Feature Importances",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT / "q3b_feature_importances.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Q3(b) plots saved.")

    # Return best model
    best_model = rf if rf_acc >= lr_acc else lr
    best_name  = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
    print(f"\nRecommended model: {best_name} "
          f"(higher accuracy; better handles non-linear feature interactions)")
    return best_model, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# PART (c) — WAITLIST CAPACITY ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def predict_waitlist(waitlist: pd.DataFrame, best_model,
                     feature_cols: pd.Index, policy: dict) -> pd.DataFrame:
    """Predict cluster membership for waitlist clients and estimate capacity."""
    X_wait = prepare_features(waitlist)
    X_wait = X_wait.reindex(columns=feature_cols, fill_value=0)

    waitlist = waitlist.copy()
    waitlist["predicted_cluster"] = best_model.predict(X_wait)
    waitlist["E_delivered"]       = waitlist["predicted_cluster"].map(
        lambda c: policy[c]["E_delivered"])
    waitlist["E_savings"]         = waitlist["predicted_cluster"].map(
        lambda c: policy[c]["E_savings"])
    return waitlist


def report_capacity(waitlist: pd.DataFrame, policy: dict) -> None:
    n              = len(waitlist)
    total_policy   = waitlist["E_delivered"].sum()
    total_baseline = n * TMAX
    saved          = total_baseline - total_policy

    print(f"\n=== Q3(c) — Capacity Impact on Waitlist (n={n}) ===")
    print(f"  Baseline (12 sessions each):        {total_baseline:.0f} sessions")
    print(f"  Under differentiated policy:         {total_policy:.1f} sessions")
    print(f"  Total sessions saved:                {saved:.1f} sessions")
    print(f"  Average sessions per child:          {total_policy / n:.2f}")
    print(f"\n  Predicted cluster distribution:")
    print(waitlist["predicted_cluster"].value_counts().sort_index()
                  .rename("n_clients").to_string())
    print(f"\n  Savings breakdown by cluster:")
    for c in sorted(waitlist["predicted_cluster"].unique()):
        sub   = waitlist[waitlist["predicted_cluster"] == c]
        saved_c = (TMAX - policy[c]["E_delivered"]) * len(sub)
        print(f"    Cluster {c} (n={len(sub)}): Q*={policy[c]['Q_star']}, "
              f"E_delivered={policy[c]['E_delivered']:.2f}, "
              f"sessions saved={saved_c:.1f}")


def plot_capacity(waitlist: pd.DataFrame, policy: dict) -> None:
    """Bar chart: expected sessions delivered per cluster vs. Tmax baseline."""
    clusters_present = sorted(waitlist["predicted_cluster"].unique())
    delivered        = [policy[c]["E_delivered"] for c in clusters_present]
    colors           = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"][:len(clusters_present)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"Cluster {c}" for c in clusters_present], delivered,
           color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(TMAX, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline ({TMAX} sessions)")
    ax.set_ylabel("Expected sessions delivered", fontsize=11)
    ax.set_title("Q3(c) — Expected Sessions Delivered vs. Baseline\n(Current Waitlist, n="
                 + str(len(waitlist)) + ")", fontsize=12, fontweight="bold")
    ax.set_ylim(0, TMAX + 1.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "q3c_capacity_savings.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Q3(c) plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    features, waitlist, labels_df, FINAL_K, clusters = load_data()
    policy = build_cluster_policy(clusters, FINAL_K)
    print(f"Cluster policies: {policy}")

    # ── Historical clients (those in the scored/unlabeled set) ───────────────
    hist = (features[features["dataset_split"] == "unlabeled"]
            .merge(labels_df, on="client_id", how="inner"))
    print(f"\nHistorical clients with cluster labels: {len(hist)}")
    print(hist["cluster"].value_counts().sort_index())

    # ── (a) Feature exploration ───────────────────────────────────────────────
    explore_features(hist, FINAL_K)

    # ── (b) Train classifiers ────────────────────────────────────────────────
    best_model, feature_cols = train_and_evaluate(hist)

    # ── (c) Waitlist predictions + capacity ──────────────────────────────────
    waitlist_pred = predict_waitlist(waitlist, best_model, feature_cols, policy)
    report_capacity(waitlist_pred, policy)
    plot_capacity(waitlist_pred, policy)

    waitlist_pred.to_csv(OUT / "waitlist_predictions.csv", index=False)
    print(f"\nWaitlist predictions saved → {OUT / 'waitlist_predictions.csv'}")


if __name__ == "__main__":
    main()
