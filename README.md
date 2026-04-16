# MSE 555 — Assignment 3: Learning from Clinical Notes
**Westfield Children's Centre — Applied Case Study**

---
## Setup

```bash
pip install -r requirements.txt
```
---

## Running the Pipeline

The three scripts must be run **in order** from the repo root.

### Step 1 — Q1: LLM Scoring

```bash
python src/q1.py
```

**What it does:**
- Validates the prompt on `data/labeled_notes.json` (40 labeled clients), prints Quadratic Weighted Kappa, exact accuracy, per-class recall, and confusion matrix.
- Applies the validated prompt to all 80 clients in `data/unlabeled_notes.json`.

**Outputs:**
- `output/q1/evaluated_labeled_results.json` — scored labeled set with true labels (Q1a)
- `output/q1/scored_notes.json` — scored unlabeled clients (Q1b, feeds Q2)
- `output/q1/scored_notes.csv` — same data as columns `client_id, session, score`

**LLM:** Claude claude-sonnet-4-6 (Anthropic), temperature = 0.0.

---

### Step 2 — Q2: Clustering & Newsvendor Optimisation

```bash
python src/q2.py
```

**What it does:**
- Builds cumulative progress trajectories from per-session scores.
- Fits K-means for K = 2, 3, 4, 5 and evaluates each by policy distinctiveness.
- Applies the newsvendor audit model to find Q* per cluster.
- Selects final K and produces all required plots.

**Outputs:**
- `output/q2/spaghetti_K{2,3,4,5}.png` — spaghetti plots for each K evaluated
- `output/q2/plot1_t_star_distributions.png` — t* histograms per cluster
- `output/q2/plot2_expected_savings.png` — E[savings](Q) curves
- `output/q2/plot3_optimized_vs_baseline.png` — optimised Q* vs. mean(t*) baseline
- `output/q2/cluster_labels.csv` — `client_id, cluster` for historical clients
- `output/q2/q2_results.pkl` — pickled results dict for q3.py

**K selection criterion:** Clusters that produce the most distinct Q* values with the widest spread in E[savings] and minimum cluster size ≥ 8.

---

### Step 3 — Q3: Predictive Analytics & Capacity Planning

```bash
python src/q3.py
```

**What it does:**
- Explores intake features (age, complexity, gender, referral reason) broken down by trajectory cluster.
- Trains a multinomial logistic regression and a random forest (80/20 stratified split).
- Applies the best classifier to predict trajectory group for each waitlist client.
- Estimates total session demand under the differentiated reassessment policy vs. the Tmax = 12 baseline.

**Outputs:**
- `output/q3/q3a_feature_distributions.png` — histograms by cluster
- `output/q3/q3a_boxplots.png` — age & complexity boxplots
- `output/q3/q3b_confusion_matrices.png` — confusion matrices for both models
- `output/q3/q3b_feature_importances.png` — random forest feature importances
- `output/q3/q3c_capacity_savings.png` — expected sessions delivered vs. baseline
- `output/q3/waitlist_predictions.csv` — per-client predicted cluster + expected sessions

---
