# MSE 555 — Assignment 3: Learning from Clinical Notes
**Westfield Children's Centre — Applied Case Study**

---

## Repository Structure

```
.
├── data/
│   ├── labeled_notes.json       # 40 hand-scored clients (Q1 validation)
│   ├── unlabeled_notes.json     # 80 clients to be scored by LLM (Q1b → Q2)
│   ├── client_features.csv      # Intake characteristics for all historical clients
│   └── waitlist.csv             # Intake characteristics for waiting clients
├── src/
│   ├── q1.py                    # Prompt engineering + LLM scoring pipeline
│   ├── q2.py                    # Clustering, newsvendor optimisation, plots
│   └── q3.py                    # Predictive classifiers + waitlist capacity
├── output/
│   ├── q1/                      # evaluated_labeled_results.json, scored_notes.json/.csv
│   ├── q2/                      # cluster_labels.csv, q2_results.pkl, all plots
│   └── q3/                      # waitlist_predictions.csv, all plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # required for q1.py
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

## Design Decisions

### Q1 — Prompt Engineering

The prompt encodes David Patel's note-reading framework by:

1. **Clinical persona** — framing the task as a senior SLP reviewing a therapy record.
2. **Exhibit A embedded** — goal levels (sound → conversation) and independence levels (imitation → spontaneous) are spelled out so the model applies the same rubric.
3. **Explicit 0–3 scale** — each score label includes a plain-language "when to assign" description drawn directly from the assignment rubric.
4. **Boundary clarifications** — 0 vs. 1 (maintenance vs. concrete gain), 1 vs. 2 (incremental vs. clinically obvious), 2 vs. 3 (clinical step vs. rare breakthrough).
5. **Calibration reminder** — score 0 is ~53 % of Patel's transitions; the prompt explicitly warns against over-scoring.
6. **JSON-only output** — prevents markdown leakage that would break the parser.

**Evaluation metric:** Quadratic Weighted Kappa (QWK). QWK penalises ordinal disagreements proportionally to their distance and corrects for chance, making it appropriate for the imbalanced label distribution here. Exact accuracy is reported as a secondary check.

### Q2 — K Selection

Final K is selected by maximising `distinct_Q_values × savings_spread` subject to every cluster having ≥ 8 clients. This operationalises the assignment criterion: clusters that produce "genuinely distinct and actionable" reassessment policies.

### Q3 — Model Recommendation

The Random Forest is the preferred model for deployment at Westfield because:
- It is non-parametric and can capture non-linear interactions between age, complexity, and referral reason without assuming a particular functional form.
- Feature importances provide interpretable guidance about which intake factors drive trajectory group membership.
- It is robust to the small sample size through ensemble averaging.

Logistic regression is retained as a parametric baseline and provides a useful sanity check.

---

## Case Implications (Q2f)

Chen and Patel's hypothesis — that children may receive similar session counts while progressing very differently — is supported by the clustering results. Clients assigned the same 12-session pathway fall into distinct trajectory groups:

- **Early-plateau clusters** show large cumulative gains concentrated in the first 6–8 sessions, with minimal additional progress thereafter. For these children, a reassessment at session 7–8 can confirm that the plateau has been reached and discharge can follow, freeing capacity without clinical compromise.
- **Late-plateau / gradual clusters** continue accumulating progress through session 10–11. Scheduling a reassessment earlier for these children would trigger the "too early" branch of the audit rule (no sessions saved) and may interrupt clinically meaningful ongoing work.

The differentiated policy translates this variation into a concrete planning tool: rather than defaulting to session 12 for all clients, Westfield can schedule reassessments at cluster-specific Q* values. Across the current waitlist of 35 children, the policy is estimated to recover approximately 60 sessions relative to the Tmax baseline — equivalent to providing care to 5 additional children at average session count.
