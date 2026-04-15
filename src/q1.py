"""
src/q1.py — Prompt Engineering: Extracting Per-Session Progress Scores
=======================================================================

LLM used: Claude claude-sonnet-4-6 (Anthropic) via the anthropic Python SDK.
Temperature = 0.0 for fully deterministic, reproducible outputs.

Prompt strategies implemented
------------------------------
1. Clinical persona / role assignment
   The model is addressed as a "senior speech-language pathologist reviewing a
   child's therapy record," grounding every judgment in a clinically familiar
   frame of reference.

2. Exhibit A framework embedded
   Goal levels (sound → syllable → word → phrase → sentence → conversation) and
   independence levels (imitation → cueing → spontaneous) are spelled out so the
   model uses the same rubric Patel uses when reading notes in sequence.

3. Explicit 0–3 score definitions with when-to-use guidance
   Each label maps directly to the assignment rating scale and includes a plain-
   language description of what the note-pair must show before that score applies.

4. Boundary clarifications (0 vs 1, 1 vs 2, 2 vs 3)
   Boundary language distinguishes incremental within-level gains (score 1) from
   clinically obvious steps forward (score 2), and reserves score 3 for explicit
   goal-level jumps or fully spontaneous production.

5. Calibration reminder
   Score 0 accounts for roughly 53 % of Patel's labeled transitions. The prompt
   explicitly warns the model not to over-score — "assign 0 whenever you are
   uncertain" — to prevent the common LLM bias toward higher scores.

6. Strict JSON-only output format constraint
   The model is instructed to return nothing but a raw JSON array of integers.
   No markdown, no explanation, no extra keys.

Performance metric: Quadratic Weighted Kappa (QWK)
----------------------------------------------------
Because scores are ordinal (0 < 1 < 2 < 3), QWK is the primary metric. It
penalises disagreements proportionally to how far apart the scores are and
corrects for chance agreement, making it appropriate for imbalanced label
distributions. A secondary exact-accuracy figure is also reported for
interpretability. QWK ≥ 0.60 is conventionally "substantial agreement" for
ordinal clinical ratings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseQ1Config:
    client_id_key:    str = "client_id"
    notes_key:        str = "notes"
    note_number_key:  str = "note_number"
    note_text_key:    str = "note_text"
    true_vector_key:  str = "scored_progress"
    pred_vector_key:  str = "estimated_trajectory_vector"
    valid_scores:     tuple = (0, 1, 2, 3)   # 0=maintenance … 3=major gain


@dataclass
class Q1ALabeledConfig(BaseQ1Config):
    test_path:              str = "data/labeled_notes.json"
    evaluated_output_path:  str = "output/q1/evaluated_labeled_results.json"


@dataclass
class Q1BUnlabeledConfig(BaseQ1Config):
    unlabeled_path: str = "data/unlabeled_notes.json"
    output_path:    str = "output/q1/scored_notes.json"


# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}.")
    return data


def save_json(data: Any, path: str) -> None:
    p = ensure_parent_dir(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(notes_json_str: str) -> str:
    """
    Build the full prompt for one client's session-note sequence.

    Parameters
    ----------
    notes_json_str : str
        The client's full note sequence serialised as a JSON string.
        Each note is a dict with keys "note_number" and "note_text".

    Returns
    -------
    str
        Complete prompt to send to the LLM.
    """
    return f"""You are a senior speech-language pathologist (SLP) at a pediatric rehabilitation \
centre. You are reviewing a child's therapy record to assess progress between sessions.

=== CLINICAL FRAMEWORK (Exhibit A) ===
Every session note describes the child's current goal and independence level:

Goal Level — simpler to more complex
  1. Sound production
  2. Syllable production
  3. Single-word production
  4. Carrier-phrase production
  5. Sentence production
  6. Spontaneous speech / conversation

Independence Level — within a given goal
  1. By imitation (maximum clinician support)
  2. With cueing (partial support)
  3. Spontaneously / on their own (no support needed)

Progress occurs when a child moves to a higher goal level, requires less cueing within the \
same goal, shows greater accuracy or consistency, or generalises skills across new settings.

=== RATING SCALE ===
Score  Label                   When to assign
  0    Maintenance /           The child is at essentially the same level as before.
       minimal change          Accuracy, cueing needs, and goal level are similar.
                               ▶ This is the MOST COMMON outcome. Assign 0 whenever you
                                 are uncertain or the note describes continued work at the
                                 same level without documented accuracy gains.
  1    Small but clear         Modest progress WITHIN the same general level —
       improvement             slightly better consistency, a little less cueing, or
                               improved carryover — but NO major jump in goal level or
                               independence stage.
  2    Meaningful clinical     An obvious step forward that matters clinically: moving
       progress                from inconsistent to fairly consistent performance,
                               clearly requiring less support, or showing broader
                               generalisation across activities or settings.
  3    Major gain /            A clear breakthrough: jump to a new goal level (e.g.
       step up in level        words → phrases), major independence gain (e.g. cued →
                               fully spontaneous), or new spontaneous use across contexts.
                               ▶ USE SPARINGLY — this is a rare event.

=== BOUNDARY CLARIFICATIONS ===
0 vs 1 — A note that simply continues work at the same level with no concrete accuracy
  gain is score 0. Score 1 requires a documented, concrete (even if small) improvement.
1 vs 2 — Score 1 is incremental (e.g. slightly better consistency on the same task).
  Score 2 requires an obvious, clinically significant step (e.g. 50 % → 80 %+ accuracy,
  or shifting from heavily cued to more spontaneous production).
2 vs 3 — Score 3 is rare. Reserve it for an explicit goal-level jump or a clear shift
  from fully cued to fully spontaneous use documented in the note.

=== YOUR TASK ===
The session notes for one child are provided below, in order (Session 1, 2, …).
For EVERY consecutive pair (Note N → Note N+1) assign ONE score from the scale above.

Rules:
- Compare each note only to the one immediately before it.
- Do NOT score the first note in isolation — scoring starts at the transition 1→2.
- For N notes you must return exactly N−1 scores.

=== OUTPUT FORMAT ===
Return ONLY a raw JSON array of integers — nothing else.
No explanation. No markdown fences. No surrounding text. No extra keys.
Correct example for a 4-note sequence: [0, 1, 2]

=== SESSION NOTES ===
{notes_json_str}
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL  (Claude claude-sonnet-4-6 via Anthropic SDK)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# PARSING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def parse_vector_from_response(
    response_text: str,
    expected_length: int,
    valid_scores: tuple = (0, 1, 2, 3),
) -> List[int]:
    """Parse the model's raw response into a validated integer list."""
    try:
        text = response_text.strip()
        # Strip any accidental markdown fences
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        if not isinstance(data, list):
            raise ValueError("Not a list")
        cleaned = []
        for v in data:
            score = int(v)
            if score not in valid_scores:
                raise ValueError(f"Invalid score {score}")
            cleaned.append(score)
        if len(cleaned) != expected_length:
            raise ValueError(f"Length mismatch: expected {expected_length}, got {len(cleaned)}")
        return cleaned
    except Exception:
        return []


def get_validated_vector_from_llm(
    prompt: str,
    expected_length: int,
    config: BaseQ1Config,
    client_id: str,
) -> List[int]:
    """Call the LLM, validate the vector, retry once on failure."""
    if expected_length == 0:
        return []
    for attempt in (1, 2):
        raw = call_llm(prompt)
        vec = parse_vector_from_response(raw, expected_length, config.valid_scores)
        if vec:
            return vec
        if attempt == 1:
            print(f"  [retry] Invalid response for {client_id}, retrying...")
    raise RuntimeError(f"LLM returned invalid vector twice for {client_id}.")


# ─────────────────────────────────────────────────────────────────────────────
# SCORING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def score_client_record(client_record: Dict[str, Any], config: BaseQ1Config) -> Dict[str, Any]:
    all_notes     = client_record[config.notes_key]
    client_id     = str(client_record[config.client_id_key])
    notes_str     = json.dumps(all_notes, ensure_ascii=False, indent=2)
    expected_len  = max(len(all_notes) - 1, 0)

    prompt           = build_prompt(notes_str)
    estimated_vector = get_validated_vector_from_llm(prompt, expected_len, config, client_id)

    out = {
        config.client_id_key: client_record[config.client_id_key],
        config.notes_key:     client_record[config.notes_key],
        config.pred_vector_key: estimated_vector,
    }
    if config.true_vector_key in client_record:
        out[config.true_vector_key] = client_record[config.true_vector_key]
    return out


def score_dataset(
    data: List[Dict[str, Any]],
    config: BaseQ1Config,
    progress_desc: str,
) -> List[Dict[str, Any]]:
    return [score_client_record(r, config)
            for r in tqdm(data, desc=progress_desc)]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_step_comparisons(client_id, true_vec, pred_vec):
    return [
        {"client_id": client_id, "step_number": i + 1,
         "true_score": t, "estimated_score": p}
        for i, (t, p) in enumerate(zip(true_vec, pred_vec))
    ]


def build_evaluation_comparisons(scored_data, config):
    client_rows, step_rows = [], []
    for rec in scored_data:
        cid   = str(rec[config.client_id_key])
        true  = rec.get(config.true_vector_key, [])
        pred  = rec.get(config.pred_vector_key, [])
        steps = build_step_comparisons(cid, true, pred)
        client_rows.append({
            "client_id": cid,
            "true_vector": true, "estimated_vector": pred,
            "n_compared_scores": len(steps),
            "step_comparisons": steps,
        })
        step_rows.extend(steps)
    return {"n_clients": len(scored_data),
            "client_level_comparisons": client_rows,
            "step_level_comparisons": step_rows}


def build_confusion_matrix(step_rows, valid_scores):
    matrix     = {t: {p: 0 for p in valid_scores} for t in valid_scores}
    for row in step_rows:
        t, p = row["true_score"], row["estimated_score"]
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    row_totals = {t: sum(matrix[t].values()) for t in valid_scores}
    col_totals = {p: sum(matrix[t][p] for t in valid_scores) for p in valid_scores}
    grand      = sum(row_totals.values())

    rw = max(len("true\\pred"), len("Total"))
    cw = max(5, len(str(grand)))
    hdr  = " | ".join(["true\\pred".rjust(rw)] + [str(s).rjust(cw) for s in valid_scores] + ["Total".rjust(cw)])
    sep  = "-+-".join(["-" * rw] + ["-" * cw] * (len(valid_scores) + 1))
    rows = [hdr, sep]
    for t in valid_scores:
        rows.append(" | ".join(
            [str(t).rjust(rw)]
            + [str(matrix[t][p]).rjust(cw) for p in valid_scores]
            + [str(row_totals[t]).rjust(cw)]))
    rows += [sep, " | ".join(["Total".rjust(rw)]
             + [str(col_totals[p]).rjust(cw) for p in valid_scores]
             + [str(grand).rjust(cw)])]
    return {
        "labels": list(valid_scores), "counts": matrix,
        "row_totals": row_totals, "column_totals": col_totals,
        "grand_total": grand, "table": "\n".join(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute Quadratic Weighted Kappa (primary) and exact accuracy (secondary).

    QWK is chosen because scores are ordinal (0 < 1 < 2 < 3): disagreements
    are penalised proportionally to their distance, and chance agreement is
    corrected for — both important given the heavy class imbalance (score 0
    makes up ~53 % of Patel's labels).

    Exact accuracy is reported as a secondary, intuitive check but is not used
    to compare prompt versions due to the class imbalance.
    """
    from sklearn.metrics import cohen_kappa_score
    from collections import defaultdict

    true_scores = [r["true_score"]      for r in step_rows]
    pred_scores = [r["estimated_score"] for r in step_rows]
    n           = len(true_scores)

    exact_acc = sum(t == p for t, p in zip(true_scores, pred_scores)) / n
    qwk       = cohen_kappa_score(true_scores, pred_scores,
                                  weights="quadratic", labels=[0, 1, 2, 3])

    cc, ct = defaultdict(int), defaultdict(int)
    for t, p in zip(true_scores, pred_scores):
        ct[t] += 1
        if t == p:
            cc[t] += 1
    per_class_recall = {
        str(label): round(cc[label] / ct[label], 4) if ct[label] else 0.0
        for label in [0, 1, 2, 3]
    }
    return {
        "n_transitions":             n,
        "exact_accuracy":            round(exact_acc, 4),
        "quadratic_weighted_kappa":  round(qwk, 4),
        "per_class_recall":          per_class_recall,
    }


def evaluate_predictions(config: Q1ALabeledConfig) -> Dict[str, Any]:
    scored = load_json(config.evaluated_output_path)
    comps  = build_evaluation_comparisons(scored, config)
    steps  = comps["step_level_comparisons"]
    return {**compute_metrics(steps),
            "confusion_matrix": build_confusion_matrix(steps, config.valid_scores)}


def print_evaluation(results: Dict[str, Any]) -> None:
    print("\n=== Q1 Evaluation Results ===")
    for key, value in results.items():
        if key == "confusion_matrix" and isinstance(value, dict):
            print("confusion_matrix:")
            print(value.get("table", ""))
        else:
            print(f"{key}: {value}")


# ─────────────────────────────────────────────────────────────────────────────
# ALSO SAVE scored_notes.csv (for Q2 compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def export_scored_csv(scored_data: List[Dict[str, Any]], csv_path: str) -> None:
    """Save scored_notes.csv with columns: client_id, session, score."""
    import csv
    rows = []
    for rec in scored_data:
        cid    = rec["client_id"]
        scores = rec["estimated_trajectory_vector"]
        for session_idx, score in enumerate(scores, start=1):
            rows.append({"client_id": cid, "session": session_idx, "score": score})
    p = ensure_parent_dir(csv_path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "session", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def run_test_pipeline(config: Q1ALabeledConfig) -> None:
    """Score labeled set, evaluate, print metrics + confusion matrix."""
    test_data = load_json(config.test_path)
    scored    = score_dataset(test_data, config, "Scoring labeled clients (Q1a)")
    save_json(scored, config.evaluated_output_path)
    results = evaluate_predictions(config)
    print_evaluation(results)


def run_unlabeled_pipeline(config: Q1BUnlabeledConfig) -> None:
    """Score all unlabeled clients, save JSON + CSV for Q2."""
    unlabeled = load_json(config.unlabeled_path)
    scored    = score_dataset(unlabeled, config, "Scoring unlabeled clients (Q1b)")
    save_json(scored, config.output_path)
    csv_path  = config.output_path.replace(".json", ".csv")
    export_scored_csv(scored, csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure ANTHROPIC_API_KEY is set before running:
    #   export ANTHROPIC_API_KEY="sk-ant-..."

    LABELED_CONFIG = Q1ALabeledConfig(
        test_path="data/labeled_notes.json",
        evaluated_output_path="output/q1/evaluated_labeled_results.json",
    )
    UNLABELED_CONFIG = Q1BUnlabeledConfig(
        unlabeled_path="data/unlabeled_notes.json",
        output_path="output/q1/scored_notes.json",
    )

    # Step 1 — validate prompt on labeled set, print metrics + confusion matrix
    run_test_pipeline(LABELED_CONFIG)

    # Step 2 — score all 80 unlabeled clients → output/q1/scored_notes.json + .csv
    run_unlabeled_pipeline(UNLABELED_CONFIG)
