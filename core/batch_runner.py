import os
import json
import time
import numpy as np

from core.variant_generator import generate_variants
from core.llm_runner import run_variants_on_all_models
from core.evaluator import evaluate_all_batched
from core.llm_judge import judge_all
from core.aggregator import aggregate

CACHE_PATH = "data/batch_cache.json"

REQUIRED_KEYS = [
    "total_questions", "per_question", "variant_win_counts",
    "variant_avg_scores", "model_avg_scores", "category_variance", "model_psi"
]

def run_batch(progress_callback=None, use_cache=True) -> dict:
    """
    Runs the full pipeline on every question in sample_questions.json.
    Caches results to disk, subsequent calls load instantly unless use_cache=False (force re-run).
    """

    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        if all(key in cached for key in REQUIRED_KEYS):
            print(f"Loaded batch results from cache ({CACHE_PATH})")
            return cached
        else:
            print("Cache is stale — re-running.")

    with open("data/sample_questions.json") as f:
        questions = json.load(f)

    per_question_results = []

    for i, q in enumerate(questions):
        if progress_callback:
            progress_callback(i, len(questions), q["question"])

        try:
            variants = generate_variants(q["question"], q["category"])
            enriched = run_variants_on_all_models(variants)
            scored   = evaluate_all_batched(enriched)
            judged   = judge_all(scored, q["question"])
            result   = aggregate(judged)

            per_question_results.append({
                "question":       q["question"],
                "category":       q["category"],
                "best_variant":   result["best_variant"],
                "worst_variant":  result["worst_variant"],
                "variant_scores": result["variant_summary"],
                "model_scores":   result["model_summary"],
                "psi_table":      result.get("psi_table", [])
            })

        except Exception as e:
            print(f"Batch error on Q{i+1}: {e}")
            continue

        time.sleep(3)

    findings = compile_batch_findings(per_question_results)

    with open(CACHE_PATH, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"Batch results cached to {CACHE_PATH}")

    return findings


def compile_batch_findings(per_question: list[dict]) -> dict:
    variant_wins   = {}
    variant_scores = {}
    model_scores   = {}
    category_data  = {}
    model_psi_values = {}

    for r in per_question:
        winner = r["best_variant"]
        if winner:
            variant_wins[winner] = variant_wins.get(winner, 0) + 1

        for style, score in r["variant_scores"].items():
            variant_scores.setdefault(style, []).append(score)

        for model, score in r["model_scores"].items():
            model_scores.setdefault(model, []).append(score)
            
        for entry in r.get("psi_table", []):
            model_psi_values.setdefault(entry["model"], []).append(entry["psi"])

        cat = r["category"]
        category_data.setdefault(cat, []).append(r["variant_scores"])

    variant_avg = {k: round(sum(v) / len(v), 4) for k, v in variant_scores.items()}
    model_avg   = {k: round(sum(v) / len(v), 4) for k, v in model_scores.items()}

    category_variance = {}
    for cat, score_dicts in category_data.items():
        all_scores = [s for d in score_dicts for s in d.values()]
        category_variance[cat] = round(float(np.std(all_scores)), 4) if all_scores else 0.0

    model_psi = {
        k: round(sum(v) / len(v), 4)
        for k, v in model_psi_values.items()
    }
    model_psi = dict(sorted(model_psi.items(), key=lambda x: x[1]))  

    return {
        "total_questions": len(per_question),
        "per_question": per_question,
        "variant_win_counts": dict(sorted(variant_wins.items(), key=lambda x: x[1], reverse=True)),
        "variant_avg_scores": dict(sorted(variant_avg.items(), key=lambda x: x[1], reverse=True)),
        "model_avg_scores": dict(sorted(model_avg.items(), key=lambda x: x[1], reverse=True)),
        "category_variance": category_variance,
        "model_psi" : model_psi
    }


# FINDINGS TABLE (batch experiment mode)
def generate_batch_findings(batch: dict) -> list[dict]:
    findings = []

    top_variant = list(batch["variant_win_counts"].keys())[0]
    top_wins    = list(batch["variant_win_counts"].values())[0]
    findings.append({
        "Finding": "Most Frequently Winning Prompt Style",
        "Result":  f"{top_variant} ({top_wins}/{batch['total_questions']})"
    })

    model_scores = batch["model_avg_scores"]
    best_model  = max(model_scores, key=model_scores.get)
    worst_model = min(model_scores, key=model_scores.get)

    findings.append({
        "Finding": "Best Performing Model (avg)",
        "Result":  f"{best_model} ({model_scores[best_model]:.3f})"
    })

    if batch.get("model_psi"):
        items = list(batch["model_psi"].items())   # already sorted ascending
        most_robust    = items[0]
        most_sensitive = items[-1]
        findings.append({
            "Finding": "Most Robust Model (avg PSI)",
            "Result":  f"{most_robust[0]} (PSI = {most_robust[1]})"
        })
        findings.append({
            "Finding": "Most Sensitive Model (avg PSI)",
            "Result":  f"{most_sensitive[0]} (PSI = {most_sensitive[1]})"
        })

    findings.append({
        "Finding": "Highest Avg Composite",
        "Result":  f"{model_scores[best_model]:.3f} ({best_model})"
    })
    findings.append({
        "Finding": "Lowest Avg Composite",
        "Result":  f"{model_scores[worst_model]:.3f} ({worst_model})"
    })

    top_var_cat = max(batch["category_variance"], key=batch["category_variance"].get)
    findings.append({
        "Finding": "Highest Variance Category",
        "Result":  f"{top_var_cat} (σ = {batch['category_variance'][top_var_cat]:.3f})"
    })

    return findings