import pandas as pd
import numpy as np

# WEIGHTS — how much each metric contributes to the final composite score
WEIGHTS = {
    "judge_score":  0.50,   
    "bertscore":    0.30,   
    "consistency":  0.20,   
}


def normalise_judge(raw_score: float) -> float:
    """Judge scores come in as 1-10 avg, normalise to 0-1"""
    return round(raw_score / 10, 4)


def compute_composite(scores: dict) -> float:
    judge_norm  = normalise_judge(float(scores.get("judge_score", 0)))
    bertscore   = float(scores.get("bertscore", 0))
    consistency = float(scores.get("consistency", 0))
    truncated   = scores.get("truncated", False)

    composite = (
        WEIGHTS["judge_score"] * judge_norm  +
        WEIGHTS["bertscore"]   * bertscore   +
        WEIGHTS["consistency"] * consistency 
    )

    if truncated:
        composite *= 0.90  # 10% penalty

    return round(composite, 4)


# PROMPT SENSITIVITY INDEX (PSI)
def compute_psi_table(df: pd.DataFrame) -> list[dict]:
    """
    PSI = std(composite) / mean(composite) across all 8 prompt variants, per model.
    Lower PSI  → model performance is stable regardless of prompt phrasing (robust)
    Higher PSI → model performance swings significantly based on prompt phrasing (sensitive)
    """
    psi_table = []

    for model in df["model"].unique():
        subset = df[df["model"] == model]
        scores = subset["composite"].values
        if len(scores) == 0:
            continue

        mean_score = float(np.mean(scores))
        std_score  = float(np.std(scores))
        psi = round(std_score / mean_score, 4) if mean_score > 0 else 0.0

        if psi < 0.05:
            sensitivity = "Low"
        elif psi < 0.10:
            sensitivity = "Medium"
        else:
            sensitivity = "High"

        best_idx  = subset["composite"].idxmax()
        worst_idx = subset["composite"].idxmin()

        psi_table.append({
            "model":       model,
            "mean_score":  round(mean_score, 4),
            "psi":         psi,
            "sensitivity": sensitivity,
            "best_style":  subset.loc[best_idx,  "variant_style"],
            "worst_style": subset.loc[worst_idx, "variant_style"]
        })

    return sorted(psi_table, key=lambda x: x["psi"])


def aggregate(judged_variants: list[dict]) -> dict:
    """
    Master aggregation function.

    Returns a dict with:
        rows          - flat list of all (variant x model) records
        df            - pandas DataFrame for easy charting
        best_variant  - highest avg composite score across models
        worst_variant - lowest avg composite score across models
        model_summary - per-model average composite score
        variant_summary - per-variant average composite score
    """
    rows = []

    for variant in judged_variants:
        for model_out in variant["model_outputs"]:
            scores  = model_out.get("scores", {})
            composite = compute_composite(scores)

            rows.append({
                "variant_style":  variant["style"],
                "model":          model_out["model"],
                "output":         model_out["output"] or "",
                "error":          model_out.get("error"),
                "truncated":      scores.get("truncated", False),
                # raw scores
                "bertscore":      scores.get("bertscore",       0),
                "consistency":    scores.get("consistency",     0),
                "accuracy":       scores.get("accuracy",        0),
                "clarity":        scores.get("clarity",         0),
                "completeness":   scores.get("completeness",    0),
                "judge_score":    scores.get("judge_score",     0),
                # final
                "composite":      composite
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return {"rows": rows, "df": df,
                "best_variant": None, "worst_variant": None,
                "model_summary": {}, "variant_summary": {}, "psi_table": []}

    # per-variant average composite (across all models)
    variant_summary = (
        df.groupby("variant_style")["composite"]
        .mean()
        .round(4)
        .sort_values(ascending=False)
        .to_dict()
    )

    # per-model average composite (across all variants)
    model_summary = (
        df.groupby("model")["composite"]
        .mean()
        .round(4)
        .sort_values(ascending=False)
        .to_dict()
    )

    best_variant  = max(variant_summary, key=variant_summary.get)
    worst_variant = min(variant_summary, key=variant_summary.get)
    psi_table = compute_psi_table(df)

    return {
        "rows":             rows,
        "df":               df,
        "best_variant":     best_variant,
        "worst_variant":    worst_variant,
        "model_summary":    model_summary,
        "variant_summary":  variant_summary,
        "psi_table":        psi_table
    }


# FINDINGS TABLE (single-question mode)
def generate_findings(results: dict) -> list[dict]:
    findings = []

    findings.append({"Finding": "Best Prompt Style",  "Result": results["best_variant"]})
    findings.append({"Finding": "Worst Prompt Style", "Result": results["worst_variant"]})

    psi_table = results.get("psi_table", [])
    if psi_table:
        most_robust    = psi_table[0]    # sorted ascending by PSI
        most_sensitive = psi_table[-1]
        findings.append({
            "Finding": "Most Robust Model",
            "Result": f"{most_robust['model']} (PSI = {most_robust['psi']})"
        })
        findings.append({
            "Finding": "Most Sensitive Model",
            "Result": f"{most_sensitive['model']} (PSI = {most_sensitive['psi']})"
        })

    model_summary = results.get("model_summary", {})
    if model_summary:
        best_model  = max(model_summary, key=model_summary.get)
        worst_model = min(model_summary, key=model_summary.get)
        findings.append({
            "Finding": "Highest Avg Composite",
            "Result": f"{model_summary[best_model]:.3f} ({best_model})"
        })
        findings.append({
            "Finding": "Lowest Avg Composite",
            "Result": f"{model_summary[worst_model]:.3f} ({worst_model})"
        })

    df = results.get("df")
    if df is not None and not df.empty:
        truncated_count = int(df["truncated"].sum())
        findings.append({
            "Finding": "Truncated Outputs",
            "Result": f"{truncated_count} / {len(df)}"
        })

    return findings