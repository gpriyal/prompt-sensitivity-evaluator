import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from bert_score import score as bert_score_compute
from sentence_transformers import SentenceTransformer, util
import numpy as np

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


# 1. BERTSCORE (individual — single-question mode)
def compute_bertscore(candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 0.0
    try:
        P, R, F1 = bert_score_compute(
            [candidate],
            [reference],
            lang="en",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return round(float(F1[0]), 4)
    except Exception as e:
        print(f"BERTScore error: {e}")
        return 0.0


# 2. COSINE CONSISTENCY SCORE
def compute_consistency_score(outputs: list[str]) -> float:
    valid = [o for o in outputs if o and len(o.strip()) > 10]
    if len(valid) < 2:
        return 0.0
    try:
        embeddings = _embedder.encode(valid, convert_to_tensor=True)
        cosine_matrix = util.cos_sim(embeddings, embeddings)
        n = len(valid)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(float(cosine_matrix[i][j]))
        return round(float(np.mean(pairs)), 4)
    except Exception:
        return 0.0


# 3. TRUNCATION DETECTOR
def is_truncated(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return True
    last_char = text.strip()[-1]
    return last_char not in [".", "!", "?", '"', "'", ")", "]", ":", "}", "~"]


# 4. BUILD REFERENCE
def build_reference(model_outputs: list[dict]) -> str:
    """
    Uses Llama-3.3-70B's output (first in model_outputs, our strongest model) 
    as the reference for BERTScore. Falls back to the next valid (non-truncated) 
    output if the flagship model's output was truncated or empty.
    """
    valid = [
        m["output"] for m in model_outputs
        if m["output"] and not is_truncated(m["output"])
    ]
    if not valid:
        return ""
    return valid[0]


# 5. EVALUATE ALL - INDIVIDUAL (single-question mode)
def evaluate_all(enriched_variants: list[dict]) -> list[dict]:

    model_output_map = {}
    for variant in enriched_variants:
        for model_out in variant["model_outputs"]:
            model_name = model_out["model"]
            if model_name not in model_output_map:
                model_output_map[model_name] = []
            if model_out["output"]:
                model_output_map[model_name].append(model_out["output"])

    consistency_scores = {
        model: compute_consistency_score(outputs)
        for model, outputs in model_output_map.items()
    }

    scored_variants = []
    for variant in enriched_variants:
        reference = build_reference(variant["model_outputs"])

        scored_model_outputs = []
        for model_out in variant["model_outputs"]:
            output_text = model_out["output"] or ""

            bertscore_val = compute_bertscore(output_text, reference) if reference else 0.0
            truncated     = is_truncated(output_text)
            consistency   = consistency_scores.get(model_out["model"], 0.0)

            scored_model_outputs.append({
                **model_out,
                "scores": {
                    "bertscore":   bertscore_val,
                    "truncated":   truncated,
                    "consistency": consistency
                }
            })

        scored_variants.append({
            **variant,
            "model_outputs": scored_model_outputs
        })

    return scored_variants


# 6. EVALUATE ALL - BATCHED (batch experiment mode)
def evaluate_all_batched(enriched_variants: list[dict]) -> list[dict]:

    model_output_map = {}
    for variant in enriched_variants:
        for model_out in variant["model_outputs"]:
            model_name = model_out["model"]
            if model_name not in model_output_map:
                model_output_map[model_name] = []
            if model_out["output"]:
                model_output_map[model_name].append(model_out["output"])

    consistency_scores = {
        model: compute_consistency_score(outputs)
        for model, outputs in model_output_map.items()
    }

    all_candidates = []
    all_references = []
    index_map = []

    for v_idx, variant in enumerate(enriched_variants):
        reference = build_reference(variant["model_outputs"])
        for m_idx, model_out in enumerate(variant["model_outputs"]):
            all_candidates.append(model_out["output"] or "")
            all_references.append(reference)
            index_map.append((v_idx, m_idx))

    safe_candidates = [
        c if c and len(c.strip()) > 5 else "no response"
        for c in all_candidates
    ]
    safe_references = [
        r if r and len(r.strip()) > 5 else "no response"
        for r in all_references
    ]

    try:
        P, R, F1 = bert_score_compute(
            safe_candidates,
            safe_references,
            lang="en",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=16
        )
        all_bertscores = [round(float(f), 4) for f in F1]
    except Exception as e:
        print(f"Batched BERTScore error: {e}")
        all_bertscores = [0.0] * len(all_candidates)

    scored_variants = []
    for variant in enriched_variants:
        scored_variants.append({
            **variant,
            "model_outputs": [dict(m) for m in variant["model_outputs"]]
        })

    for flat_idx, (v_idx, m_idx) in enumerate(index_map):
        model_out   = scored_variants[v_idx]["model_outputs"][m_idx]
        output_text = model_out["output"] or ""

        model_out["scores"] = {
            "bertscore":   all_bertscores[flat_idx],
            "truncated":   is_truncated(output_text),
            "consistency": consistency_scores.get(model_out["model"], 0.0)
        }

    return scored_variants