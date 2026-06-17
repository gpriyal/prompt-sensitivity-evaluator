import os
import streamlit as st
import json
import pandas as pd
from core.variant_generator import generate_variants
from core.llm_runner import run_variants_on_all_models
from core.evaluator import evaluate_all
from core.llm_judge import judge_all
from core.aggregator import aggregate, generate_findings
from core.batch_runner import run_batch, generate_batch_findings

from ui.charts import (variant_bar_chart, model_bar_chart, heatmap_chart,
    metric_comparison_chart, psi_chart, batch_win_chart, batch_model_chart
)

st.set_page_config(
    page_title="Prompt Sensitivity Evaluator",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Prompt Sensitivity Evaluator")
st.caption("See how different prompt styles affect LLM output quality and consistency.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    category = st.selectbox(
        "Question Category",
        ["General", "Science", "Coding", "Medical", "History"]
    )
    st.markdown("---")
    st.markdown("**Models in use:**")
    st.markdown("- 🟢 Groq · Llama-3.3-70B")
    st.markdown("- 🟡 Groq · Llama-3.1-8B")
    st.markdown("- 🔴 Mistral · Small")
    st.markdown("---")
    st.markdown("**Evaluation Metrics:**")
    st.markdown("- BERTScore (semantic quality)")
    st.markdown("- LLM-as-Judge (accuracy/clarity/completeness)")
    st.markdown("- Cosine Consistency (cross-variant stability)")
    st.markdown("- Prompt Sensitivity Index (PSI)")
    st.markdown("---")

    with open("data/sample_questions.json") as f:
        samples = json.load(f)
    filtered = [q for q in samples if q["category"] == category]
    sample_qs = [q["question"] for q in filtered] + ["Enter my own"]
    selected_sample = st.selectbox("Or pick a sample question", sample_qs)

# Input
default_q = selected_sample if selected_sample != "Enter my own" else ""
base_question = st.text_area(
    "Enter your base question",
    value=default_q,
    height=80,
    placeholder="e.g. How does photosynthesis work?"
)

run_button = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

# Pipeline
if run_button:
    if not base_question.strip():
        st.error("Please enter a question first.")
    else:
        st.markdown("---")

        # Step 1 — Generate variants
        with st.status("Step 1/4 — Generating prompt variants...", expanded=False):
            variants = generate_variants(base_question, category)
            st.write(f"✅ {len(variants)} variants generated")

        # Step 2 — LLM inference
        with st.status("Step 2/4 — Running LLM inference (this takes ~60s)...", expanded=False):
            enriched = run_variants_on_all_models(variants)
            st.write("✅ Inference complete across all 3 models")

        # Step 3 — Local evaluation
        with st.status("Step 3/4 — Computing BERTScore + consistency metrics...", expanded=False):
            scored = evaluate_all(enriched)
            st.write("✅ Local metrics computed")

        # Step 4 — LLM judge
        with st.status("Step 4/4 — Running LLM-as-Judge scoring...", expanded=False):
            judged = judge_all(scored, base_question)
            st.write("✅ Judge scoring complete")

        # Aggregate
        results = aggregate(judged)
        df = results["df"]

        st.success("✅ Full evaluation complete!")
        st.markdown("---")

        # Winner / Loser callout
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 Best Prompt Style: **{results['best_variant']}**")
        with col2:
            st.error(f"⚠️ Worst Prompt Style: **{results['worst_variant']}**")

        st.markdown("---")

        # Raw scores table
        st.subheader("Full Results Table")
        display_df = df[[
            "variant_style", "model", "composite",
            "judge_score", "bertscore", "consistency", 
            "truncated"
        ]].sort_values("composite", ascending=False)
        st.dataframe(display_df, use_container_width=True)

        # Per-variant summary
        st.markdown("---")
        st.subheader("Variant Ranking (avg across models)")
        variant_df = (
            df.groupby("variant_style")["composite"]
            .mean()
            .round(4)
            .reset_index()
            .rename(columns={"composite": "avg_composite"})
            .sort_values("avg_composite", ascending=False)
        )
        st.dataframe(variant_df, use_container_width=True)

        # Per-model summary
        st.markdown("---")
        st.subheader("Model Ranking (avg across variants)")
        model_df = (
            df.groupby("model")["composite"]
            .mean()
            .round(4)
            .reset_index()
            .rename(columns={"composite": "avg_composite"})
            .sort_values("avg_composite", ascending=False)
        )
        st.dataframe(model_df, use_container_width=True)

        # PSI Table
        st.markdown("---")
        st.subheader("Prompt Sensitivity Index (PSI)")
        st.caption(
            "PSI (Prompt Sensitivity Index) = std deviation ÷ mean of composite scores "
            "across all 8 prompt variants, for this question."
        )

        psi_display = pd.DataFrame(results["psi_table"]).rename(columns={
            "model":       "Model",
            "mean_score":  "Avg Composite",
            "psi":         "PSI (CV)",
            "sensitivity": "Sensitivity",
            "best_style":  "Best-Fit Style",
            "worst_style": "Worst-Fit Style"
        })
        with st.expander("ℹ️ How to read PSI", expanded=False):
            psi_ranges = pd.DataFrame({
                "PSI Range":      ["< 0.05", "0.05 – 0.10", "0.10 – 0.20", "> 0.20"],
                "Interpretation": ["Extremely robust", "Robust", "Moderately sensitive", "Highly sensitive"]
            })
            st.dataframe(psi_ranges, use_container_width=True, hide_index=True)

        st.dataframe(psi_display, use_container_width=True, hide_index=True)

        # Findings Table
        st.markdown("---")
        st.subheader("Findings")

        findings_display = pd.DataFrame(generate_findings(results))
        st.dataframe(findings_display, use_container_width=True, hide_index=True)

        # Charts
        st.markdown("---")
        st.subheader("Visual Analysis")

        # Bar charts side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                variant_bar_chart(results["variant_summary"]),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                model_bar_chart(results["model_summary"]),
                use_container_width=True
            )

        # PSI chart
        st.plotly_chart(
            psi_chart(results["psi_table"]),
            use_container_width=True
        )

        # Heatmap full width
        st.plotly_chart(heatmap_chart(df), use_container_width=True)

        # Metric breakdown full width
        st.plotly_chart(metric_comparison_chart(df), use_container_width=True)

        # Outputs explorer
        st.markdown("---")
        st.subheader("Explore Raw Outputs")
        for variant in judged:
            with st.expander(f"🔹 {variant['style']}", expanded=False):
                for model_out in variant["model_outputs"]:
                    scores = model_out.get("scores", {})
                    trunc_flag = " ⚠️ TRUNCATED" if scores.get("truncated") else ""
                    st.markdown(f"**{model_out['model']}**{trunc_flag}")
                    st.markdown(f"`composite: {results['df'].loc[(results['df']['variant_style']==variant['style']) & (results['df']['model']==model_out['model']), 'composite'].values[0] if not results['df'].empty else 'N/A'}`")
                    if model_out["error"]:
                        st.error(f"Error: {model_out['error']}")
                    else:
                        st.write(model_out["output"])
                    st.caption(
                        f"BERTScore: {scores.get('bertscore', 0):.3f} | "
                        f"Judge: {scores.get('judge_score', 0):.1f}/10 | "
                        f"Consistency: {scores.get('consistency', 0):.3f}"
                    )
                    st.markdown("---")

        # CSV Export
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Full Results as CSV",
            data=csv,
            file_name="prompt_eval_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# BATCH EXPERIMENT SECTION
st.markdown("---")
st.header("Batch Experiment")
st.caption(
    "Runs the full evaluation pipeline across every sample question "
    "to surface cross-question findings."
)

with st.expander("About this experiment", expanded=False):
    st.markdown("""
    - Runs all sample questions × 8 prompt variants × 3 models
    - Uses batched BERTScore for efficiency (same scores, faster computation)
    - First run takes ~20–30 minutes; results are cached afterward
    - Cached results load instantly on every subsequent visit
    """)

cache_exists = os.path.exists("data/batch_cache.json")

col1, col2 = st.columns([2, 1])
with col1:
    force_rerun = st.checkbox(
        "Force re-run (ignore cache)",
        value=False,
        help="Leave unchecked to load cached results instantly. "
             "Check this only if you've changed prompt variants or models "
             "and want fresh results (~20–30 min)."
    )
with col2:
    run_batch_btn = st.button(
        "🚀 Run Batch Experiment" if not cache_exists else "🔄 Load / Refresh Results",
        type="primary",
        use_container_width=True
    )

if run_batch_btn:
    if force_rerun or not cache_exists:
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def update_progress(i, total, question):
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Running {i+1}/{total}: {question[:60]}...")

        with st.spinner("Running batch experiment — this takes a while on first run..."):
            batch = run_batch(progress_callback=update_progress, use_cache=False)

        progress_bar.empty()
        status_text.empty()
    else:
        batch = run_batch(use_cache=True)

    st.success(f"✅ Batch complete — {batch['total_questions']} questions evaluated")
    st.markdown("---")

    # Key Findings
    st.subheader("🏆 Key Findings")

    top_variant = list(batch["variant_win_counts"].keys())[0]
    top_wins    = list(batch["variant_win_counts"].values())[0]
    top_model   = list(batch["model_avg_scores"].keys())[0]
    top_model_score = list(batch["model_avg_scores"].values())[0]
    top_variance_cat = max(batch["category_variance"], key=batch["category_variance"].get)

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.metric(
            "Most Winning Prompt Style",
            top_variant,
            f"{top_wins}/{batch['total_questions']} questions"
        )
    with fcol2:
        st.metric(
            "Best Performing Model",
            top_model,
            f"avg {top_model_score:.3f}"
        )
    with fcol3:
        st.metric(
            "Highest Variance Category",
            top_variance_cat,
            f"σ = {batch['category_variance'][top_variance_cat]:.3f}"
        )

    st.markdown("---")

    # Charts
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.plotly_chart(
            batch_win_chart(batch["variant_win_counts"], batch["total_questions"]),
            use_container_width=True
        )
    with bcol2:
        st.plotly_chart(
            batch_model_chart(batch["model_avg_scores"]),
            use_container_width=True
        )

    # PSI Table (Batch)
    st.markdown("---")
    st.subheader("Prompt Sensitivity Index (PSI) — by Model")
    st.caption(
        "Average PSI per model across all questions. PSI = std deviation ÷ mean "
        "of composite scores across the 8 prompt variants per question. "
    )

    def _sensitivity_label(v):
        if v < 0.05:   return "Extremely robust"
        elif v < 0.10: return "Robust"
        elif v < 0.20: return "Moderately sensitive"
        else:          return "Highly sensitive"

    psi_batch_df = pd.DataFrame([
        {"Model": k, "Avg PSI": v, "Sensitivity": _sensitivity_label(v)}
        for k, v in batch["model_psi"].items()
    ])
    st.dataframe(psi_batch_df, use_container_width=True, hide_index=True)

    # Findings Table (Batch)
    st.markdown("---")
    st.subheader("Findings Summary")
    batch_findings_df = pd.DataFrame(generate_batch_findings(batch))
    st.dataframe(batch_findings_df, use_container_width=True, hide_index=True)    

    # Per-question breakdown table
    st.markdown("---")
    st.subheader("Per-Question Results")

    per_q_rows = []
    for r in batch["per_question"]:
        per_q_rows.append({
            "Question":      r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
            "Category":      r["category"],
            "Best Variant":  r["best_variant"],
            "Worst Variant": r["worst_variant"]
        })
    st.dataframe(pd.DataFrame(per_q_rows), use_container_width=True)

    # Export
    st.markdown("---")
    export_rows = []
    for r in batch["per_question"]:
        for style, score in r["variant_scores"].items():
            export_rows.append({
                "question":      r["question"],
                "category":      r["category"],
                "variant_style": style,
                "avg_composite": score,
                "is_winner":     r["best_variant"] == style
            })
    batch_df  = pd.DataFrame(export_rows)
    batch_csv = batch_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Batch Results CSV",
        data=batch_csv,
        file_name="batch_experiment_results.csv",
        mime="text/csv",
        use_container_width=True
    )

elif cache_exists:
    st.info("📁 Cached batch results available. Click the button above to view them.")
else:
    st.info("👆 No batch results yet. Click the button above to run the experiment (~20–30 min, runs once).")