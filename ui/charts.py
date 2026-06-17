import plotly.graph_objects as go
import pandas as pd


def variant_bar_chart(variant_summary: dict) -> go.Figure:
    styles = list(variant_summary.keys())
    scores = list(variant_summary.values())
    colors = [
        "#2ecc71" if s == max(scores)
        else "#e74c3c" if s == min(scores)
        else "#3498db"
        for s in scores
    ]
    fig = go.Figure(go.Bar(
        x=scores,
        y=styles,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside"
    ))
    fig.update_layout(
        title="Prompt Style Ranking (avg composite score)",
        xaxis_title="Composite Score",
        xaxis_range=[0, 1],
        height=420,
        margin=dict(l=20, r=40, t=40, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13),
        legend=dict(
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.3)"
        )
    )
    return fig


def model_bar_chart(model_summary: dict) -> go.Figure:
    models = list(model_summary.keys())
    scores = list(model_summary.values())
    fig = go.Figure(go.Bar(
        x=models,
        y=scores,
        marker_color=["#9b59b6", "#e67e22", "#1abc9c"],
        text=[f"{s:.3f}" for s in scores],
        textposition="outside"
    ))
    fig.update_layout(
        title="Model Performance (avg composite score)",
        yaxis_title="Composite Score",
        yaxis_range=[0, 1],
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13),
        legend=dict(
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.3)"
        )
    )
    return fig


def heatmap_chart(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(
        index="variant_style",
        columns="model",
        values="composite",
        aggfunc="mean"
    ).round(3)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=pivot.values.round(3),
        texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(
        title="Score Heatmap: Prompt Style × Model",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13),
        legend=dict(
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.3)"
        )
    )
    return fig


def metric_comparison_chart(df: pd.DataFrame) -> go.Figure:
    summary = df.groupby("variant_style").agg({
        "bertscore":   "mean",
        "consistency": "mean",
        "judge_score": lambda x: round(x.mean() / 10, 4)
    }).reset_index()

    fig = go.Figure()
    metric_colors = {
        "bertscore":   "#3498db",
        "consistency": "#2ecc71",
        "judge_score": "#9b59b6"
    }
    metric_labels = {
        "bertscore":   "BERTScore",
        "consistency": "Consistency",
        "judge_score": "Judge Score (norm)"
    }
    for metric, color in metric_colors.items():
        fig.add_trace(go.Bar(
            name=metric_labels[metric],
            x=summary["variant_style"],
            y=summary[metric],
            marker_color=color
        ))
    fig.update_layout(
        barmode="group",
        title="Metric Breakdown by Prompt Style",
        yaxis_title="Score (0–1)",
        yaxis_range=[0, 1],
        height=420,
        margin=dict(l=20, r=20, t=40, b=80),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13),
        xaxis_tickangle=-30,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.3)"
        )
    )
    return fig


def batch_win_chart(variant_win_counts: dict, total_questions: int) -> go.Figure:
    styles = list(variant_win_counts.keys())
    wins   = list(variant_win_counts.values())
    fig = go.Figure(go.Bar(
        x=wins,
        y=styles,
        orientation="h",
        marker_color="#3498db",
        text=[f"{w}/{total_questions}" for w in wins],
        textposition="outside"
    ))
    fig.update_layout(
        title="Prompt Style Win Counts (Batch Experiment)",
        xaxis_title="Questions Won",
        xaxis_range=[0, total_questions],
        height=400,
        margin=dict(l=20, r=40, t=40, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13)
    )
    return fig


def batch_model_chart(model_avg_scores: dict) -> go.Figure:
    models = list(model_avg_scores.keys())
    scores = list(model_avg_scores.values())
    fig = go.Figure(go.Bar(
        x=models,
        y=scores,
        marker_color=["#9b59b6", "#e67e22", "#1abc9c"],
        text=[f"{s:.3f}" for s in scores],
        textposition="outside"
    ))
    fig.update_layout(
        title="Average Model Performance Across All Questions",
        yaxis_title="Avg Composite Score",
        yaxis_range=[0, 1],
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13)
    )
    return fig


def psi_chart(psi_table: list[dict]) -> go.Figure:
    models   = [row["model"] for row in psi_table]
    psi_vals = [row["psi"]   for row in psi_table]

    colors = [
        "#2ecc71" if v < 0.05 else
        "#f1c40f" if v < 0.10 else
        "#e67e22" if v < 0.20 else
        "#e74c3c"
        for v in psi_vals
    ]

    fig = go.Figure(go.Bar(
        x=models,
        y=psi_vals,
        marker_color=colors,
        text=[f"{v:.3f}" for v in psi_vals],
        textposition="outside"
    ))
    fig.update_layout(
        title="Prompt Sensitivity Index (Lower = More Robust)",
        yaxis_title="PSI (std/mean)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13)
    )
    return fig