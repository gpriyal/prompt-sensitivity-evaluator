# Prompt Sensitivity Evaluator

A research tool that measures how prompt phrasing affects LLM output quality across multiple models — empirically, not anecdotally.

## What It Does
- Generates 8 stylistically distinct prompt variants from any input question
- Runs each variant through 3 models across 2 providers
- Scores every output on semantic quality, consistency, and judge rating
- Computes a Prompt Sensitivity Index (PSI) to quantify model robustness
- Runs batch experiments across 10 questions and surfaces cross-question findings

## Models
| Model | Provider |
|---|---|
| Llama-3.3-70B | Groq |
| Llama-3.1-8B | Groq |
| Mistral-Small | Mistral AI |

## Prompt Variant Taxonomy
| Style | What It Tests |
|---|---|
| Direct / Concise | Baseline quality with no framing |
| Detailed / Comprehensive | Whether requesting depth improves quality |
| Chain-of-Thought | Whether explicit reasoning helps accuracy |
| Role-Based Expert | Whether persona framing changes depth |
| Explain Like I'm 5 | Whether simplification sacrifices accuracy |
| Polite / Formal | Whether politeness affects output |
| Aggressive / Demanding | Whether commanding tone triggers hedging |
| Structured / Formatted | Whether format constraints affect content |

## Evaluation Metrics
| Metric | Method | Weight |
|---|---|---|
| Judge Score | Llama-3.3-70B rates accuracy, clarity, completeness (1–10) | 50% |
| BERTScore | RoBERTa-large F1 against frontier model reference | 30% |
| Consistency | Mean pairwise cosine similarity across variants | 20% |

**Prompt Sensitivity Index (PSI)** = std deviation ÷ mean of composite 
scores across all 8 variants per model. Lower PSI = more robust model.

| PSI Range | Interpretation |
|---|---|
| < 0.05 | Extremely robust |
| 0.05 – 0.10 | Robust |
| 0.10 – 0.20 | Moderately sensitive |
| > 0.20 | Highly sensitive |

## Tech Stack
Python · Streamlit · Groq API · Mistral API · 
bert-score · sentence-transformers · Plotly · Pandas
