# PROMPT VARIANT GENERATOR

# Models 8 realistic human framings of the same question,
# each isolating a different prompting dimension:
#   - depth-seeking (Direct vs Detailed)
#   - reasoning elicitation (Chain-of-Thought)
#   - persona framing (Role-Based Expert)
#   - audience adaptation (ELI5)
#   - tone (Polite vs Aggressive)
#   - format constraint (Structured)

# Length guidance is embedded per-variant where the style
# naturally produces long output, to prevent truncation
# without imposing a uniform constraint that flattens
# variance between styles.

# Maps question category to a domain-expert persona
# used in the Role-Based Expert variant
EXPERT_PERSONAS = {
    "Science":  "a research scientist",
    "Coding":   "a senior software engineer",
    "Medical":  "a licensed medical doctor",
    "History":  "a historian specialising in this period",
    "General":  "a subject-matter expert"
}


def generate_variants(base_question: str, category: str = "General") -> list[dict]:
    """
    Returns 8 prompt variants modelling distinct human framings
    of the same base question.
    """

    persona = EXPERT_PERSONAS.get(category, "a subject-matter expert")
    variants = [
        {
            "style": "Direct / Concise",
            "prompt": f"{base_question} Answer directly and concisely."
        },
        {
            "style": "Detailed / Comprehensive",
            "prompt": (
                f"Provide a comprehensive, in-depth explanation of the following, "
                f"covering all major aspects. Keep your response under 300 words. "
                f"Question: {base_question}"
            )
        },
        {
            "style": "Chain-of-Thought",
            "prompt": (
                f"Think through this step by step, showing your reasoning before "
                f"giving your final answer. Keep your full response under 300 words. "
                f"Question: {base_question}"
            )
        },
        {
            "style": "Role-Based Expert",
            "prompt": (
                f"You are {persona}. Drawing on your domain expertise, answer the "
                f"following question with technical precision. Keep your response "
                f"under 300 words. Question: {base_question}"
            )
        },
        {
            "style": "Explain Like I'm 5",
            "prompt": (
                f"Explain this in simple terms, as if to a curious child, using an "
                f"everyday analogy. Question: {base_question}"
            )
        },
        {
            "style": "Polite / Formal",
            "prompt": (
                f"Could you please provide a clear and accurate explanation of the "
                f"following: {base_question}"
            )
        },
        {
            "style": "Aggressive / Demanding",
            "prompt": (
                f"Answer this now. Be direct and give me the facts immediately — "
                f"no disclaimers, no fluff. Question: {base_question}"
            )
        },
        {
            "style": "Structured / Formatted",
            "prompt": (
                f"Present your answer using clear headings and bullet points. "
                f"Keep your response under 300 words. Question: {base_question}"
            )
        }
    ]
    return variants


def get_variant_styles() -> list[str]:
    return [
        "Direct / Concise",
        "Detailed / Comprehensive",
        "Chain-of-Thought",
        "Role-Based Expert",
        "Explain Like I'm 5",
        "Polite / Formal",
        "Aggressive / Demanding",
        "Structured / Formatted"
    ]