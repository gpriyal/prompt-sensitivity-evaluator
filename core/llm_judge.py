import os
import time
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_SYSTEM_PROMPT = """You are a strict and impartial AI output evaluator.
You will be given a question and an AI-generated answer.
Score the answer on these 3 criteria, each from 1 to 10:

1. accuracy    - Is the information factually correct and trustworthy?
2. clarity     - Is the answer well-structured, easy to understand, and coherent?
3. completeness - Does it fully address the question without missing key aspects?

Respond ONLY with a JSON object in exactly this format, no extra text:
{"accuracy": <int>, "clarity": <int>, "completeness": <int>}"""


def judge_output(question: str, output: str, retries: int = 3) -> dict:
    """
    Returns a dict with accuracy, clarity, completeness scores (1-10 each)
    and a computed average judge_score. Returns zeros on failure.
    """
    if not output or len(output.strip()) < 10:
        return {
            "accuracy":     0,
            "clarity":      0,
            "completeness": 0,
            "judge_score":  0.0
        }

    user_message = f"Question: {question}\n\nAnswer to evaluate:\n{output}"

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                temperature=0.0,    # zero temp for deterministic judging
                max_tokens=100
            )

            raw = response.choices[0].message.content.strip()

            # safely parse JSON - strip markdown fences if model adds them
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            accuracy     = int(parsed.get("accuracy",     0))
            clarity      = int(parsed.get("clarity",      0))
            completeness = int(parsed.get("completeness", 0))
            judge_score  = round((accuracy + clarity + completeness) / 3, 4)

            return {
                "accuracy":     accuracy,
                "clarity":      clarity,
                "completeness": completeness,
                "judge_score":  judge_score   # normalise to 0-1 for aggregator
            }

        except (json.JSONDecodeError, KeyError):
            # model didn't return valid JSON — retry
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {
                    "accuracy":     0,
                    "clarity":      0,
                    "completeness": 0,
                    "judge_score":  0.0
                }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return {
                    "accuracy":     0,
                    "clarity":      0,
                    "completeness": 0,
                    "judge_score":  0.0
                }


def judge_all(scored_variants: list[dict], base_question: str) -> list[dict]:
    """
    Adds judge scores to every model output in every variant.
    Adds a rate-limit buffer between calls.
    """
    judged_variants = []

    for variant in scored_variants:
        judged_outputs = []
        for model_out in variant["model_outputs"]:
            output_text = model_out["output"] or ""
            judge_result = judge_output(base_question, output_text)
            time.sleep(0.8)   # rate limit buffer

            judged_outputs.append({
                **model_out,
                "scores": {
                    **model_out.get("scores", {}),
                    **judge_result    # merges into existing scores dict
                }
            })

        judged_variants.append({
            **variant,
            "model_outputs": judged_outputs
        })

    return judged_variants