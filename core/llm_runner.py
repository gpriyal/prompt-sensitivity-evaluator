import os
import time
from groq import Groq
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

GROQ_MODEL_1 = "llama-3.3-70b-versatile"
GROQ_MODEL_2 = "llama-3.1-8b-instant"      
MISTRAL_MODEL = "mistral-small-latest"

SYSTEM_PROMPT = (
    "You are a knowledgeable, helpful assistant. Respond accurately and naturally "
    "to the user's request, following any instructions about tone, depth, reasoning "
    "style, or format given in the prompt itself."
)


def run_groq_model(prompt: str, model: str, label: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,   # was 0.7 - improves run-to-run reproducibility
                max_tokens=1500
            )
            return {
                "model": label,
                "output": response.choices[0].message.content.strip(),
                "error": None
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return {"model": label, "output": None, "error": str(e)}


def run_mistral_model(prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500     
            )
            return {
                "model": "Mistral · Small",
                "output": response.choices[0].message.content.strip(),
                "error": None
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return {"model": "Mistral · Small", "output": None, "error": str(e)}


def run_all_models(prompt: str) -> list[dict]:
    results = []
    results.append(run_groq_model(prompt, GROQ_MODEL_1, "Groq · Llama-3.3-70B"))
    time.sleep(2)
    results.append(run_groq_model(prompt, GROQ_MODEL_2, "Groq · Llama-3.1-8B"))
    time.sleep(2)
    results.append(run_mistral_model(prompt))
    return results


def run_variants_on_all_models(variants: list[dict]) -> list[dict]:
    enriched = []
    for variant in variants:
        outputs = run_all_models(variant["prompt"])
        enriched.append({
            **variant,
            "model_outputs": outputs
        })
        time.sleep(3)
    return enriched