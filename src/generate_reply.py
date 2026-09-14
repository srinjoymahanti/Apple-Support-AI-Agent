from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


PROMPT_PATH = Path(
    "configs/reply_generation_prompt.txt"
)

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def create_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


def generate_reply(
    client: OpenAI,
    prompt_template: str,
    customer_message: str,
    intent: str,
    retrieved_cases: list[dict],
) -> dict:

    evidence_text = ""

    for i, case in enumerate(
        retrieved_cases,
        start=1,
    ):
        evidence_text += (
            f"\n--- Historical case {i} ---\n"
            f"Customer: {case['customer_text']}\n"
            f"AppleSupport: {case['brand_text']}\n"
            f"Similarity: {case['similarity']:.4f}\n"
        )

    prompt = (
        prompt_template
        + "\n\nCUSTOMER MESSAGE:\n"
        + customer_message
        + "\n\nCLASSIFIED INTENT:\n"
        + intent
        + "\n\nHISTORICAL EVIDENCE:\n"
        + evidence_text
    )

    response = (
        client
        .chat
        .completions
        .create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
            response_format={
                "type": "json_object"
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    result = json.loads(
        content
    )

    return result


def main() -> None:

    client = create_client()

    prompt = load_prompt()

    customer_message = input(
        "Customer message: "
    ).strip()

    if not customer_message:
        print(
            "Message cannot be empty."
        )
        return

    intent = input(
        "Intent: "
    ).strip()

    if not intent:
        print(
            "Intent cannot be empty."
        )
        return

    # Temporary test evidence.
    # We will connect this automatically to retrieve.py
    # in the next step.
    retrieved_cases = []

    result = generate_reply(
        client=client,
        prompt_template=prompt,
        customer_message=customer_message,
        intent=intent,
        retrieved_cases=retrieved_cases,
    )

    print(
        "\n========== GENERATED RESPONSE ==========\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()