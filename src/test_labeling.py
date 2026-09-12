from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


INPUT_PATH = Path(
    "data/processed/labeling_sample.csv"
)

PROMPT_PATH = Path(
    "configs/labeling_prompt.txt"
)

TEST_SIZE = 20

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {PROMPT_PATH}"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    df = pd.read_csv(INPUT_PATH)

    # Test only the first 20 messages.
    sample = df.head(TEST_SIZE).copy()

    prompt_template = load_prompt()

    print(
        f"Testing LLM labeling on {len(sample)} messages..."
    )
    print(f"Model: {MODEL}")

    for _, row in sample.iterrows():

        prompt = (
            prompt_template
            + "\n\nCustomer message:\n"
            + str(row["text"])
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                },
            )

            content = response.choices[0].message.content

            print("\n" + "=" * 80)
            print(f"Tweet ID: {row['tweet_id']}")
            print(f"Message: {row['text']}")
            print("\nModel output:")

            try:
                parsed = json.loads(content)

                print(
                    json.dumps(
                        parsed,
                        indent=2,
                    )
                )

            except json.JSONDecodeError:
                print(
                    "Invalid JSON returned by model:"
                )
                print(content)

        except Exception as exc:
            print("\n" + "=" * 80)
            print(
                f"Tweet ID: {row['tweet_id']}"
            )
            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()