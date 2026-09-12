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

OUTPUT_PATH = Path(
    "data/processed/taxonomy_validation.csv"
)

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

TEST_SIZE = 50

VALID_INTENTS = {
    "battery_power",
    "ios_software_issue",
    "app_service_issue",
    "keyboard_input_issue",
    "device_hardware_issue",
    "account_security",
    "activation_setup",
    "feature_how_to",
    "purchase_order",
    "accessories",
    "other_unclear",
}


def classify_message(
    client: OpenAI,
    prompt: str,
    message: str,
) -> dict:

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    prompt
                    + "\n\nCustomer message:\n"
                    + message
                ),
            }
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        },
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    result = json.loads(content)

    intent = result.get("intent")
    confidence = result.get("confidence")

    if intent not in VALID_INTENTS:
        raise ValueError(
            f"Invalid intent returned: {intent}"
        )

    try:
        confidence = float(confidence)
    except (
        TypeError,
        ValueError,
    ):
        confidence = None

    return {
        "intent": intent,
        "confidence": confidence,
    }


def main() -> None:

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(PROMPT_PATH)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    prompt = PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    df = pd.read_csv(INPUT_PATH)

    # Reproducible sample.
    sample = df.sample(
        n=min(TEST_SIZE, len(df)),
        random_state=123,
    ).copy()

    results = []

    print(
        f"Validating taxonomy on "
        f"{len(sample)} messages..."
    )

    for i, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        message = str(row["text"])

        print(
            f"[{i}/{len(sample)}] "
            f"Tweet {row['tweet_id']}"
        )

        try:

            prediction = classify_message(
                client,
                prompt,
                message,
            )

            results.append(
                {
                    "tweet_id": row["tweet_id"],
                    "text": message,
                    "predicted_intent": (
                        prediction["intent"]
                    ),
                    "confidence": (
                        prediction["confidence"]
                    ),
                }
            )

        except Exception as exc:

            print(
                f"  ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "tweet_id": row["tweet_id"],
                    "text": message,
                    "predicted_intent": "ERROR",
                    "confidence": None,
                }
            )

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n========== PREDICTION DISTRIBUTION =========="
    )

    print(
        result_df[
            "predicted_intent"
        ].value_counts()
    )

    print(
        "\n========== ALL PREDICTIONS =========="
    )

    for _, row in result_df.iterrows():

        print("\n" + "-" * 80)
        print(
            f"Tweet ID: {row['tweet_id']}"
        )
        print(
            f"Intent: {row['predicted_intent']}"
        )
        print(
            f"Confidence: {row['confidence']}"
        )
        print(
            f"Text: {row['text']}"
        )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()