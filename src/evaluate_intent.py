from pathlib import Path
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


GOLDEN_PATH = Path(
    "data/processed/golden_set.csv"
)

OUTPUT_PATH = Path(
    "data/processed/golden_intent_predictions.csv"
)

PROMPT_PATH = Path(
    "configs/labeling_prompt.txt"
)

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

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


def create_client() -> OpenAI:
    load_dotenv()

    api_key = __import__("os").getenv(
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


def load_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def classify(
    client: OpenAI,
    prompt_template: str,
    text: str,
) -> tuple[str, float | None]:

    prompt = (
        prompt_template
        + "\n\nCustomer message:\n"
        + text
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

    result = json.loads(content)

    intent = result.get("intent")
    confidence = result.get("confidence")

    if intent not in VALID_INTENTS:
        raise ValueError(
            f"Invalid intent: {intent}"
        )

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None

    return intent, confidence


def main() -> None:

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_PATH}"
        )

    df = pd.read_csv(
        GOLDEN_PATH
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["gold_intent"] = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["text"] != "")
        & (df["gold_intent"] != "")
    ].copy()

    print(
        f"Golden examples: {len(df):,}"
    )

    client = create_client()
    prompt_template = load_prompt()

    # ---------------------------------------------------------
    # Resume support
    # ---------------------------------------------------------

    if OUTPUT_PATH.exists():

        existing = pd.read_csv(
            OUTPUT_PATH
        )

        existing["tweet_id"] = (
            existing["tweet_id"]
            .astype(str)
            .str.strip()
        )

        completed = set(
            existing["tweet_id"]
        )

    else:

        existing = pd.DataFrame()
        completed = set()

    predictions = []

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        tweet_id = str(
            row["tweet_id"]
        ).strip()

        if tweet_id in completed:
            continue

        print(
            f"[{i}/{len(df)}] "
            f"Classifying {tweet_id}..."
        )

        try:

            predicted, confidence = classify(
                client=client,
                prompt_template=prompt_template,
                text=row["text"],
            )

            predictions.append(
                {
                    "tweet_id": tweet_id,
                    "text": row["text"],
                    "gold_intent": row[
                        "gold_intent"
                    ],
                    "predicted_intent": predicted,
                    "confidence": confidence,
                    "correct": (
                        predicted
                        == row["gold_intent"]
                    ),
                }
            )

        except Exception as exc:

            print(
                f"  ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

        # -----------------------------------------------------
        # Save after every 10 predictions.
        # -----------------------------------------------------

        if len(predictions) >= 10:

            new_df = pd.DataFrame(
                predictions
            )

            if OUTPUT_PATH.exists():

                old_df = pd.read_csv(
                    OUTPUT_PATH
                )

                combined = pd.concat(
                    [
                        old_df,
                        new_df,
                    ],
                    ignore_index=True,
                )

                combined = (
                    combined
                    .drop_duplicates(
                        subset=[
                            "tweet_id"
                        ],
                        keep="last",
                    )
                )

            else:

                combined = new_df

            combined.to_csv(
                OUTPUT_PATH,
                index=False,
            )

            print(
                f"  Checkpoint saved: "
                f"{len(combined):,}"
            )

            predictions = []

    # Save remaining predictions.
    if predictions:

        new_df = pd.DataFrame(
            predictions
        )

        if OUTPUT_PATH.exists():

            old_df = pd.read_csv(
                OUTPUT_PATH
            )

            combined = pd.concat(
                [
                    old_df,
                    new_df,
                ],
                ignore_index=True,
            )

            combined = (
                combined
                .drop_duplicates(
                    subset=["tweet_id"],
                    keep="last",
                )
            )

        else:

            combined = new_df

        combined.to_csv(
            OUTPUT_PATH,
            index=False,
        )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    results = pd.read_csv(
        OUTPUT_PATH
    )

    y_true = results[
        "gold_intent"
    ]

    y_pred = results[
        "predicted_intent"
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        "\n========== GPT-OSS-120B RESULTS =========="
    )

    print(
        f"Evaluated examples: {len(results):,}"
    )

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    print(
        cm_df.to_string()
    )

    print(
        f"\nSaved predictions to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()