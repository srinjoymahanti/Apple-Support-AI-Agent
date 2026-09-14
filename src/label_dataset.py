from pathlib import Path
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError


INPUT_PATH = Path(
    "data/processed/labeling_sample.csv"
)

PROMPT_PATH = Path(
    "configs/labeling_prompt.txt"
)

OUTPUT_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

BATCH_SIZE = 20
MAX_RETRIES = 5

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


def load_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # Make sure our columns exist.
    required_columns = {
        "tweet_id",
        "text",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


def load_existing_results() -> pd.DataFrame | None:
    if not OUTPUT_PATH.exists():
        return None

    print(
        f"Existing output found: {OUTPUT_PATH}"
    )

    return pd.read_csv(
        OUTPUT_PATH
    )


def save_results(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def classify_message(
    client: OpenAI,
    prompt_template: str,
    message: str,
) -> dict:

    prompt = (
        prompt_template
        + "\n\nCustomer message:\n"
        + message
    )

    for attempt in range(MAX_RETRIES):

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

            content = (
                response
                .choices[0]
                .message
                .content
            )

            result = json.loads(content)

            intent = result.get(
                "intent"
            )

            confidence = result.get(
                "confidence"
            )

            if intent not in VALID_INTENTS:
                raise ValueError(
                    f"Invalid intent: {intent}"
                )

            try:
                confidence = float(
                    confidence
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = None

            return {
                "intent": intent,
                "confidence": confidence,
            }

        except RateLimitError as exc:

            wait_time = min(
                60,
                2 ** attempt,
            )

            print(
                f"\nRate limit hit. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

        except json.JSONDecodeError as exc:

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Invalid JSON after "
                    f"{MAX_RETRIES} attempts: "
                    f"{exc}"
                )

            time.sleep(1)

        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(1)

    raise RuntimeError(
        "Classification failed."
    )


def main() -> None:

    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    prompt_template = load_prompt()

    df = load_dataset()

    # ---------------------------------------------------------
    # Resume support
    # ---------------------------------------------------------

    existing = load_existing_results()

    if existing is not None:

        print(
            f"Existing labeled rows: "
            f"{len(existing):,}"
        )

        completed_ids = set(
            existing["tweet_id"]
            .astype(str)
        )

        df["tweet_id_str"] = (
            df["tweet_id"]
            .astype(str)
        )

        remaining = df[
            ~df["tweet_id_str"].isin(
                completed_ids
            )
        ].copy()

        # Preserve the original columns.
        df = remaining.drop(
            columns=["tweet_id_str"]
        )

    print(
        f"\nMessages remaining: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------
    # Start labeling
    # ---------------------------------------------------------

    results = []

    for start in range(
        0,
        len(df),
        BATCH_SIZE,
    ):

        batch = df.iloc[
            start:start + BATCH_SIZE
        ]

        print(
            f"\nProcessing "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(df)}"
        )

        for _, row in batch.iterrows():

            tweet_id = row[
                "tweet_id"
            ]

            message = str(
                row["text"]
            )

            print(
                f"  Labeling {tweet_id}..."
            )

            try:

                prediction = classify_message(
                    client,
                    prompt_template,
                    message,
                )

                result = row.to_dict()

                result[
                    "intent"
                ] = prediction["intent"]

                result[
                    "label_confidence"
                ] = prediction["confidence"]

                result[
                    "label_source"
                ] = "llm_assisted"

                results.append(
                    result
                )

            except Exception as exc:

                print(
                    f"  ERROR for {tweet_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # -----------------------------------------------------
        # Save after every batch
        # -----------------------------------------------------

        if results:

            new_results = pd.DataFrame(
                results
            )

            if OUTPUT_PATH.exists():

                previous = pd.read_csv(
                    OUTPUT_PATH
                )

                combined = pd.concat(
                    [
                        previous,
                        new_results,
                    ],
                    ignore_index=True,
                )

                combined = combined.drop_duplicates(
                    subset=["tweet_id"],
                    keep="last",
                )

                save_results(
                    combined
                )

            else:

                save_results(
                    new_results
                )

            results = []

            print(
                "  Progress saved."
            )

        # Small pause between batches.
        time.sleep(1)

    print(
        "\nLabeling complete."
    )

    if OUTPUT_PATH.exists():

        final_df = pd.read_csv(
            OUTPUT_PATH
        )

        print(
            f"Total labeled rows: "
            f"{len(final_df):,}"
        )

        print(
            "\nIntent distribution:"
        )

        print(
            final_df["intent"]
            .value_counts()
        )


if __name__ == "__main__":
    main()