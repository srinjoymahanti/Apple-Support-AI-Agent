from pathlib import Path
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError


INPUT_PATH = Path(
    "data/processed/targeted_candidates.csv"
)

PROMPT_PATH = Path(
    "configs/labeling_prompt.txt"
)

OUTPUT_PATH = Path(
    "data/processed/targeted_candidates_labeled.csv"
)

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

SAVE_EVERY = 10
MAX_RETRIES = 6

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


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "tweet_id",
        "text",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["tweet_id"] = (
        df["tweet_id"]
        .astype(str)
        .str.strip()
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def save_results(
    df: pd.DataFrame
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def load_existing() -> pd.DataFrame | None:

    if not OUTPUT_PATH.exists():
        return None

    existing = pd.read_csv(
        OUTPUT_PATH
    )

    existing["tweet_id"] = (
        existing["tweet_id"]
        .astype(str)
        .str.strip()
    )

    return existing


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

    for attempt in range(
        MAX_RETRIES
    ):

        try:

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

        except RateLimitError:

            wait_time = min(
                60,
                2 ** attempt,
            )

            print(
                f"    Rate limit hit. "
                f"Waiting {wait_time}s..."
            )

            time.sleep(
                wait_time
            )

        except json.JSONDecodeError:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            time.sleep(2)

        except Exception:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            time.sleep(2)

    raise RuntimeError(
        "Classification failed after retries."
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

    df = load_data()

    existing = load_existing()

    if existing is not None:

        print(
            "Existing targeted labels:"
            f" {len(existing):,}"
        )

        completed_ids = set(
            existing["tweet_id"]
        )

        df = df[
            ~df["tweet_id"].isin(
                completed_ids
            )
        ].copy()

    print(
        f"Targeted messages remaining: "
        f"{len(df):,}"
    )

    if df.empty:

        print(
            "Nothing left to label."
        )

        return

    results = []

    for row_number, (
        index,
        row,
    ) in enumerate(
        df.iterrows(),
        start=1,
    ):

        tweet_id = row[
            "tweet_id"
        ]

        message = row[
            "text"
        ]

        print(
            f"[{row_number}/{len(df)}] "
            f"Labeling {tweet_id}..."
        )

        try:

            prediction = (
                classify_message(
                    client,
                    prompt_template,
                    message,
                )
            )

            result = row.to_dict()

            result["intent"] = (
                prediction["intent"]
            )

            result[
                "label_confidence"
            ] = prediction[
                "confidence"
            ]

            result[
                "label_source"
            ] = "llm_targeted"

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"    ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        # Save every SAVE_EVERY examples.
        if (
            len(results)
            >= SAVE_EVERY
        ):

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

                combined = (
                    new_results
                )

            save_results(
                combined
            )

            print(
                f"    Checkpoint saved: "
                f"{len(combined):,}"
            )

            results = []

    # Save anything left.
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

            combined = new_results

        save_results(
            combined
        )

    print(
        "\nTargeted labeling complete."
    )

    final_df = pd.read_csv(
        OUTPUT_PATH
    )

    print(
        f"Total targeted labels: "
        f"{len(final_df):,}"
    )

    print(
        "\nIntent distribution:"
    )

    print(
        final_df["intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()