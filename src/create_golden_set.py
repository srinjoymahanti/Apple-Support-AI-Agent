from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_PATH = Path(
    "data/processed/applesupport_customers.csv"
)

TRAINING_PATH = Path(
    "data/processed/human_verified_training.csv"
)

OUTPUT_PATH = Path(
    "data/processed/golden_set.csv"
)

RANDOM_STATE = 42
GOLDEN_SIZE = 200


def main() -> None:

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_PATH}"
        )

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAINING_PATH}"
        )

    source = pd.read_csv(
        SOURCE_PATH
    )

    training = pd.read_csv(
        TRAINING_PATH
    )

    source["tweet_id"] = (
        source["tweet_id"]
        .astype(str)
        .str.strip()
    )

    source["text"] = (
        source["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    training["tweet_id"] = (
        training["tweet_id"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Exclude every example already used for human-verified
    # training.
    # ---------------------------------------------------------

    training_ids = set(
        training[
            training["human_intent"]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ]["tweet_id"]
    )

    candidates = source[
        ~source["tweet_id"].isin(
            training_ids
        )
    ].copy()

    candidates = candidates[
        candidates["text"] != ""
    ].copy()

    print(
        f"Available unseen candidates: "
        f"{len(candidates):,}"
    )

    if len(candidates) < GOLDEN_SIZE:
        raise ValueError(
            "Not enough unseen examples for "
            "the golden set."
        )

    # ---------------------------------------------------------
    # Random, reproducible sample.
    #
    # We do NOT use LLM labels here.
    # These will be human-labeled separately.
    # ---------------------------------------------------------

    golden, _ = train_test_split(
        candidates,
        train_size=GOLDEN_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    golden = golden[
        [
            "tweet_id",
            "author_id",
            "created_at",
            "text",
        ]
    ].copy()

    # Empty columns for independent evaluation.
    golden["gold_intent"] = ""
    golden["gold_notes"] = ""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    golden.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nGolden set created: "
        f"{len(golden):,} examples"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()