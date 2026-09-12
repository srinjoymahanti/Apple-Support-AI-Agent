from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

OUTPUT_PATH = Path(
    "data/processed/human_review.csv"
)

REVIEW_SIZE = 100
RANDOM_STATE = 42


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Loaded {len(df):,} preliminary labels."
    )

    # Prefer lower-confidence examples because they are
    # more informative for taxonomy validation.
    df["confidence_num"] = pd.to_numeric(
        df["label_confidence"],
        errors="coerce",
    )

    low_conf = df[
        df["confidence_num"] < 0.75
    ].copy()

    high_conf = df[
        df["confidence_num"] >= 0.75
    ].copy()

    # Take all available low-confidence examples first.
    review = low_conf.sample(
        n=min(
            len(low_conf),
            REVIEW_SIZE,
        ),
        random_state=RANDOM_STATE,
    )

    remaining = REVIEW_SIZE - len(review)

    if remaining > 0 and len(high_conf) > 0:
        additional = high_conf.sample(
            n=min(
                remaining,
                len(high_conf),
            ),
            random_state=RANDOM_STATE,
        )

        review = pd.concat(
            [
                review,
                additional,
            ],
            ignore_index=True,
        )

    review = review.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # Human reviewer fields.
    review["human_intent"] = ""
    review["human_decision"] = ""
    review["human_notes"] = ""

    review = review[
        [
            "tweet_id",
            "text",
            "intent",
            "label_confidence",
            "human_intent",
            "human_decision",
            "human_notes",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Created human review file with "
        f"{len(review):,} examples."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()