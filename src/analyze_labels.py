from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/labeled_training_data.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Total labeled examples: {len(df):,}"
    )

    print(
        "\n========== INTENT DISTRIBUTION =========="
    )

    counts = (
        df["intent"]
        .value_counts()
    )

    percentages = (
        df["intent"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    distribution = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print(
        distribution.to_string()
    )

    print(
        "\n========== CONFIDENCE =========="
    )

    print(
        df["label_confidence"]
        .describe()
    )

    print(
        "\n========== LOW CONFIDENCE EXAMPLES =========="
    )

    low_confidence = df[
        pd.to_numeric(
            df["label_confidence"],
            errors="coerce",
        ) < 0.7
    ]

    for _, row in (
        low_confidence
        .head(30)
        .iterrows()
    ):
        print("\n" + "-" * 80)
        print(
            f"Tweet ID: {row['tweet_id']}"
        )
        print(
            f"Intent: {row['intent']}"
        )
        print(
            f"Confidence: {row['label_confidence']}"
        )
        print(
            f"Text: {row['text']}"
        )


if __name__ == "__main__":
    main()