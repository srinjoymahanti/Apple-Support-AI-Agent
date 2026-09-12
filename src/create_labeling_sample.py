from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/applesupport_customers.csv"
)

OUTPUT_PATH = Path(
    "data/processed/labeling_sample.csv"
)

SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    print("Loading AppleSupport messages...")

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Available messages: {len(df):,}"
    )

    # Remove empty messages.
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[df["text"] != ""]

    # Reproducible random sample.
    sample_size = min(
        SAMPLE_SIZE,
        len(df),
    )

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    # Reset index for a clean dataset.
    sample = sample.reset_index(
        drop=True
    )

    # Add fields that we will fill during labeling.
    sample["intent"] = ""
    sample["label_source"] = ""
    sample["label_confidence"] = ""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nCreated {len(sample):,} examples."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("\nColumns:")
    print(
        sample.columns.tolist()
    )


if __name__ == "__main__":
    main()