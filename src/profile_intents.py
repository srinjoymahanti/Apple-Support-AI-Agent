from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/applesupport_customers.csv"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Total AppleSupport customer messages: "
        f"{len(df):,}"
    )

    print("\n========== RANDOM SAMPLE ==========")

    sample = df.sample(
        n=min(100, len(df)),
        random_state=42,
    )

    for index, row in sample.iterrows():
        print(
            f"\n[{row['tweet_id']}]"
        )
        print(row["text"])

    print("\n========== MESSAGE LENGTH ==========")

    df["message_length"] = (
        df["text"]
        .astype(str)
        .str.len()
    )

    print(
        df["message_length"].describe()
    )


if __name__ == "__main__":
    main()