from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_support_twitter.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Download the Kaggle dataset and place the CSV there."
        )

    print("Reading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\n========== BASIC INFORMATION ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head().to_string())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    if "inbound" in df.columns:
        print("\n========== INBOUND / OUTBOUND ==========")
        print(df["inbound"].value_counts(dropna=False))

    print("\n========== UNIQUE AUTHORS ==========")

    if "author_id" in df.columns:
        print(f"Unique authors: {df['author_id'].nunique():,}")
        print("\nTop authors:")
        print(df["author_id"].value_counts().head(20))

    print("\n========== RESPONSE LINKS ==========")

    response_columns = [
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]

    for column in response_columns:
        if column in df.columns:
            non_null = df[column].notna().sum()
            print(
                f"{column}: "
                f"{non_null:,} non-null / {len(df):,} total"
            )


if __name__ == "__main__":
    main()