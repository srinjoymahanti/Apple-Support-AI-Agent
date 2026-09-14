from pathlib import Path

import pandas as pd


MAIN_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

TARGETED_PATH = Path(
    "data/processed/targeted_candidates_labeled.csv"
)


def main() -> None:
    main = pd.read_csv(MAIN_PATH)
    targeted = pd.read_csv(TARGETED_PATH)

    print(f"Main labels:     {len(main):,}")
    print(f"Targeted labels: {len(targeted):,}")

    main["tweet_id"] = (
        main["tweet_id"]
        .astype(str)
        .str.strip()
    )

    targeted["tweet_id"] = (
        targeted["tweet_id"]
        .astype(str)
        .str.strip()
    )

    combined = pd.concat(
        [main, targeted],
        ignore_index=True,
    )

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["tweet_id"],
        keep="first",
    )

    removed = before - len(combined)

    combined.to_csv(
        MAIN_PATH,
        index=False,
    )

    print(f"\nDuplicates removed: {removed}")
    print(f"Final labeled rows: {len(combined):,}")

    print("\nFinal intent distribution:")
    print(
        combined["intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()