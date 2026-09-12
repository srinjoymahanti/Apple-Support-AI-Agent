from pathlib import Path

import pandas as pd


LLM_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

CORRECTED_PATH = Path(
    "data/processed/human_review_corrected_all_rows.csv"
)

VERIFIED_PATH = Path(
    "data/processed/human_verified_training.csv"
)


REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
    "intent",
    "label_source",
    "label_confidence",
    "human_intent",
    "human_decision",
    "human_notes",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "human_intent",
        "human_decision",
        "human_notes",
    ]:
        if column not in df.columns:
            df[column] = ""

    df["tweet_id"] = pd.to_numeric(
        df["tweet_id"],
        errors="coerce",
    )

    df["human_intent"] = (
        df["human_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["human_decision"] = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["human_notes"] = (
        df["human_notes"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def main() -> None:

    if not LLM_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {LLM_PATH}"
        )

    if not CORRECTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {CORRECTED_PATH}"
        )

    print("Loading LLM labels...")
    llm_df = clean(
        pd.read_csv(LLM_PATH)
    )

    print(
        f"LLM-labeled rows: {len(llm_df):,}"
    )

    print("Loading human corrections...")
    human_df = clean(
        pd.read_csv(CORRECTED_PATH)
    )

    print(
        f"Human-review rows: {len(human_df):,}"
    )

    # Keep only useful columns from the correction file.
    human_updates = human_df[
        [
            "tweet_id",
            "human_intent",
            "human_decision",
            "human_notes",
        ]
    ].copy()

    human_updates = human_updates[
        human_updates["human_intent"] != ""
    ].copy()

    # One row per tweet ID.
    human_updates = human_updates.drop_duplicates(
        subset=["tweet_id"],
        keep="last",
    )

    # Remove any old human-review values before applying
    # the current authoritative corrections.
    llm_df = llm_df.drop(
        columns=[
            "human_intent",
            "human_decision",
            "human_notes",
        ],
        errors="ignore",
    )

    merged = llm_df.merge(
        human_updates,
        on="tweet_id",
        how="left",
    )

    # Fill unreviewed rows cleanly.
    merged["human_intent"] = (
        merged["human_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    merged["human_decision"] = (
        merged["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    merged["human_notes"] = (
        merged["human_notes"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Mark whether the row has actually been human verified.
    merged["human_verified"] = (
        merged["human_intent"] != ""
    )

    # Remove accidental duplicate tweet IDs.
    merged = merged.drop_duplicates(
        subset=["tweet_id"],
        keep="last",
    )

    # Put columns in a predictable order.
    merged = merged[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
            "intent",
            "label_source",
            "label_confidence",
            "human_intent",
            "human_decision",
            "human_notes",
            "human_verified",
        ]
    ]

    VERIFIED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        VERIFIED_PATH,
        index=False,
    )

    print(
        "\n========== CONSOLIDATION =========="
    )

    print(
        f"Total rows: {len(merged):,}"
    )

    print(
        "Human verified:",
        int(merged["human_verified"].sum()),
    )

    print(
        "Unverified:",
        int((~merged["human_verified"]).sum()),
    )

    print(
        "\nHuman review decisions:"
    )

    print(
        merged.loc[
            merged["human_verified"],
            "human_decision",
        ].value_counts()
    )

    print(
        f"\nSaved to: {VERIFIED_PATH}"
    )


if __name__ == "__main__":
    main()