from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/labeling_sample.csv"
)

LABELED_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

OUTPUT_PATH = Path(
    "data/processed/targeted_candidates.csv"
)


TARGETS = {
    "activation_setup": [
        "activate",
        "activation",
        "activate my iphone",
        "activation lock",
        "set up",
        "setup",
        "setting up",
        "new iphone",
        "restore",
        "initial setup",
    ],
    "account_security": [
        "apple id",
        "password",
        "account",
        "hacked",
        "hack",
        "security",
        "fraud",
        "stolen",
        "lost iphone",
        "icloud",
        "verify",
        "verification",
        "login",
        "sign in",
    ],
    "accessories": [
        "charger",
        "charging cable",
        "lightning cable",
        "usb cable",
        "airpods",
        "earpods",
        "headphones",
        "adapter",
        "case",
        "keyboard",
        "beats",
        "thunderbolt",
        "accessory",
        "accessories",
    ],
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    if not LABELED_PATH.exists():
        raise FileNotFoundError(
            f"Labeled file not found: {LABELED_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    labeled = pd.read_csv(
        LABELED_PATH
    )

    # ---------------------------------------------------------
    # Normalize IDs
    # ---------------------------------------------------------

    df["tweet_id"] = (
        df["tweet_id"]
        .astype(str)
        .str.strip()
    )

    labeled["tweet_id"] = (
        labeled["tweet_id"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Remove already-labeled messages
    # ---------------------------------------------------------

    completed_ids = set(
        labeled["tweet_id"]
    )

    candidates = df[
        ~df["tweet_id"].isin(
            completed_ids
        )
    ].copy()

    candidates["text"] = (
        candidates["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Find keyword matches
    # ---------------------------------------------------------

    for target_intent, keywords in TARGETS.items():

        pattern = "|".join(
            map(
                lambda x: x.lower(),
                keywords,
            )
        )

        mask = candidates["text"].str.lower().str.contains(
            pattern,
            regex=True,
            na=False,
        )

        matched = candidates[
            mask
        ].copy()

        matched[
            "target_intent"
        ] = target_intent

        matched[
            "matched_target"
        ] = target_intent

        # Keep only a manageable number per target.
        matched = matched.head(50)

        if target_intent == "activation_setup":
            activation = matched

        elif target_intent == "account_security":
            security = matched

        elif target_intent == "accessories":
            accessories = matched

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    result = pd.concat(
        [
            activation,
            security,
            accessories,
        ],
        ignore_index=True,
    )

    # Remove duplicate tweets.
    result = result.drop_duplicates(
        subset=["tweet_id"]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Targeted candidates saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nTotal candidates: "
        f"{len(result):,}"
    )

    print(
        "\nCandidate distribution:"
    )

    print(
        result[
            "target_intent"
        ]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()