from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

OUTPUT_PATH = Path(
    "data/processed/human_verified_training.csv"
)

TARGETED_PATH = Path(
    "data/processed/targeted_candidates_labeled.csv"
)

# Maximum number of human-verified examples we want per intent
TARGET_PER_INTENT = 50

INTENTS = [
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
]

INTENT_SHORTCUTS = {
    str(i + 1): intent
    for i, intent in enumerate(INTENTS)
}


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Load previous human-review work if it already exists.
    # ---------------------------------------------------------
    if OUTPUT_PATH.exists():

        previous = pd.read_csv(OUTPUT_PATH)

        # Only merge columns produced by human review.
        review_columns = [
            "human_intent",
            "human_decision",
            "human_notes",
        ]

        available_columns = [
            column
            for column in review_columns
            if column in previous.columns
        ]

        if available_columns and "tweet_id" in previous.columns:

            previous_review = previous[
                ["tweet_id"] + available_columns
            ].copy()

            # Avoid duplicate tweet IDs.
            previous_review = previous_review.drop_duplicates(
                subset=["tweet_id"],
                keep="last",
            )

            df = df.merge(
                previous_review,
                on="tweet_id",
                how="left",
                suffixes=("", "_previous"),
            )

            # Fill existing review information.
            for column in available_columns:

                previous_column = f"{column}_previous"

                if previous_column in df.columns:

                    if column not in df.columns:
                        df[column] = ""

                    df[column] = (
                        df[column]
                        .fillna("")
                        .astype(str)
                    )

                    previous_values = (
                        df[previous_column]
                        .fillna("")
                        .astype(str)
                    )

                    df[column] = df[column].where(
                        df[column].str.strip() != "",
                        previous_values,
                    )

                    df.drop(
                        columns=[previous_column],
                        inplace=True,
                    )

    # Create missing review columns.
    for column in [
        "human_intent",
        "human_decision",
        "human_notes",
    ]:
        if column not in df.columns:
            df[column] = ""

    return df


def print_intents() -> None:

    print("\nChoose the correct intent:\n")

    for number, intent in INTENT_SHORTCUTS.items():

        print(
            f"{number}. {intent}"
        )

    print("s. Skip")
    print("q. Quit and save")


def get_review_counts(df: pd.DataFrame) -> pd.Series:

    reviewed = df[
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ]

    counts = (
        reviewed["human_intent"]
        .value_counts()
    )

    # Ensure all 11 intents are present.
    counts = counts.reindex(
        INTENTS,
        fill_value=0,
    )

    return counts


def build_review_queue(
    df: pd.DataFrame,
) -> list:

    # Only rows with an LLM prediction.
    candidates = df[
        df["intent"] != ""
    ].copy()

    # Only unreviewed rows.
    candidates = candidates[
        candidates["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        == ""
    ].copy()

    review_counts = get_review_counts(df)

    print("\n========== HUMAN REVIEW STATUS ==========\n")

    for intent in INTENTS:

        current = int(
            review_counts.get(intent, 0)
        )

        available = int(
            (
                candidates["intent"]
                == intent
            ).sum()
        )

        remaining = max(
            0,
            TARGET_PER_INTENT - current,
        )

        to_review = min(
            remaining,
            available,
        )

        print(
            f"{intent:25s}"
            f" reviewed={current:3d}"
            f" available={available:3d}"
            f" target={TARGET_PER_INTENT:3d}"
            f" queue={to_review:3d}"
        )

    # ---------------------------------------------------------
    # Build queue by largest deficit first.
    # ---------------------------------------------------------
    queue_parts = []

    deficits = []

    for intent in INTENTS:

        current = int(
            review_counts.get(intent, 0)
        )

        deficit = max(
            0,
            TARGET_PER_INTENT - current,
        )

        available = int(
            (
                candidates["intent"]
                == intent
            ).sum()
        )

        if deficit > 0 and available > 0:

            deficits.append(
                (intent, deficit)
            )

    # Largest deficit first.
    deficits.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    for intent, deficit in deficits:

        intent_rows = candidates[
            candidates["intent"]
            == intent
        ].copy()

        # Deterministic shuffle so review order is reproducible.
        intent_rows = (
            intent_rows
            .sample(
                frac=1,
                random_state=42,
            )
        )

        intent_rows = intent_rows.head(
            deficit
        )

        queue_parts.append(
            intent_rows
        )

    if not queue_parts:
        return []

    queue = pd.concat(
        queue_parts,
        axis=0,
    )

    # Keep the priority order instead of original dataframe order.
    return list(queue.index)


def save_data(
    df: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def main() -> None:

    df = load_data()

    # Make sure only predicted rows are considered.
    df = df[
        df["intent"] != ""
    ].copy()

    queue = build_review_queue(df)

    if not queue:

        print(
            "\nNo examples currently need "
            "human review."
        )

        save_data(df)

        return

    print(
        f"\nTotal examples in review queue: "
        f"{len(queue)}"
    )

    reviewed_this_session = 0

    for index in queue:

        print("\n" + "=" * 90)

        print(
            f"Review {reviewed_this_session + 1}"
            f" / {len(queue)}"
        )

        print(
            f"Tweet ID: "
            f"{df.at[index, 'tweet_id']}"
        )

        print("\nCustomer message:")

        print(
            df.at[index, "text"]
        )

        print("\nLLM prediction:")

        print(
            f"  {df.at[index, 'intent']}"
        )

        print(
            f"LLM confidence: "
            f"{df.at[index, 'label_confidence']}"
        )

        print_intents()

        choice = input(
            "\nYour choice: "
        ).strip().lower()

        if choice == "q":

            save_data(df)

            print(
                "\nReview stopped."
            )

            print(
                f"Reviewed this session: "
                f"{reviewed_this_session}"
            )

            return

        if choice == "s":

            continue

        if choice not in INTENT_SHORTCUTS:

            print(
                "Invalid choice. Try again."
            )

            continue

        selected_intent = (
            INTENT_SHORTCUTS[choice]
        )

        original_intent = (
            df.at[index, "intent"]
        )

        if selected_intent == original_intent:

            decision = "correct"

            notes = (
                "LLM label confirmed by human review."
            )

        else:

            decision = "incorrect"

            notes = (
                "LLM label corrected by human review."
            )

        df.at[index, "human_intent"] = (
            selected_intent
        )

        df.at[index, "human_decision"] = (
            decision
        )

        df.at[index, "human_notes"] = (
            notes
        )

        reviewed_this_session += 1

        # Save after EVERY human label.
        save_data(df)

        print(
            f"\nSaved."
            f"\nLLM:   {original_intent}"
            f"\nHuman: {selected_intent}"
        )

        # Show current counts.
        counts = get_review_counts(df)

        print(
            "\nCurrent verified counts:"
        )

        for intent in INTENTS:

            print(
                f"  {intent:25s}: "
                f"{int(counts[intent])}"
            )

    save_data(df)

    print(
        "\nReview queue completed."
    )

    print(
        f"Reviewed this session: "
        f"{reviewed_this_session}"
    )


if __name__ == "__main__":
    main()