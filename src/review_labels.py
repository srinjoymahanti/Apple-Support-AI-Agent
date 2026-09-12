from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/labeled_training_data.csv"
)

OUTPUT_PATH = Path(
    "data/processed/human_verified_training.csv"
)

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

    return df


def print_intents() -> None:
    print("\nChoose the correct intent:\n")

    for number, intent in INTENT_SHORTCUTS.items():
        print(
            f"{number}. {intent}"
        )

    print("s. Skip")
    print("q. Quit and save")


def main() -> None:

    df = load_data()

    # Start with only rows that have an LLM prediction.
    df = df[
        df["intent"] != ""
    ].copy()

    # Human-review columns.
    if "human_intent" not in df.columns:
        df["human_intent"] = ""

    if "human_decision" not in df.columns:
        df["human_decision"] = ""

    if "human_notes" not in df.columns:
        df["human_notes"] = ""

    reviewed = 0

    for index in df.index:

        # Skip already reviewed examples.
        if str(
            df.at[index, "human_decision"]
        ).strip():
            continue

        print("\n" + "=" * 90)

        print(
            f"Example {reviewed + 1}"
        )

        print(
            f"Tweet ID: {df.at[index, 'tweet_id']}"
        )

        print(
            "\nCustomer message:"
        )

        print(
            df.at[index, "text"]
        )

        print(
            "\nLLM prediction:"
        )

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
                f"{reviewed}"
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

        else:

            decision = "incorrect"

        df.at[index, "human_intent"] = (
            selected_intent
        )

        df.at[index, "human_decision"] = (
            decision
        )

        if decision == "correct":

            df.at[index, "human_notes"] = (
                "LLM label confirmed by human review."
            )

        else:

            df.at[index, "human_notes"] = (
                "LLM label corrected by human review."
            )

        reviewed += 1

        # Save after every example.
        save_data(df)

        print(
            f"\nSaved. Human label: "
            f"{selected_intent}"
        )

    save_data(df)

    print(
        "\nAll available examples reviewed."
    )


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


if __name__ == "__main__":
    main()