from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/golden_set.csv"
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


def save(df: pd.DataFrame) -> None:
    df.to_csv(
        INPUT_PATH,
        index=False,
    )


def print_intents() -> None:

    print("\nChoose the correct intent:\n")

    for number, intent in INTENT_SHORTCUTS.items():
        print(
            f"{number}. {intent}"
        )

    print("s. Skip")
    print("q. Quit and save")


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "gold_intent" not in df.columns:
        df["gold_intent"] = ""

    if "gold_notes" not in df.columns:
        df["gold_notes"] = ""

    df["gold_intent"] = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["gold_notes"] = (
        df["gold_notes"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviewed = 0

    for index in df.index:

       # Resume support.
        # Only skip rows that have a real gold label.
        gold_intent = df.at[index, "gold_intent"]

        if pd.notna(gold_intent) and str(gold_intent).strip():
            continue

        print("\n" + "=" * 90)

        print(
            f"Golden example "
            f"{reviewed + 1}"
        )

        print(
            f"Tweet ID: "
            f"{df.at[index, 'tweet_id']}"
        )

        print(
            "\nCustomer message:"
        )

        print(
            df.at[index, "text"]
        )

        print_intents()

        choice = input(
            "\nYour choice: "
        ).strip().lower()

        if choice == "q":

            save(df)

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
                "Invalid choice."
            )

            continue

        gold_intent = (
            INTENT_SHORTCUTS[choice]
        )

        df.at[index, "gold_intent"] = (
            gold_intent
        )

        df.at[index, "gold_notes"] = (
            "Independently human-labeled."
        )

        reviewed += 1

        # Save after every example.
        save(df)

        print(
            f"\nSaved gold label: "
            f"{gold_intent}"
        )

    save(df)

    print(
        "\nGolden-set review completed."
    )

    print(
        f"Reviewed this session: "
        f"{reviewed}"
    )

    print(
        "\nCurrent gold-label distribution:"
    )

    print(
        df[
            df["gold_intent"]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ]["gold_intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()