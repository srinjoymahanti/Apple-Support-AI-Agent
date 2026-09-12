from pathlib import Path

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


INPUT_PATH = Path(
    "data/processed/human_review_corrected_all_rows.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["human_intent"] = (
        df["human_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["human_intent"] != "")
        & (df["text"] != "")
    ].copy()

    if df.empty:
        raise ValueError(
            "No human-labeled examples found."
        )

    y = df["human_intent"]

    majority_intent = y.value_counts().idxmax()

    print(
        f"Human-labeled examples: {len(df):,}"
    )

    print(
        f"Majority intent: {majority_intent}"
    )

    predictions = [majority_intent] * len(y)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(
        "\n========== MAJORITY BASELINE =========="
    )

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )


if __name__ == "__main__":
    main()