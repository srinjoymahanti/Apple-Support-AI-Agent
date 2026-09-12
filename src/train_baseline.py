import joblib

from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


INPUT_PATH = Path(
    "data/processed/human_verified_training.csv"
)

MODEL_OUTPUT = Path(
    "data/processed/tfidf_baseline.joblib"
)

VECTORIZER_OUTPUT = Path(
    "data/processed/tfidf_vectorizer.joblib"
)

RANDOM_STATE = 42
TEST_SIZE = 0.25


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # ---------------------------------------------------------
    # Keep only human-reviewed rows with a final human label.
    # ---------------------------------------------------------

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

    # Use ONLY examples that have been manually verified.
    df["human_verified"] = (
        df["human_verified"]
        .fillna(False)
        .astype(bool)
    )

    df = df[
        (df["human_intent"] != "")
        & (df["text"] != "")
    ].copy()

    print(
        f"Human-labeled examples: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Show class distribution
    # ---------------------------------------------------------

    print(
        "\n========== CLASS DISTRIBUTION =========="
    )

    print(
        df["human_intent"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Check whether every class has enough samples.
    # ---------------------------------------------------------

    class_counts = (
        df["human_intent"]
        .value_counts()
    )

    too_small = class_counts[
        class_counts < 2
    ]

    if not too_small.empty:

        print(
            "\nWARNING:"
        )

        print(
            "The following classes have fewer "
            "than 2 examples:"
        )

        print(
            too_small.to_string()
        )

        print(
            "\nThese classes cannot reliably participate "
            "in a stratified train/test split."
        )

        print(
            "They will be excluded from this baseline run."
        )

        valid_classes = class_counts[
            class_counts >= 2
        ].index

        df = df[
            df["human_intent"].isin(
                valid_classes
            )
        ].copy()

    # ---------------------------------------------------------
    # Features and labels
    # ---------------------------------------------------------

    X = df["text"]
    y = df["human_intent"]

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"\nTraining examples: {len(X_train):,}"
    )

    print(
        f"Testing examples:  {len(X_test):,}"
    )

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        f"\nTF-IDF features: "
        f"{X_train_tfidf.shape[1]:,}"
    )

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    MODEL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

    joblib.dump(
    model,
    MODEL_OUTPUT,
)

    joblib.dump(
    vectorizer,
    VECTORIZER_OUTPUT,
)

    print(
    f"\nSaved model to: {MODEL_OUTPUT}"
)

    print(
    f"Saved vectorizer to: {VECTORIZER_OUTPUT}"
)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test_tfidf
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(
        "\n========== BASELINE RESULTS =========="
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

    # ---------------------------------------------------------
    # Per-intent results
    # ---------------------------------------------------------

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    print(
        cm_df.to_string()
    )


if __name__ == "__main__":
    main()