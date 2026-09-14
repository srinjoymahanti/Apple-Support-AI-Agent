from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_PATH = Path(
    "data/processed/apple_support_retrieval.csv"
)

VECTORIZER_PATH = Path(
    "data/processed/retrieval_tfidf_vectorizer.joblib"
)

MATRIX_PATH = Path(
    "data/processed/retrieval_tfidf_matrix.joblib"
)

DATA_PATH = Path(
    "data/processed/retrieval_index_data.csv"
)


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["brand_text"] = (
        df["brand_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Create retrieval text from the customer's problem.
    # ---------------------------------------------------------

    retrieval_text = (
        df["customer_text"]
        .str.lower()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        retrieval_text
    )

    print(
        f"Retrieval documents: {len(df):,}"
    )

    print(
        f"TF-IDF features: {matrix.shape[1]:,}"
    )

    # ---------------------------------------------------------
    # Save everything needed for retrieval.
    # ---------------------------------------------------------

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        DATA_PATH,
        index=False,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    joblib.dump(
        matrix,
        MATRIX_PATH,
    )

    print(
        f"Saved data:       {DATA_PATH}"
    )

    print(
        f"Saved vectorizer: {VECTORIZER_PATH}"
    )

    print(
        f"Saved matrix:     {MATRIX_PATH}"
    )


if __name__ == "__main__":
    main()