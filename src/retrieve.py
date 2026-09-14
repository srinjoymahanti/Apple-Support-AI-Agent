from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(
    "data/processed/retrieval_index_data.csv"
)

VECTORIZER_PATH = Path(
    "data/processed/retrieval_tfidf_vectorizer.joblib"
)

MATRIX_PATH = Path(
    "data/processed/retrieval_tfidf_matrix.joblib"
)


def load_retriever():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data not found: {DATA_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF matrix not found: {MATRIX_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    matrix = joblib.load(
        MATRIX_PATH
    )

    return df, vectorizer, matrix


def retrieve(
    query: str,
    top_k: int = 5,
):
    df, vectorizer, matrix = (
        load_retriever()
    )

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = df.iloc[
        top_indices
    ].copy()

    results[
        "similarity"
    ] = similarities[
        top_indices
    ]

    return results


def main():
    query = input(
        "Enter customer message: "
    ).strip()

    if not query:
        print(
            "Please enter a message."
        )
        return

    results = retrieve(
        query,
        top_k=5,
    )

    print(
        "\n========== TOP HISTORICAL CASES =========="
    )

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        print(
            f"\n--- Result {rank} ---"
        )

        print(
            f"Similarity: "
            f"{row['similarity']:.4f}"
        )

        print(
            "\nCustomer:"
        )

        print(
            row["customer_text"]
        )

        print(
            "\nAppleSupport:"
        )

        print(
            row["brand_text"]
        )


if __name__ == "__main__":
    main()