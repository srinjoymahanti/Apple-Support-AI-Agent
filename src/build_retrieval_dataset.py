from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/apple_support_response_pairs.csv"
)

OUTPUT_PATH = Path(
    "data/processed/apple_support_retrieval.csv"
)


USEFUL_RESPONSE_TERMS = [
    "go to",
    "settings",
    "select",
    "tap",
    "open",
    "check",
    "restart",
    "reset",
    "update",
    "install",
    "remove",
    "turn on",
    "turn off",
    "enable",
    "disable",
    "follow",
    "steps",
    "try",
    "make sure",
    "press",
    "hold",
    "connect",
    "reconnect",
    "restore",
    "backup",
    "download",
    "delete",
    "verify",
    "change",
]


GENERIC_RESPONSE_PATTERNS = [
    "send us a dm",
    "send us a direct message",
    "dm us",
    "we'll continue there",
    "we'd like to look into this with you",
    "we're here for you",
    "we'd be happy to help",
    "contact us",
    "reach out to us",
    "fill us in",
]


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Input pairs: {len(df):,}"
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
    # Basic quality filters
    # ---------------------------------------------------------

    df = df[
        (df["customer_text"].str.len() >= 10)
        & (df["brand_text"].str.len() >= 20)
    ].copy()

    # ---------------------------------------------------------
    # Remove generic/routing-only responses
    # ---------------------------------------------------------

    brand_lower = (
        df["brand_text"]
        .str.lower()
    )

    generic_mask = brand_lower.apply(
        lambda text: any(
            pattern in text
            for pattern in GENERIC_RESPONSE_PATTERNS
        )
    )

    df = df[
        ~generic_mask
    ].copy()

    # ---------------------------------------------------------
    # Keep responses that contain some actionable signal
    # ---------------------------------------------------------

    actionable_mask = brand_lower.apply(
        lambda text: any(
            term in text
            for term in USEFUL_RESPONSE_TERMS
        )
    )

    df["actionable_response"] = (
        actionable_mask
    )

    retrieval_df = df[
        df["actionable_response"]
    ].copy()

    # ---------------------------------------------------------
    # Remove obvious duplicate response text
    # ---------------------------------------------------------

    retrieval_df = (
        retrieval_df
        .drop_duplicates(
            subset=[
                "customer_text",
                "brand_text",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Add useful metadata
    # ---------------------------------------------------------

    retrieval_df[
        "customer_text_length"
    ] = retrieval_df[
        "customer_text"
    ].str.len()

    retrieval_df[
        "brand_text_length"
    ] = retrieval_df[
        "brand_text"
    ].str.len()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    retrieval_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nUseful retrieval pairs: "
        f"{len(retrieval_df):,}"
    )

    print(
        f"Filtered out: "
        f"{len(df) - len(retrieval_df):,}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nSample responses:"
    )

    print(
        retrieval_df[
            [
                "customer_text",
                "brand_text",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()