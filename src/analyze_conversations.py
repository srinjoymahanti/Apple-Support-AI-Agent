from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_support_twitter.csv")
OUTPUT_PATH = Path("data/processed/conversation_stats.csv")


# These are the strongest candidates from the initial ranking.
CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "AmericanAir",
    "TMobileHelp",
]


def load_data() -> pd.DataFrame:
    print("Loading dataset...")

    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
    )

    # Normalize IDs.
    df["tweet_id"] = pd.to_numeric(
        df["tweet_id"],
        errors="coerce",
    ).astype("Int64")

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce",
    ).astype("Int64")

    return df


def analyze_brand(
    df: pd.DataFrame,
    brand: str,
) -> dict:
    """
    Analyze conversations involving one support account.

    A conversation is approximated by following the response chain
    backwards from a brand reply.
    """

    brand_tweets = df[
        (df["author_id"] == brand)
        & (df["inbound"] == False)
    ].copy()

    if brand_tweets.empty:
        return {
            "brand": brand,
            "brand_tweets": 0,
            "conversations": 0,
        }

    # Fast lookup: tweet_id -> row
    tweet_lookup = (
        df.set_index("tweet_id")
    )

    conversation_lengths = []
    customer_message_counts = []
    brand_message_counts = []

    resolved_like_conversations = 0

    for _, brand_tweet in brand_tweets.iterrows():

        current_id = brand_tweet["tweet_id"]

        visited = set()

        customer_count = 0
        brand_count = 0
        total_count = 0

        while (
            pd.notna(current_id)
            and current_id not in visited
        ):
            visited.add(current_id)

            if current_id not in tweet_lookup.index:
                break

            row = tweet_lookup.loc[current_id]

            total_count += 1

            if row["inbound"]:
                customer_count += 1
            else:
                brand_count += 1

            parent_id = row["in_response_to_tweet_id"]

            if pd.isna(parent_id):
                break

            current_id = parent_id

        # Only count chains containing both sides.
        if customer_count > 0 and brand_count > 0:

            conversation_lengths.append(total_count)
            customer_message_counts.append(customer_count)
            brand_message_counts.append(brand_count)

            # A simple initial "resolved-like" heuristic:
            #
            # conversation has both customer and brand messages
            # and contains at least one brand response.
            if brand_count >= 1:
                resolved_like_conversations += 1

    if not conversation_lengths:
        return {
            "brand": brand,
            "brand_tweets": len(brand_tweets),
            "conversations": 0,
        }

    lengths = pd.Series(conversation_lengths)

    return {
        "brand": brand,
        "brand_tweets": len(brand_tweets),
        "conversations": len(conversation_lengths),
        "avg_conversation_length": lengths.mean(),
        "median_conversation_length": lengths.median(),
        "max_conversation_length": lengths.max(),
        "avg_customer_messages": pd.Series(
            customer_message_counts
        ).mean(),
        "avg_brand_messages": pd.Series(
            brand_message_counts
        ).mean(),
        "resolved_like_conversations": resolved_like_conversations,
    }


def main() -> None:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = load_data()

    print(
        f"Loaded {len(df):,} tweets."
    )

    results = []

    for brand in CANDIDATES:

        print(
            f"\nAnalyzing {brand}..."
        )

        stats = analyze_brand(
            df,
            brand,
        )

        results.append(stats)

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n========== CONVERSATION ANALYSIS =========="
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()