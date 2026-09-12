from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_support_twitter.csv")
OUTPUT_PATH = Path("data/processed/sample_conversations.txt")

CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
]


def load_data() -> pd.DataFrame:
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

    df["tweet_id"] = pd.to_numeric(
        df["tweet_id"],
        errors="coerce",
    ).astype("Int64")

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce",
    ).astype("Int64")

    return df


def build_lookup(df: pd.DataFrame):
    return df.set_index("tweet_id")


def reconstruct_conversation(
    start_tweet_id,
    lookup,
):
    """
    Follow the response chain backwards from a brand tweet.
    """

    chain = []
    visited = set()

    current_id = start_tweet_id

    while (
        pd.notna(current_id)
        and current_id not in visited
    ):
        visited.add(current_id)

        if current_id not in lookup.index:
            break

        row = lookup.loc[current_id]

        chain.append(
            {
                # tweet_id is the DataFrame index
                "tweet_id": int(row.name),
                "author_id": row["author_id"],
                "inbound": bool(row["inbound"]),
                "created_at": row["created_at"],
                "text": row["text"],
            }
        )

        parent_id = row["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            break

        current_id = parent_id

    # We followed the conversation backwards,
    # so reverse it to chronological order.
    chain.reverse()

    return chain


def has_both_sides(chain):
    has_customer = any(
        item["inbound"]
        for item in chain
    )

    has_brand = any(
        not item["inbound"]
        for item in chain
    )

    return has_customer and has_brand


def format_conversation(
    brand,
    index,
    chain,
):
    lines = []

    lines.append("=" * 80)
    lines.append(
        f"BRAND: {brand} | SAMPLE: {index}"
    )
    lines.append("=" * 80)

    for item in chain:
        speaker = (
            "CUSTOMER"
            if item["inbound"]
            else "BRAND"
        )

        lines.append(
            f"\n[{speaker}] {item['author_id']}"
        )
        lines.append(
            f"Tweet ID: {item['tweet_id']}"
        )
        lines.append(
            f"Time: {item['created_at']}"
        )
        lines.append(
            f"Text: {item['text']}"
        )

    lines.append("")

    return "\n".join(lines)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading dataset...")
    df = load_data()

    print(
        f"Loaded {len(df):,} tweets."
    )

    lookup = build_lookup(df)

    all_output = []

    for brand in CANDIDATES:

        print(
            f"\nFinding conversations for {brand}..."
        )

        # Get tweets sent by this brand.
        brand_tweets = df[
            (df["author_id"] == brand)
            & (df["inbound"] == False)
        ].copy()

        # Random, reproducible sample.
        # We sample more than the required 10 because some
        # sampled tweets may belong to invalid/very long chains.
        sample_size = min(50, len(brand_tweets))

        brand_tweets = brand_tweets.sample(
            n=sample_size,
            random_state=42,
        )

        samples = []
        seen_conversations = set()

        for _, tweet in brand_tweets.iterrows():

            chain = reconstruct_conversation(
                tweet["tweet_id"],
                lookup,
            )

            if not has_both_sides(chain):
                continue

            # Ignore extremely long chains for manual inspection.
            if len(chain) > 20:
                continue

            # Earliest tweet acts as our simple conversation ID.
            conversation_id = chain[0]["tweet_id"]

            # Avoid inspecting the same conversation twice.
            if conversation_id in seen_conversations:
                continue

            seen_conversations.add(
                conversation_id
            )

            samples.append(chain)

            if len(samples) == 10:
                break

        print(
            f"Collected {len(samples)} conversations."
        )

        for i, chain in enumerate(
            samples,
            start=1,
        ):
            all_output.append(
                format_conversation(
                    brand,
                    i,
                    chain,
                )
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(all_output),
        encoding="utf-8",
    )

    print(
        f"\nSaved samples to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()