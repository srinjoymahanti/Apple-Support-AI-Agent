from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_support_twitter.csv")
OUTPUT_PATH = Path(
    "data/processed/applesupport_customers.csv"
)

BRAND = "AppleSupport"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )

    # Brand tweets
    brand_tweets = df[
        (df["author_id"] == BRAND)
        & (df["inbound"] == False)
    ][
        [
            "tweet_id",
            "in_response_to_tweet_id",
        ]
    ].copy()

    # Customer tweets
    customer_tweets = df[
        df["inbound"] == True
    ].copy()

    # Normalize IDs
    brand_tweets["in_response_to_tweet_id"] = (
        pd.to_numeric(
            brand_tweets["in_response_to_tweet_id"],
            errors="coerce",
        ).astype("Int64")
    )

    customer_tweets["tweet_id"] = (
        pd.to_numeric(
            customer_tweets["tweet_id"],
            errors="coerce",
        ).astype("Int64")
    )

    # Find customers whose tweet received a direct AppleSupport reply.
    apple_customer_ids = set(
        brand_tweets["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
    )

    customer_tweets = customer_tweets[
        customer_tweets["tweet_id"].isin(
            apple_customer_ids
        )
    ].copy()

    # Remove empty messages.
    customer_tweets["text"] = (
        customer_tweets["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    customer_tweets = customer_tweets[
        customer_tweets["text"] != ""
    ]

    # Remove duplicates.
    customer_tweets = customer_tweets.drop_duplicates(
        subset=["tweet_id"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_tweets.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nAppleSupport customer messages: "
        f"{len(customer_tweets):,}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()