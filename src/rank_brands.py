from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_support_twitter.csv")
OUTPUT_PATH = Path("data/processed/brand_ranking.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    print("Loading dataset...")

    # We only need these columns for brand ranking.
    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "in_response_to_tweet_id",
        "text",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
    )

    print(f"Total tweets: {len(df):,}")

    # ---------------------------------------------------------
    # Separate customer and brand tweets
    # ---------------------------------------------------------

    customer = df[df["inbound"] == True].copy()
    brand = df[df["inbound"] == False].copy()

    print(f"Customer tweets: {len(customer):,}")
    print(f"Brand tweets:    {len(brand):,}")

    # ---------------------------------------------------------
    # Normalize response IDs
    #
    # tweet_id is integer while in_response_to_tweet_id is
    # often read as float because of missing values.
    # ---------------------------------------------------------

    customer["in_response_to_tweet_id"] = (
        pd.to_numeric(
            customer["in_response_to_tweet_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    brand["in_response_to_tweet_id"] = (
        pd.to_numeric(
            brand["in_response_to_tweet_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    # ---------------------------------------------------------
    # 1. Brand replies directly to customer
    #
    # brand.in_response_to_tweet_id
    #       ↓
    # customer.tweet_id
    # ---------------------------------------------------------

    customer_lookup = customer[
        ["tweet_id"]
    ].rename(
        columns={
            "tweet_id": "customer_tweet_id"
        }
    )

    brand_replies = brand.merge(
        customer_lookup,
        left_on="in_response_to_tweet_id",
        right_on="customer_tweet_id",
        how="inner",
    )

    print(
        f"\nDirect brand → customer replies: "
        f"{len(brand_replies):,}"
    )

    # ---------------------------------------------------------
    # 2. Customer replies directly to brand
    #
    # customer.in_response_to_tweet_id
    #       ↓
    # brand.tweet_id
    # ---------------------------------------------------------

    brand_lookup = brand[
        ["tweet_id", "author_id"]
    ].rename(
        columns={
            "tweet_id": "brand_tweet_id",
            "author_id": "brand_account",
        }
    )

    customer_replies = customer.merge(
        brand_lookup,
        left_on="in_response_to_tweet_id",
        right_on="brand_tweet_id",
        how="inner",
    )

    print(
        f"Direct customer → brand replies: "
        f"{len(customer_replies):,}"
    )

    # ---------------------------------------------------------
    # 3. Aggregate by brand
    # ---------------------------------------------------------

    replies_to_customers = (
        brand_replies.groupby("author_id")
        .agg(
            brand_replies=("tweet_id", "count"),
            unique_customers_replied_to=(
                "customer_tweet_id",
                "nunique",
            ),
        )
    )

    customers_replying_to_brand = (
    customer_replies.groupby("brand_account")
    .agg(
        customer_replies=("tweet_id", "count"),
        unique_customer_threads=(
            "tweet_id",
            "nunique",
        ),
    )
    )

    ranking = replies_to_customers.join(
        customers_replying_to_brand,
        how="outer",
    ).fillna(0)

    # ---------------------------------------------------------
    # 4. Convert numeric columns
    # ---------------------------------------------------------

    numeric_columns = [
        "brand_replies",
        "unique_customers_replied_to",
        "customer_replies",
        "unique_customer_threads",
    ]

    for column in numeric_columns:
        ranking[column] = ranking[column].astype(int)

    # ---------------------------------------------------------
    # 5. Resolution-oriented metrics
    # ---------------------------------------------------------

    ranking["reply_coverage_ratio"] = (
        ranking["brand_replies"]
        / ranking["customer_replies"].replace(0, pd.NA)
    )

    ranking["reply_coverage_ratio"] = (
        ranking["reply_coverage_ratio"]
        .fillna(0)
        .clip(upper=1.0)
    )

    # We want brands that have:
    #
    # - substantial support volume
    # - many customer interactions
    # - many direct replies
    #
    ranking["support_volume_score"] = (
        ranking["brand_replies"] * 0.6
        + ranking["unique_customers_replied_to"] * 0.4
    )

    # ---------------------------------------------------------
    # 6. Remove tiny accounts
    # ---------------------------------------------------------

    ranking = ranking[
        ranking["brand_replies"] >= 1000
    ].copy()

    ranking = ranking.sort_values(
        "support_volume_score",
        ascending=False,
    )

    ranking["rank"] = range(1, len(ranking) + 1)

    # Put rank first
    ranking = ranking[
        [
            "rank",
            "brand_replies",
            "unique_customers_replied_to",
            "customer_replies",
            "unique_customer_threads",
            "reply_coverage_ratio",
            "support_volume_score",
        ]
    ]

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking.to_csv(OUTPUT_PATH)

    print(
        "\n========== TOP 30 BRANDS =========="
    )

    print(
        ranking.head(30).to_string()
    )

    print(
        f"\nSaved ranking to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()