from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/raw/customer_support_twitter.csv"
)

MESSAGES_OUTPUT = Path(
    "data/processed/apple_support_conversation_messages.csv"
)

THREADS_OUTPUT = Path(
    "data/processed/apple_support_conversations.csv"
)

PAIRS_OUTPUT = Path(
    "data/processed/apple_support_response_pairs.csv"
)


# ============================================================
# Configuration
# ============================================================

BRAND = "AppleSupport"


# ============================================================
# Helpers
# ============================================================

def normalize_id(value) -> Optional[str]:
    """
    Normalize tweet IDs so numeric values such as
    1234567.0 become '1234567'.
    """
    if pd.isna(value):
        return None

    # Handle numeric IDs read as floats.
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

    value = str(value).strip()

    if not value or value.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    # Handle strings such as "1234567.0".
    if value.endswith(".0"):
        try:
            return str(int(float(value)))
        except ValueError:
            pass

    return value


def find_root(
    tweet_id: str,
    parent_of: Dict[str, Optional[str]],
    cache: Dict[str, str],
) -> str:
    """
    Find the root tweet of a conversation tree.

    Uses iterative path compression to avoid recursion depth issues.
    """

    if tweet_id in cache:
        return cache[tweet_id]

    path = []
    current = tweet_id
    visited = set()

    while True:

        if current in cache:
            root = cache[current]
            break

        if current in visited:
            # Defensive handling for malformed/cyclic data.
            root = current
            break

        visited.add(current)
        path.append(current)

        parent = parent_of.get(current)

        if not parent:
            root = current
            break

        # Parent is missing from the dataset.
        if parent not in parent_of:
            root = parent
            break

        current = parent

    for node in path:
        cache[node] = root

    return root


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    print("Reading dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id",
        ],
        low_memory=False,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    df["tweet_id"] = (
        df["tweet_id"]
        .map(normalize_id)
    )

    df["in_response_to_tweet_id"] = (
        df["in_response_to_tweet_id"]
        .map(normalize_id)
    )

    df["text"] = (
        df["text"]
        .map(clean_text)
    )

    df["author_id"] = (
        df["author_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    # Remove rows without tweet ID.
    df = df[
        df["tweet_id"].notna()
    ].copy()

    # Remove duplicate tweet IDs defensively.
    df = df.drop_duplicates(
        subset=["tweet_id"],
        keep="first",
    )

    print(
        f"Unique tweets: {len(df):,}"
    )

    # --------------------------------------------------------
    # Build lookup dictionaries
    # --------------------------------------------------------

    print(
        "\nBuilding reply graph..."
    )

    parent_of = dict(
        zip(
            df["tweet_id"],
            df["in_response_to_tweet_id"],
        )
    )

    author_of = dict(
        zip(
            df["tweet_id"],
            df["author_id"],
        )
    )

    inbound_of = dict(
        zip(
            df["tweet_id"],
            df["inbound"],
        )
    )

    created_of = dict(
        zip(
            df["tweet_id"],
            df["created_at"],
        )
    )

    text_of = dict(
        zip(
            df["tweet_id"],
            df["text"],
        )
    )

    # --------------------------------------------------------
    # Find AppleSupport tweets
    # --------------------------------------------------------

    brand_mask = (
        df["author_id"] == BRAND
    )

    brand_df = df[
        brand_mask
    ].copy()

    print(
        f"{BRAND} tweets: "
        f"{len(brand_df):,}"
    )

    # --------------------------------------------------------
    # Find AppleSupport replies to customer messages
    # --------------------------------------------------------

    direct_pairs = []

    for _, row in brand_df.iterrows():

        brand_id = row["tweet_id"]
        parent_id = row[
            "in_response_to_tweet_id"
        ]

        if not parent_id:
            continue

        # Parent must exist.
        if parent_id not in text_of:
            continue

        # Parent must not be an AppleSupport tweet.
        parent_author = author_of.get(
            parent_id
        )

        if parent_author == BRAND:
            continue

        direct_pairs.append(
            {
                "customer_tweet_id": parent_id,
                "brand_tweet_id": brand_id,
                "customer_text": text_of.get(
                    parent_id,
                    "",
                ),
                "brand_text": text_of.get(
                    brand_id,
                    "",
                ),
                "customer_created_at": created_of.get(
                    parent_id
                ),
                "brand_created_at": created_of.get(
                    brand_id
                ),
            }
        )

    pairs_df = pd.DataFrame(
        direct_pairs
    )

    print(
        f"Direct customer → AppleSupport pairs: "
        f"{len(pairs_df):,}"
    )

    # --------------------------------------------------------
    # Compute conversation roots
    # --------------------------------------------------------

    print(
        "\nComputing conversation roots..."
    )

    root_cache: Dict[str, str] = {}

    relevant_ids = set()

    # Every tweet in a direct customer → brand pair
    # belongs to a relevant conversation.
    if not pairs_df.empty:

        relevant_ids.update(
            pairs_df["customer_tweet_id"]
            .tolist()
        )

        relevant_ids.update(
            pairs_df["brand_tweet_id"]
            .tolist()
        )

    roots = {}

    for tweet_id in relevant_ids:

        roots[tweet_id] = find_root(
            tweet_id,
            parent_of,
            root_cache,
        )

    relevant_roots = set(
        roots.values()
    )

    print(
        f"Relevant conversation roots: "
        f"{len(relevant_roots):,}"
    )

    # --------------------------------------------------------
    # Assign roots to all tweets
    # --------------------------------------------------------
    #
    # We only need tweets that belong to relevant
    # conversation trees.
    # --------------------------------------------------------

    print(
        "\nExtracting conversation messages..."
    )

    # To identify all members of each root tree, compute root
    # for every tweet. This is the main processing step.
    conversation_rows: List[dict] = []

    for tweet_id in df["tweet_id"]:

        root = find_root(
            tweet_id,
            parent_of,
            root_cache,
        )

        if root not in relevant_roots:
            continue

        inbound = inbound_of.get(
            tweet_id
        )

        author = str(
            author_of.get(
                tweet_id,
                ""
            )
        ).strip()

        if inbound is True:
            speaker = "customer"

        elif author == BRAND:
            speaker = "brand"

        else:
            speaker = "other"

        conversation_rows.append(
            {
                "conversation_id": root,
                "tweet_id": tweet_id,
                "author_id": author,
                "speaker": speaker,
                "inbound": inbound,
                "created_at": created_of.get(
                    tweet_id
                ),
                "in_response_to_tweet_id":
                    parent_of.get(tweet_id),
                "text": text_of.get(
                    tweet_id,
                    "",
                ),
            }
        )

    messages_df = pd.DataFrame(
        conversation_rows
    )

    if messages_df.empty:
        raise RuntimeError(
            "No AppleSupport conversations were reconstructed."
        )

    messages_df = (
        messages_df
        .sort_values(
            [
                "conversation_id",
                "created_at",
                "tweet_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add turn number
    # --------------------------------------------------------

    messages_df[
        "turn_number"
    ] = (
        messages_df
        .groupby(
            "conversation_id"
        )
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Resolution-signal heuristic
    # --------------------------------------------------------
    #
    # This is NOT ground truth.
    # It is only a weak signal for later analysis.
    # --------------------------------------------------------

    resolution_pattern = (
        r"\b("
        r"thanks|thank you|thx|"
        r"worked|works|fixed|"
        r"solved|problem solved|"
        r"all good|got it|"
        r"that worked|"
        r"it's working|"
        r"it is working"
        r")\b"
    )

    messages_df[
        "weak_resolution_signal"
    ] = (
        messages_df["text"]
        .str.lower()
        .str.contains(
            resolution_pattern,
            regex=True,
            na=False,
        )
    )

    # --------------------------------------------------------
    # Save messages
    # --------------------------------------------------------

    MESSAGES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    messages_df.to_csv(
        MESSAGES_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Conversation-level summaries
    # --------------------------------------------------------

    thread_rows = []

    for conversation_id, group in (
        messages_df
        .groupby("conversation_id")
    ):

        group = group.sort_values(
            [
                "created_at",
                "tweet_id",
            ]
        )

        customer_messages = group[
            group["speaker"]
            == "customer"
        ]

        brand_messages = group[
            group["speaker"]
            == "brand"
        ]

        if customer_messages.empty:
            continue

        first_customer = (
            customer_messages
            .iloc[0]
        )

        last_customer = (
            customer_messages
            .iloc[-1]
        )

        last_brand = (
            brand_messages
            .iloc[-1]
            if not brand_messages.empty
            else None
        )

        thread_rows.append(
            {
                "conversation_id":
                    conversation_id,

                "message_count":
                    len(group),

                "customer_message_count":
                    len(customer_messages),

                "brand_message_count":
                    len(brand_messages),

                "start_time":
                    group["created_at"].min(),

                "end_time":
                    group["created_at"].max(),

                "first_customer_tweet_id":
                    first_customer["tweet_id"],

                "first_customer_text":
                    first_customer["text"],

                "last_customer_tweet_id":
                    last_customer["tweet_id"],

                "last_customer_text":
                    last_customer["text"],

                "last_brand_tweet_id":
                    (
                        last_brand["tweet_id"]
                        if last_brand is not None
                        else None
                    ),

                "last_brand_text":
                    (
                        last_brand["text"]
                        if last_brand is not None
                        else None
                    ),

                "has_weak_resolution_signal":
                    bool(
                        group[
                            "weak_resolution_signal"
                        ].any()
                    ),
            }
        )

    threads_df = pd.DataFrame(
        thread_rows
    )

    threads_df = threads_df.sort_values(
        "start_time"
    )

    threads_df.to_csv(
        THREADS_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Enhance direct response pairs
    # --------------------------------------------------------

    if not pairs_df.empty:

        pairs_df[
            "conversation_id"
        ] = pairs_df[
            "customer_tweet_id"
        ].map(
            root_cache
        )

        pairs_df[
            "customer_resolution_signal"
        ] = (
            pairs_df["customer_text"]
            .str.lower()
            .str.contains(
                resolution_pattern,
                regex=True,
                na=False,
            )
        )

        pairs_df.to_csv(
            PAIRS_OUTPUT,
            index=False,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n========== CONVERSATION DATASET =========="
    )

    print(
        f"Conversations: "
        f"{threads_df['conversation_id'].nunique():,}"
    )

    print(
        f"Messages:      "
        f"{len(messages_df):,}"
    )

    print(
        f"Customer msgs: "
        f"{(messages_df['speaker'] == 'customer').sum():,}"
    )

    print(
        f"Brand msgs:    "
        f"{(messages_df['speaker'] == 'brand').sum():,}"
    )

    print(
        f"Other msgs:    "
        f"{(messages_df['speaker'] == 'other').sum():,}"
    )

    print(
        "\nConversation length distribution:"
    )

    print(
        threads_df[
            "message_count"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nWeak resolution signal:"
    )

    print(
        threads_df[
            "has_weak_resolution_signal"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSaved files:"
    )

    print(
        f"  Messages: {MESSAGES_OUTPUT}"
    )

    print(
        f"  Threads:  {THREADS_OUTPUT}"
    )

    print(
        f"  Pairs:    {PAIRS_OUTPUT}"
    )


if __name__ == "__main__":
    main()