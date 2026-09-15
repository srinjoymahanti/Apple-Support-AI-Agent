from pathlib import Path

import pandas as pd

from retrieve import retrieve


GOLDEN_PATH = Path(
    "data/processed/golden_set.csv"
)

OUTPUT_PATH = Path(
    "data/processed/retrieval_evaluation.csv"
)

EVAL_SIZE = 50
RANDOM_STATE = 42


def load_previous_results() -> tuple[pd.DataFrame, set[str]]:
    """
    Load previously saved retrieval judgments.

    Only rows with a valid top1_relevant value of 0 or 1
    are considered completed.
    """

    if not OUTPUT_PATH.exists():
        return pd.DataFrame(), set()

    previous = pd.read_csv(
        OUTPUT_PATH
    )

    required_columns = {
        "tweet_id",
        "top1_relevant",
    }

    missing_columns = required_columns - set(
        previous.columns
    )

    if missing_columns:
        print(
            "Existing evaluation file is missing "
            f"columns: {sorted(missing_columns)}"
        )

        return pd.DataFrame(), set()

    previous["tweet_id"] = (
        previous["tweet_id"]
        .astype(str)
        .str.strip()
    )

    previous["top1_relevant"] = pd.to_numeric(
        previous["top1_relevant"],
        errors="coerce",
    )

    completed_ids = set(
        previous.loc[
            previous["top1_relevant"].isin([0, 1]),
            "tweet_id",
        ]
    )

    return previous, completed_ids


def save_result(
    previous: pd.DataFrame,
    new_result: dict,
) -> pd.DataFrame:
    """
    Append one completed judgment and save immediately.
    """

    new_row = pd.DataFrame(
        [new_result]
    )

    if previous.empty:
        updated = new_row
    else:
        updated = pd.concat(
            [
                previous,
                new_row,
            ],
            ignore_index=True,
        )

        # Keep one row per tweet ID.
        updated["tweet_id"] = (
            updated["tweet_id"]
            .astype(str)
            .str.strip()
        )

        updated = updated.drop_duplicates(
            subset=["tweet_id"],
            keep="last",
        )

    updated.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return updated


def main() -> None:

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_PATH}"
        )

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    golden["gold_intent"] = (
        golden["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    golden["text"] = (
        golden["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    golden["tweet_id"] = (
        golden["tweet_id"]
        .astype(str)
        .str.strip()
    )

    golden = golden[
        (golden["gold_intent"] != "")
        & (golden["text"] != "")
    ].copy()

    if len(golden) < EVAL_SIZE:
        raise ValueError(
            "Not enough labeled golden examples."
        )

    # ---------------------------------------------------------
    # Fixed reproducible sample
    # ---------------------------------------------------------

    sample = golden.sample(
        n=EVAL_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    # ---------------------------------------------------------
    # Load previous progress
    # ---------------------------------------------------------

    previous, completed_ids = (
        load_previous_results()
    )

    print(
        f"Retrieval evaluation examples: "
        f"{len(sample)}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    remaining = sample[
        ~sample["tweet_id"].isin(
            completed_ids
        )
    ].copy()

    print(
        f"Remaining to evaluate: "
        f"{len(remaining)}"
    )

    if remaining.empty:
        print(
            "\nAll retrieval evaluation examples "
            "have already been completed."
        )

        # Calculate final metric from all saved results
        valid_previous = previous[
            previous["tweet_id"].isin(
                sample["tweet_id"]
            )
            & previous["top1_relevant"].isin(
                [0, 1]
            )
        ].copy()

        if not valid_previous.empty:
            top1_recall = (
                valid_previous[
                    "top1_relevant"
                ].mean()
            )

            print(
                "\n========== RETRIEVAL RESULTS =========="
            )

            print(
                f"Evaluated examples: "
                f"{len(valid_previous)}"
            )

            print(
                f"Recall@1: "
                f"{top1_recall:.4f}"
            )

            print(
                f"\nSaved evaluation to: "
                f"{OUTPUT_PATH}"
            )

        return

    # ---------------------------------------------------------
    # Evaluate unanswered examples
    # ---------------------------------------------------------

    for _, row in remaining.iterrows():

        query = row["text"]

        print(
            "\n" + "=" * 90
        )

        # Calculate the example number based on
        # the fixed sample order.
        example_number = (
            sample.index.get_loc(
                row.name
            )
            + 1
        )

        print(
            f"Example "
            f"{example_number}/{len(sample)}"
        )

        print(
            f"Tweet ID: "
            f"{row['tweet_id']}"
        )

        print(
            f"\nCustomer query:\n"
            f"{query}"
        )

        print(
            f"\nGold intent: "
            f"{row['gold_intent']}"
        )

        results = retrieve(
            query,
            top_k=5,
        )

        print(
            "\n========== TOP 5 RETRIEVED CASES =========="
        )

        if results.empty:
            print(
                "\nNo retrieval results found."
            )

            print(
                "\nRate retrieval:"
            )

            print(
                "1 = Relevant"
            )

            print(
                "0 = Not relevant"
            )

            print(
                "s = Skip"
            )

            print(
                "q = Quit"
            )

            choice = input(
                "\nNo result available. "
                "Enter [0/s/q]: "
            ).strip().lower()

            if choice == "q":
                print(
                    "\nStopping evaluation."
                )
                break

            if choice == "s":
                continue

            if choice != "0":
                print(
                    "Invalid choice. Skipping."
                )
                continue

            top_result_customer = ""
            top_result_brand = ""
            top_similarity = 0.0

        else:

            for rank, (_, result) in enumerate(
                results.iterrows(),
                start=1,
            ):

                print(
                    f"\n--- Result {rank} ---"
                )

                print(
                    f"Similarity: "
                    f"{float(result['similarity']):.4f}"
                )

                print(
                    f"Customer: "
                    f"{result['customer_text']}"
                )

                print(
                    f"AppleSupport: "
                    f"{result['brand_text']}"
                )

            print(
                "\nRate retrieval:"
            )

            print(
                "1 = Relevant"
            )

            print(
                "0 = Not relevant"
            )

            print(
                "s = Skip"
            )

            print(
                "q = Quit"
            )

            choice = input(
                "\nIs Result 1 relevant? "
                "[1/0/s/q]: "
            ).strip().lower()

            if choice == "q":
                print(
                    "\nStopping evaluation."
                )
                break

            if choice == "s":
                continue

            if choice not in {"0", "1"}:
                print(
                    "Invalid choice. Skipping."
                )
                continue

            top_result = results.iloc[0]

            top_result_customer = (
                top_result["customer_text"]
            )

            top_result_brand = (
                top_result["brand_text"]
            )

            top_similarity = float(
                top_result["similarity"]
            )

        # -----------------------------------------------------
        # Save immediately after every valid judgment
        # -----------------------------------------------------

        new_result = {
            "tweet_id": row["tweet_id"],
            "query": query,
            "gold_intent": row[
                "gold_intent"
            ],
            "top_result_customer": (
                top_result_customer
            ),
            "top_result_brand": (
                top_result_brand
            ),
            "top_similarity": (
                top_similarity
            ),
            "top1_relevant": int(
                choice
            ),
        }

        previous = save_result(
            previous,
            new_result,
        )

        completed_ids.add(
            str(row["tweet_id"])
        )

        print(
            "\nSaved progress."
        )

        print(
            f"Completed: "
            f"{len(completed_ids)}/"
            f"{len(sample)}"
        )

    # ---------------------------------------------------------
    # Final / interim results
    # ---------------------------------------------------------

    if not OUTPUT_PATH.exists():
        print(
            "\nNo evaluation results recorded."
        )
        return

    all_results = pd.read_csv(
        OUTPUT_PATH
    )

    all_results["tweet_id"] = (
        all_results["tweet_id"]
        .astype(str)
        .str.strip()
    )

    all_results["top1_relevant"] = pd.to_numeric(
        all_results["top1_relevant"],
        errors="coerce",
    )

    evaluated = all_results[
        all_results["tweet_id"].isin(
            sample["tweet_id"]
        )
        & all_results["top1_relevant"].isin(
            [0, 1]
        )
    ].copy()

    if evaluated.empty:
        print(
            "\nNo evaluation results recorded."
        )
        return

    top1_recall = (
        evaluated[
            "top1_relevant"
        ].mean()
    )

    print(
        "\n========== RETRIEVAL RESULTS =========="
    )

    print(
        f"Evaluated examples: "
        f"{len(evaluated)}/{len(sample)}"
    )

    print(
        f"Recall@1: "
        f"{top1_recall:.4f}"
    )

    print(
        f"\nSaved evaluation to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()