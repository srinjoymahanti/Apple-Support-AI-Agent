from pathlib import Path
import pandas as pd

GOLDEN_PATH = Path("data/processed/golden_set.csv")
REVIEW_PATH = Path("data/processed/human_agreement_review.csv")
ANSWER_KEY_PATH = Path("data/processed/human_agreement_answer_key.csv")

N = 50
RANDOM_STATE = 42

def main() -> None:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_PATH}"
        )

    golden = pd.read_csv(GOLDEN_PATH)

    required = {"tweet_id", "text", "gold_intent"}
    missing = required - set(golden.columns)
    if missing:
        raise ValueError(
            f"Golden set is missing columns: {sorted(missing)}"
        )

    golden["tweet_id"] = golden["tweet_id"].astype(str).str.strip()
    golden["text"] = (
        golden["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    golden["gold_intent"] = (
        golden["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    golden = golden[
        (golden["text"] != "")
        & (golden["gold_intent"] != "")
    ].copy()

    if len(golden) < N:
        raise ValueError(
            f"Need at least {N} labeled golden examples; found {len(golden)}."
        )

    sample = golden.sample(
        n=N,
        random_state=RANDOM_STATE,
    ).copy()

    # Blind sheet: second reviewer must only see these columns.
    review = sample[
        ["tweet_id", "text"]
    ].copy()
    review["reviewer_2_intent"] = ""

    # Answer key: keep reviewer 1's gold labels separate.
    answer_key = sample[
        ["tweet_id", "text", "gold_intent"]
    ].copy()
    answer_key = answer_key.rename(
        columns={"gold_intent": "reviewer_1_intent"}
    )

    REVIEW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    answer_key.to_csv(
        ANSWER_KEY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Created blind review file: {REVIEW_PATH}")
    print(f"Created answer key: {ANSWER_KEY_PATH}")
    print(f"Examples: {len(review)}")
    print("\nSecond reviewer should use these intents:")
    for intent in sorted(golden["gold_intent"].unique()):
        print(f"- {intent}")

if __name__ == "__main__":
    main()
