from pathlib import Path
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from classify_intent import classify_intent
from retrieve import retrieve
from generate_reply import (
    create_client,
    generate_reply,
    load_prompt,
)


GOLDEN_PATH = Path(
    "data/processed/golden_set.csv"
)

OUTPUT_PATH = Path(
    "data/processed/reply_evaluation.csv"
)

PROMPT_PATH = Path(
    "configs/reply_generation_prompt.txt"
)

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

EVAL_SIZE = 30
RANDOM_STATE = 42

MIN_RETRIEVAL_SIMILARITY = 0.20

MAX_RETRIES = 5


def create_classifier_client() -> OpenAI:
    """
    Create a Groq-compatible OpenAI client.
    """

    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


def call_classifier(
    client: OpenAI,
    customer_message: str,
) -> dict:
    """
    Call the existing intent classifier.

    Important:
    classify_intent() expects the parameter
    name `message`, not `customer_message`.
    """

    return classify_intent(
        client=client,
        message=customer_message,
    )


def get_retrieved_cases(
    customer_message: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve historical AppleSupport cases and
    keep only cases above the similarity threshold.
    """

    results = retrieve(
        customer_message,
        top_k=top_k,
    )

    if results.empty:
        return []

    retrieved_cases = []

    for _, row in results.iterrows():

        similarity = float(
            row["similarity"]
        )

        if similarity < MIN_RETRIEVAL_SIMILARITY:
            continue

        retrieved_cases.append(
            {
                "customer_text": str(
                    row["customer_text"]
                ),
                "brand_text": str(
                    row["brand_text"]
                ),
                "similarity": similarity,
            }
        )

    return retrieved_cases


def call_reply_generator(
    client: OpenAI,
    prompt_template: str,
    customer_message: str,
    intent: str,
    retrieved_cases: list[dict],
) -> dict:
    """
    Call the existing grounded reply generator.
    """

    return generate_reply(
        client=client,
        prompt_template=prompt_template,
        customer_message=customer_message,
        intent=intent,
        retrieved_cases=retrieved_cases,
    )


def load_previous_results() -> tuple[
    pd.DataFrame,
    set[str],
]:
    """
    Load previously saved evaluation results.

    A tweet is considered completed when a generated
    reply has been saved for it.
    """

    if not OUTPUT_PATH.exists():
        return (
            pd.DataFrame(),
            set(),
        )

    previous = pd.read_csv(
        OUTPUT_PATH
    )

    if "tweet_id" not in previous.columns:
        return (
            pd.DataFrame(),
            set(),
        )

    previous["tweet_id"] = (
        previous["tweet_id"]
        .astype(str)
        .str.strip()
    )

    if "generated_reply" not in previous.columns:
        return (
            previous,
            set(),
        )

    generated = (
        previous["generated_reply"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    completed_ids = set(
        previous.loc[
            generated != "",
            "tweet_id",
        ]
    )

    return (
        previous,
        completed_ids,
    )


def save_result(
    previous: pd.DataFrame,
    result: dict,
) -> pd.DataFrame:
    """
    Append one result and save immediately.
    """

    new_row = pd.DataFrame(
        [result]
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


def run_with_retry(
    function,
    *args,
    **kwargs,
):
    """
    Retry API calls with exponential backoff.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:
            return function(
                *args,
                **kwargs,
            )

        except Exception as exc:

            last_error = exc

            print(
                f"\nAPI error "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:

                wait_seconds = min(
                    2 ** attempt,
                    60,
                )

                print(
                    f"Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        f"API call failed after "
        f"{MAX_RETRIES} attempts."
    ) from last_error


def print_retrieved_cases(
    retrieved_cases: list[dict],
) -> None:
    """
    Print historical evidence.
    """

    print(
        "\n========== HISTORICAL EVIDENCE =========="
    )

    if not retrieved_cases:

        print(
            "\nNo sufficiently similar "
            "historical cases found."
        )

        return

    for i, case in enumerate(
        retrieved_cases,
        start=1,
    ):

        print(
            f"\n--- Case {i} ---"
        )

        print(
            f"Similarity: "
            f"{case['similarity']:.4f}"
        )

        print(
            f"Customer: "
            f"{case['customer_text']}"
        )

        print(
            f"AppleSupport: "
            f"{case['brand_text']}"
        )


def main() -> None:

    # ---------------------------------------------------------
    # Validate input files
    # ---------------------------------------------------------

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: "
            f"{GOLDEN_PATH}"
        )

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Reply prompt not found: "
            f"{PROMPT_PATH}"
        )

    # ---------------------------------------------------------
    # Load golden set
    # ---------------------------------------------------------

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    golden["tweet_id"] = (
        golden["tweet_id"]
        .astype(str)
        .str.strip()
    )

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
    # Load checkpoint
    # ---------------------------------------------------------

    previous, completed_ids = (
        load_previous_results()
    )

    print(
        "\n========== REPLY EVALUATION =========="
    )

    print(
        f"Evaluation examples: "
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
        f"Remaining: "
        f"{len(remaining)}"
    )

    if remaining.empty:

        print(
            "\nAll reply evaluations "
            "are already complete."
        )

        return

    # ---------------------------------------------------------
    # API clients
    # ---------------------------------------------------------

    classifier_client = (
        create_classifier_client()
    )

    reply_client = create_client()

    reply_prompt = load_prompt()

    # ---------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------

    for _, row in remaining.iterrows():

        query = row["text"]

        tweet_id = str(
            row["tweet_id"]
        )

        gold_intent = row[
            "gold_intent"
        ]

        example_number = (
            sample.index.get_loc(
                row.name
            )
            + 1
        )

        print(
            "\n" + "=" * 90
        )

        print(
            f"Example "
            f"{example_number}/{len(sample)}"
        )

        print(
            f"Tweet ID: "
            f"{tweet_id}"
        )

        print(
            f"\nCustomer query:\n"
            f"{query}"
        )

        print(
            f"\nGold intent: "
            f"{gold_intent}"
        )

        # -----------------------------------------------------
        # 1. Intent classification
        # -----------------------------------------------------

        print(
            "\n========== CLASSIFICATION =========="
        )

        try:

            classification = (
                run_with_retry(
                    call_classifier,
                    classifier_client,
                    query,
                )
            )

        except Exception as exc:

            print(
                f"\nClassification failed: "
                f"{exc}"
            )

            continue

        predicted_intent = str(
            classification.get(
                "intent",
                "",
            )
        ).strip()

        intent_confidence = (
            classification.get(
                "confidence",
                None,
            )
        )

        print(
            f"\nPredicted intent: "
            f"{predicted_intent}"
        )

        print(
            f"Intent confidence: "
            f"{intent_confidence}"
        )

        # -----------------------------------------------------
        # 2. Retrieval
        # -----------------------------------------------------

        print(
            "\n========== RETRIEVAL =========="
        )

        try:

            retrieved_cases = (
                get_retrieved_cases(
                    query,
                    top_k=5,
                )
            )

        except Exception as exc:

            print(
                f"\nRetrieval failed: "
                f"{exc}"
            )

            continue

        print_retrieved_cases(
            retrieved_cases
        )

        # -----------------------------------------------------
        # 3. Reply generation
        # -----------------------------------------------------

        print(
            "\n========== GENERATED REPLY =========="
        )

        try:

            reply_result = (
                run_with_retry(
                    call_reply_generator,
                    reply_client,
                    reply_prompt,
                    query,
                    predicted_intent,
                    retrieved_cases,
                )
            )

        except Exception as exc:

            print(
                f"\nReply generation failed: "
                f"{exc}"
            )

            continue

        generated_reply = str(
            reply_result.get(
                "draft_reply",
                "",
            )
        ).strip()

        evidence_sufficient = (
            reply_result.get(
                "evidence_sufficient",
                None,
            )
        )

        reply_confidence = (
            reply_result.get(
                "confidence",
                None,
            )
        )

        generator_reason = str(
            reply_result.get(
                "reason",
                "",
            )
        ).strip()

        print(
            f"\n{generated_reply}"
        )

        print(
            f"\nEvidence sufficient: "
            f"{evidence_sufficient}"
        )

        print(
            f"Reply confidence: "
            f"{reply_confidence}"
        )

        print(
            f"Reason: "
            f"{generator_reason}"
        )

        # -----------------------------------------------------
        # 4. Human evaluation
        # -----------------------------------------------------

        print(
            "\n========== HUMAN EVALUATION =========="
        )

        print(
            "\nGroundedness:"
        )

        print(
            "1 = Fully supported by historical evidence"
        )

        print(
            "0 = Contains unsupported information"
        )

        groundedness = input(
            "\nIs the reply fully grounded? "
            "[1/0/s/q]: "
        ).strip().lower()

        if groundedness == "q":

            print(
                "\nStopping evaluation."
            )

            break

        if groundedness == "s":
            continue

        if groundedness not in {
            "0",
            "1",
        }:

            print(
                "Invalid choice. Skipping."
            )

            continue

        print(
            "\nHelpfulness:"
        )

        print(
            "1 = Helpful to the customer"
        )

        print(
            "0 = Not sufficiently helpful"
        )

        helpfulness = input(
            "\nIs the reply helpful? "
            "[1/0]: "
        ).strip().lower()

        if helpfulness not in {
            "0",
            "1",
        }:

            print(
                "Invalid choice. Skipping."
            )

            continue

        print(
            "\nProblem addressed:"
        )

        print(
            "1 = Directly addresses the customer's problem"
        )

        print(
            "0 = Does not address the customer's problem"
        )

        addresses_problem = input(
            "\nDoes it address the problem? "
            "[1/0]: "
        ).strip().lower()

        if addresses_problem not in {
            "0",
            "1",
        }:

            print(
                "Invalid choice. Skipping."
            )

            continue

        print(
            "\nUnsupported claim:"
        )

        print(
            "1 = Contains unsupported claim"
        )

        print(
            "0 = No unsupported claim"
        )

        unsupported_claim = input(
            "\nDoes it contain an unsupported claim? "
            "[1/0]: "
        ).strip().lower()

        if unsupported_claim not in {
            "0",
            "1",
        }:

            print(
                "Invalid choice. Skipping."
            )

            continue

        # -----------------------------------------------------
        # 5. Save result immediately
        # -----------------------------------------------------

        if retrieved_cases:

            top_similarity = max(
                case["similarity"]
                for case in retrieved_cases
            )

        else:

            top_similarity = 0.0

        result = {
            "tweet_id": tweet_id,
            "query": query,
            "gold_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": int(
                predicted_intent
                == gold_intent
            ),
            "intent_confidence": (
                intent_confidence
            ),
            "num_retrieved_cases": len(
                retrieved_cases
            ),
            "top_similarity": (
                top_similarity
            ),
            "generated_reply": (
                generated_reply
            ),
            "evidence_sufficient": (
                evidence_sufficient
            ),
            "reply_confidence": (
                reply_confidence
            ),
            "generator_reason": (
                generator_reason
            ),
            "grounded": int(
                groundedness
            ),
            "helpful": int(
                helpfulness
            ),
            "addresses_problem": int(
                addresses_problem
            ),
            "unsupported_claim": int(
                unsupported_claim
            ),
        }

        previous = save_result(
            previous,
            result,
        )

        completed_ids.add(
            tweet_id
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
    # 6. Final metrics
    # ---------------------------------------------------------

    if not OUTPUT_PATH.exists():

        print(
            "\nNo evaluation results recorded."
        )

        return

    evaluation = pd.read_csv(
        OUTPUT_PATH
    )

    evaluation["tweet_id"] = (
        evaluation["tweet_id"]
        .astype(str)
        .str.strip()
    )

    evaluation = evaluation[
        evaluation["tweet_id"].isin(
            sample["tweet_id"]
        )
    ].copy()

    if evaluation.empty:

        print(
            "\nNo completed evaluations."
        )

        return

    print(
        "\n========== REPLY EVALUATION RESULTS =========="
    )

    print(
        f"Evaluated examples: "
        f"{len(evaluation)}/{len(sample)}"
    )

    print(
        f"Intent accuracy: "
        f"{evaluation['intent_correct'].mean():.4f}"
    )

    print(
        f"Groundedness: "
        f"{evaluation['grounded'].mean():.4f}"
    )

    print(
        f"Helpfulness: "
        f"{evaluation['helpful'].mean():.4f}"
    )

    print(
        f"Addresses problem: "
        f"{evaluation['addresses_problem'].mean():.4f}"
    )

    print(
        f"Unsupported claim rate: "
        f"{evaluation['unsupported_claim'].mean():.4f}"
    )

    print(
        f"\nSaved evaluation to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()