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
)


# ============================================================
# Configuration
# ============================================================

GOLDEN_PATH = Path(
    "data/processed/golden_set.csv"
)

OUTPUT_PATH = Path(
    "data/processed/escalation_evaluation.csv"
)

PROMPT_PATH = Path(
    "configs/reply_generation_prompt.txt"
)

TOP_K = 5

MIN_RETRIEVAL_SIMILARITY = 0.20

INTENT_CONFIDENCE_THRESHOLD = 0.80

TOP_SIMILARITY_THRESHOLD = 0.30

EVAL_SIZE = 30

RANDOM_STATE = 42

MAX_RETRIES = 5

BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# API client
# ============================================================

def create_client_from_env() -> OpenAI:
    """
    Create an OpenAI-compatible Groq client.
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


# ============================================================
# Retry helper
# ============================================================

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


# ============================================================
# Retrieval helpers
# ============================================================

def build_retrieved_cases(
    results: pd.DataFrame,
) -> list[dict]:
    """
    Convert retriever output into the structure expected
    by generate_reply().
    """

    cases = []

    if results.empty:
        return cases

    for _, row in results.iterrows():

        similarity = float(
            row["similarity"]
        )

        if similarity < MIN_RETRIEVAL_SIMILARITY:
            continue

        cases.append(
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

    return cases


# ============================================================
# Agent decision logic
# ============================================================

def make_decision(
    intent_confidence,
    top_similarity: float,
    evidence_sufficient: bool,
) -> tuple[str, str]:
    """
    Reproduce the exact AUTO_HANDLE / ESCALATE logic
    currently implemented in agent.py.
    """

    if (
        intent_confidence is not None
        and intent_confidence
        >= INTENT_CONFIDENCE_THRESHOLD
        and top_similarity
        >= TOP_SIMILARITY_THRESHOLD
        and evidence_sufficient is True
    ):

        decision = "AUTO_HANDLE"

        reason = (
            "High intent confidence and sufficiently "
            "similar historical evidence were found."
        )

        return decision, reason

    decision = "ESCALATE"

    reasons = []

    if (
        intent_confidence is None
        or intent_confidence
        < INTENT_CONFIDENCE_THRESHOLD
    ):
        reasons.append(
            "low intent confidence"
        )

    if (
        top_similarity
        < TOP_SIMILARITY_THRESHOLD
    ):
        reasons.append(
            "weak historical similarity"
        )

    if not evidence_sufficient:
        reasons.append(
            "insufficient evidence"
        )

    reason = (
        ", ".join(reasons)
        if reasons
        else "Safety threshold not satisfied."
    )

    return decision, reason


# ============================================================
# Checkpoint loading
# ============================================================

def load_previous_results() -> tuple[
    pd.DataFrame,
    set[str],
]:
    """
    Load previously saved escalation evaluations.

    A row is considered completed when human_decision
    contains AUTO_HANDLE or ESCALATE.
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
            previous,
            set(),
        )

    previous["tweet_id"] = (
        previous["tweet_id"]
        .astype(str)
        .str.strip()
    )

    if "human_decision" not in previous.columns:
        return (
            previous,
            set(),
        )

    completed_ids = set(
        previous.loc[
            previous["human_decision"].isin(
                [
                    "AUTO_HANDLE",
                    "ESCALATE",
                ]
            ),
            "tweet_id",
        ]
    )

    return (
        previous,
        completed_ids,
    )


# ============================================================
# Save checkpoint
# ============================================================

def save_result(
    previous: pd.DataFrame,
    result: dict,
) -> pd.DataFrame:
    """
    Save one completed evaluation immediately.
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


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fixed evaluation sample
    # --------------------------------------------------------

    sample = golden.sample(
        n=EVAL_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    # --------------------------------------------------------
    # Load previous progress
    # --------------------------------------------------------

    previous, completed_ids = (
        load_previous_results()
    )

    print(
        "\n========== ESCALATION EVALUATION =========="
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
            "\nAll escalation evaluations "
            "are already complete."
        )

        return

    # --------------------------------------------------------
    # API clients
    # --------------------------------------------------------

    client = create_client_from_env()

    reply_prompt = PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    for _, row in remaining.iterrows():

        tweet_id = str(
            row["tweet_id"]
        )

        query = row["text"]

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

        # ----------------------------------------------------
        # 1. Intent classification
        # ----------------------------------------------------

        print(
            "\n========== INTENT CLASSIFICATION =========="
        )

        try:

            classification = (
                run_with_retry(
                    classify_intent,
                    client=client,
                    message=query,
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
            f"Predicted intent: "
            f"{predicted_intent}"
        )

        print(
            f"Intent confidence: "
            f"{intent_confidence}"
        )

        # ----------------------------------------------------
        # 2. Historical retrieval
        # ----------------------------------------------------

        print(
            "\n========== HISTORICAL RETRIEVAL =========="
        )

        try:

            retrieval_results = retrieve(
                query,
                top_k=TOP_K,
            )

        except Exception as exc:

            print(
                f"\nRetrieval failed: "
                f"{exc}"
            )

            continue

        retrieved_cases = (
            build_retrieved_cases(
                retrieval_results
            )
        )

        print(
            f"Usable retrieved cases: "
            f"{len(retrieved_cases)}"
        )

        top_similarity = 0.0

        if retrieved_cases:

            top_similarity = max(
                case["similarity"]
                for case in retrieved_cases
            )

        print(
            f"Top similarity: "
            f"{top_similarity:.4f}"
        )

        # ----------------------------------------------------
        # 3. Reply generation
        # ----------------------------------------------------

        print(
            "\n========== REPLY GENERATION =========="
        )

        try:

            reply_result = (
                run_with_retry(
                    generate_reply,
                    client=client,
                    prompt_template=reply_prompt,
                    customer_message=query,
                    intent=predicted_intent,
                    retrieved_cases=retrieved_cases,
                )
            )

        except Exception as exc:

            print(
                f"\nReply generation failed: "
                f"{exc}"
            )

            continue

        draft_reply = str(
            reply_result.get(
                "draft_reply",
                "",
            )
        ).strip()

        evidence_sufficient = (
            reply_result.get(
                "evidence_sufficient",
                False,
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
            "\nDraft reply:"
        )

        print(
            draft_reply
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
            f"Generator reason: "
            f"{generator_reason}"
        )

        # ----------------------------------------------------
        # 4. Agent routing decision
        # ----------------------------------------------------

        agent_decision, agent_reason = (
            make_decision(
                intent_confidence=(
                    intent_confidence
                ),
                top_similarity=(
                    top_similarity
                ),
                evidence_sufficient=(
                    evidence_sufficient
                ),
            )
        )

        print(
            "\n========== AGENT ROUTING =========="
        )

        print(
            f"Agent decision: "
            f"{agent_decision}"
        )

        print(
            f"Agent reason: "
            f"{agent_reason}"
        )

        # ----------------------------------------------------
        # 5. Human routing judgment
        # ----------------------------------------------------

        print(
            "\n========== HUMAN ROUTING JUDGMENT =========="
        )

        print(
            "\nChoose the safest appropriate action "
            "for this customer."
        )

        print(
            "1 = AUTO_HANDLE"
        )

        print(
            "0 = ESCALATE"
        )

        human_decision = input(
            "\nShould this be auto-handled? "
            "[1/0/s/q]: "
        ).strip().lower()

        if human_decision == "q":

            print(
                "\nStopping evaluation."
            )

            break

        if human_decision == "s":

            continue

        if human_decision not in {
            "0",
            "1",
        }:

            print(
                "Invalid choice. Skipping."
            )

            continue

        if human_decision == "1":

            human_decision_label = (
                "AUTO_HANDLE"
            )

        else:

            human_decision_label = (
                "ESCALATE"
            )

        print(
            f"\nHuman decision: "
            f"{human_decision_label}"
        )

        # ----------------------------------------------------
        # Human escalation reason
        # ----------------------------------------------------

        if human_decision_label == "ESCALATE":

            print(
                "\nWhy should this be escalated?"
            )

            print(
                "1 = Insufficient evidence"
            )

            print(
                "2 = Unclear customer problem"
            )

            print(
                "3 = Low confidence"
            )

            print(
                "4 = Retrieved evidence is not relevant"
            )

            print(
                "5 = Other / safety concern"
            )

            reason_code = input(
                "\nEscalation reason [1-5]: "
            ).strip()

            valid_reason_codes = {
                "1",
                "2",
                "3",
                "4",
                "5",
            }

            if reason_code not in valid_reason_codes:

                print(
                    "Invalid reason. "
                    "Using Other / safety concern."
                )

                reason_code = "5"

        else:

            reason_code = ""

        # ----------------------------------------------------
        # Agreement
        # ----------------------------------------------------

        routing_agreement = int(
            agent_decision
            == human_decision_label
        )

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        evaluation_result = {
            "tweet_id": tweet_id,
            "query": query,
            "gold_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "intent_confidence": (
                intent_confidence
            ),
            "top_similarity": (
                top_similarity
            ),
            "num_retrieved_cases": len(
                retrieved_cases
            ),
            "evidence_sufficient": (
                evidence_sufficient
            ),
            "reply_confidence": (
                reply_confidence
            ),
            "generated_reply": (
                draft_reply
            ),
            "generator_reason": (
                generator_reason
            ),
            "agent_decision": (
                agent_decision
            ),
            "agent_reason": (
                agent_reason
            ),
            "human_decision": (
                human_decision_label
            ),
            "human_escalation_reason": (
                reason_code
            ),
            "routing_agreement": (
                routing_agreement
            ),
        }

        previous = save_result(
            previous,
            evaluation_result,
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

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

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

    routing_agreement = (
        evaluation[
            "routing_agreement"
        ].mean()
    )

    total_auto = (
        evaluation[
            "human_decision"
        ]
        == "AUTO_HANDLE"
    ).sum()

    total_escalate = (
        evaluation[
            "human_decision"
        ]
        == "ESCALATE"
    ).sum()

    agent_auto = (
        evaluation[
            "agent_decision"
        ]
        == "AUTO_HANDLE"
    ).sum()

    agent_escalate = (
        evaluation[
            "agent_decision"
        ]
        == "ESCALATE"
    ).sum()

    print(
        "\n========== ESCALATION RESULTS =========="
    )

    print(
        f"Evaluated examples: "
        f"{len(evaluation)}/{len(sample)}"
    )

    print(
        f"Routing agreement: "
        f"{routing_agreement:.4f}"
    )

    print(
        f"Human AUTO_HANDLE: "
        f"{total_auto}"
    )

    print(
        f"Human ESCALATE: "
        f"{total_escalate}"
    )

    print(
        f"Agent AUTO_HANDLE: "
        f"{agent_auto}"
    )

    print(
        f"Agent ESCALATE: "
        f"{agent_escalate}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = pd.crosstab(
        evaluation["human_decision"],
        evaluation["agent_decision"],
        rownames=["Human"],
        colnames=["Agent"],
        dropna=False,
    )

    print(
        "\nRouting confusion matrix:"
    )

    print(
        confusion
    )

    # --------------------------------------------------------
    # Escalation precision / recall
    # --------------------------------------------------------

    human_escalated = (
        evaluation[
            "human_decision"
        ]
        == "ESCALATE"
    )

    agent_escalated = (
        evaluation[
            "agent_decision"
        ]
        == "ESCALATE"
    )

    true_positive = (
        human_escalated
        & agent_escalated
    ).sum()

    false_positive = (
        ~human_escalated
        & agent_escalated
    ).sum()

    false_negative = (
        human_escalated
        & ~agent_escalated
    ).sum()

    if (
        true_positive
        + false_positive
    ) > 0:

        escalation_precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
        )

    else:

        escalation_precision = 0.0

    if (
        true_positive
        + false_negative
    ) > 0:

        escalation_recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
        )

    else:

        escalation_recall = 0.0

    print(
        f"\nEscalation precision: "
        f"{escalation_precision:.4f}"
    )

    print(
        f"Escalation recall: "
        f"{escalation_recall:.4f}"
    )

    print(
        f"\nSaved evaluation to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()