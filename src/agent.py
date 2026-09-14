from pathlib import Path
import sys

from retrieve import retrieve
from generate_reply import (
    create_client,
    generate_reply,
)
from classify_intent import classify_intent


# ============================================================
# Configuration
# ============================================================

TOP_K = 5

MIN_RETRIEVAL_SIMILARITY = 0.20


# ============================================================
# Helper functions
# ============================================================

def build_retrieved_cases(results):
    """
    Convert retriever output into the format expected
    by generate_reply().
    """

    cases = []

    for _, row in results.iterrows():

        similarity = float(
            row["similarity"]
        )

        # Ignore very weak matches.
        if similarity < MIN_RETRIEVAL_SIMILARITY:
            continue

        cases.append(
            {
                "customer_text": row[
                    "customer_text"
                ],
                "brand_text": row[
                    "brand_text"
                ],
                "similarity": similarity,
            }
        )

    return cases


def load_reply_prompt() -> str:
    """
    Load the grounded reply-generation prompt.
    """

    prompt_path = Path(
        "configs/reply_generation_prompt.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}"
        )

    return prompt_path.read_text(
        encoding="utf-8"
    )


# ============================================================
# Main agent
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Get customer message
    # --------------------------------------------------------

    customer_message = input(
        "Customer message: "
    ).strip()

    if not customer_message:

        print(
            "Customer message cannot be empty."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Create Groq client
    # --------------------------------------------------------

    client = create_client()

    # --------------------------------------------------------
    # Intent classification
    # --------------------------------------------------------

    print(
        "\n========== INTENT CLASSIFICATION ==========\n"
    )

    try:

        classification = classify_intent(
            client=client,
            message=customer_message,
        )

    except Exception as exc:

        print(
            f"Intent classification failed: "
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)

    intent = classification[
        "intent"
    ]

    intent_confidence = classification[
        "confidence"
    ]

    print(
        f"Intent: {intent}"
    )

    print(
        f"Intent confidence: "
        f"{intent_confidence}"
    )

    # --------------------------------------------------------
    # Historical retrieval
    # --------------------------------------------------------

    print(
        "\n========== HISTORICAL RETRIEVAL ==========\n"
    )

    try:

        results = retrieve(
            customer_message,
            top_k=TOP_K,
        )

    except Exception as exc:

        print(
            f"Retrieval failed: "
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)

    retrieved_cases = (
        build_retrieved_cases(
            results
        )
    )

    print(
        f"Retrieved cases: "
        f"{len(retrieved_cases)}"
    )

    if not retrieved_cases:

        print(
            "No sufficiently similar historical "
            "cases were found."
        )

    else:

        for i, case in enumerate(
            retrieved_cases,
            start=1,
        ):

            print(
                f"\n--- Case {i} "
                f"(similarity="
                f"{case['similarity']:.4f}) ---"
            )

            print(
                "Customer:"
            )

            print(
                case["customer_text"]
            )

            print(
                "\nAppleSupport:"
            )

            print(
                case["brand_text"]
            )

    # --------------------------------------------------------
    # Reply generation
    # --------------------------------------------------------

    print(
        "\n========== REPLY GENERATION ==========\n"
    )

    prompt_template = (
        load_reply_prompt()
    )

    try:

        result = generate_reply(
            client=client,
            prompt_template=prompt_template,
            customer_message=customer_message,
            intent=intent,
            retrieved_cases=retrieved_cases,
        )

    except Exception as exc:

        print(
            f"Reply generation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Extract result
    # --------------------------------------------------------

    draft_reply = result.get(
        "draft_reply",
        "",
    )

    evidence_sufficient = result.get(
        "evidence_sufficient",
        False,
    )

    reply_confidence = result.get(
        "confidence",
        None,
    )

    reason = result.get(
        "reason",
        "",
    )

    # --------------------------------------------------------
    # Auto-handle vs escalation
    # --------------------------------------------------------

    top_similarity = 0.0

    if retrieved_cases:
        top_similarity = max(
            case["similarity"]
            for case in retrieved_cases
        )

    if (
        intent_confidence is not None
        and intent_confidence >= 0.80
        and top_similarity >= 0.30
        and evidence_sufficient is True
    ):
        decision = "AUTO_HANDLE"
        decision_reason = (
            "High intent confidence and sufficiently "
            "similar historical evidence were found."
        )

    else:
        decision = "ESCALATE"

        reasons = []

        if (
            intent_confidence is None
            or intent_confidence < 0.80
        ):
            reasons.append(
                "low intent confidence"
            )

        if top_similarity < 0.30:
            reasons.append(
                "weak historical similarity"
            )

        if not evidence_sufficient:
            reasons.append(
                "insufficient evidence"
            )

        decision_reason = (
            ", ".join(reasons)
            if reasons
            else "Safety threshold not satisfied."
        )

    # --------------------------------------------------------
    # Display final result
    # --------------------------------------------------------

    print(
        "\n========== AGENT RESULT ==========\n"
    )

    print(
        f"Intent: {intent}"
    )

    print(
        f"Intent confidence: "
        f"{intent_confidence}"
    )

    print(
        f"Evidence sufficient: "
        f"{evidence_sufficient}"
    )

    print(
        f"Reply confidence: "
        f"{reply_confidence}"
    )

    print(
    f"Top retrieval similarity: "
    f"{top_similarity:.4f}"
)

    print(
        f"Decision: {decision}"
    )

    print(
        f"Decision reason: "
        f"{decision_reason}"
    )

    print(
        "\nDraft reply:"
    )

    print(
        draft_reply
    )

    print(
        "\nReason:"
    )

    print(
        reason
    )


if __name__ == "__main__":
    main()