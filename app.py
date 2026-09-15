import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Existing project modules
# ============================================================

from classify_intent import classify_intent
from retrieve import retrieve
from generate_reply import generate_reply


# ============================================================
# Configuration
# ============================================================

TOP_K = 5

MIN_RETRIEVAL_SIMILARITY = 0.20

INTENT_CONFIDENCE_THRESHOLD = 0.80

TOP_SIMILARITY_THRESHOLD = 0.30

BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide",
)


# ============================================================
# Client creation
# ============================================================

@st.cache_resource
def create_groq_client() -> OpenAI:
    """
    Create the Groq-compatible OpenAI client.

    Local:
        Reads GROQ_API_KEY from .env

    Streamlit Cloud:
        Reads GROQ_API_KEY from st.secrets
    """

    load_dotenv()

    api_key = None

    # Streamlit secrets
    try:
        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )
    except Exception:
        pass

    # Local .env fallback
    if not api_key:
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Add it to .env locally or "
            "Streamlit Secrets when deployed."
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


@st.cache_resource
def load_reply_prompt() -> str:
    """
    Load the grounded reply-generation prompt.
    """

    prompt_path = (
        ROOT_DIR
        / "configs"
        / "reply_generation_prompt.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}"
        )

    return prompt_path.read_text(
        encoding="utf-8"
    )


# ============================================================
# Helper functions
# ============================================================

def build_retrieved_cases(
    results,
) -> list[dict]:
    """
    Convert retriever output into the format expected
    by generate_reply().
    """

    cases = []

    if results.empty:
        return cases

    for _, row in results.iterrows():

        similarity = float(
            row["similarity"]
        )

        if (
            similarity
            < MIN_RETRIEVAL_SIMILARITY
        ):
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


def make_routing_decision(
    intent_confidence,
    top_similarity,
    evidence_sufficient,
):
    """
    Same routing logic used by agent.py.
    """

    if (
        intent_confidence is not None
        and intent_confidence
        >= INTENT_CONFIDENCE_THRESHOLD
        and top_similarity
        >= TOP_SIMILARITY_THRESHOLD
        and evidence_sufficient is True
    ):

        return (
            "AUTO_HANDLE",
            (
                "High intent confidence and sufficiently "
                "similar historical evidence were found."
            ),
        )

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

    return (
        "ESCALATE",
        reason,
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🍎 Apple Support")

    st.markdown(
        """
        **AI customer-support prototype**

        Pipeline:

        `Intent → Retrieval → Reply → Routing`
        """
    )

    st.divider()

    st.subheader("Routing policy")

    st.write(
        f"Intent confidence ≥ "
        f"`{INTENT_CONFIDENCE_THRESHOLD:.2f}`"
    )

    st.write(
        f"Retrieval similarity ≥ "
        f"`{TOP_SIMILARITY_THRESHOLD:.2f}`"
    )

    st.write(
        "Evidence must be sufficient."
    )

    st.divider()

    st.caption(
        "Model: GPT-OSS-120B via Groq"
    )

    st.caption(
        "Retriever: TF-IDF"
    )


# ============================================================
# Main UI
# ============================================================

st.title(
    "🍎 Apple Support AI Agent"
)

st.write(
    "Classify a customer message, retrieve similar "
    "historical AppleSupport cases, generate a grounded "
    "reply, and decide whether to auto-handle or escalate."
)


customer_message = st.text_area(
    "Customer message",
    placeholder=(
        "Example: My battery is draining really fast "
        "after the latest iOS update."
    ),
    height=150,
)


analyze_button = st.button(
    "Analyze & Generate Reply",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Run pipeline
# ============================================================

if analyze_button:

    customer_message = (
        customer_message.strip()
    )

    if not customer_message:

        st.warning(
            "Please enter a customer message."
        )

        st.stop()

    # --------------------------------------------------------
    # Create client
    # --------------------------------------------------------

    try:

        client = create_groq_client()

    except Exception as exc:

        st.error(
            f"Could not initialize API client: {exc}"
        )

        st.stop()

    # --------------------------------------------------------
    # 1. Intent classification
    # --------------------------------------------------------

    with st.spinner(
        "Classifying customer intent..."
    ):

        try:

            classification = classify_intent(
                client=client,
                message=customer_message,
            )

        except Exception as exc:

            st.error(
                f"Intent classification failed: {exc}"
            )

            st.stop()

    intent = classification[
        "intent"
    ]

    intent_confidence = classification[
        "confidence"
    ]


    # --------------------------------------------------------
    # 2. Historical retrieval
    # --------------------------------------------------------

    with st.spinner(
        "Searching historical AppleSupport cases..."
    ):

        try:

            retrieval_results = retrieve(
                customer_message,
                top_k=TOP_K,
            )

        except Exception as exc:

            st.error(
                f"Retrieval failed: {exc}"
            )

            st.stop()

    retrieved_cases = (
        build_retrieved_cases(
            retrieval_results
        )
    )

    if retrieved_cases:

        top_similarity = max(
            case["similarity"]
            for case in retrieved_cases
        )

    else:

        top_similarity = 0.0


    # --------------------------------------------------------
    # 3. Reply generation
    # --------------------------------------------------------

    with st.spinner(
        "Generating grounded reply..."
    ):

        try:

            prompt_template = (
                load_reply_prompt()
            )

            reply_result = generate_reply(
                client=client,
                prompt_template=prompt_template,
                customer_message=customer_message,
                intent=intent,
                retrieved_cases=retrieved_cases,
            )

        except Exception as exc:

            st.error(
                f"Reply generation failed: {exc}"
            )

            st.stop()

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


    # --------------------------------------------------------
    # 4. Routing
    # --------------------------------------------------------

    decision, decision_reason = (
        make_routing_decision(
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


    # ========================================================
    # Results layout
    # ========================================================

    st.divider()

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    st.subheader(
        "1. Intent Classification"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Intent",
            intent,
        )

    with col2:

        if intent_confidence is not None:

            st.metric(
                "Confidence",
                f"{float(intent_confidence):.2f}",
            )

        else:

            st.metric(
                "Confidence",
                "N/A",
            )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    st.subheader(
        "2. Historical Evidence"
    )

    st.write(
        f"Usable historical cases: "
        f"**{len(retrieved_cases)}**"
    )

    st.write(
        f"Top similarity: "
        f"**{top_similarity:.4f}**"
    )

    if not retrieved_cases:

        st.info(
            "No sufficiently similar historical "
            "cases were found."
        )

    else:

        for i, case in enumerate(
            retrieved_cases,
            start=1,
        ):

            with st.expander(
                f"Case {i} — "
                f"Similarity {case['similarity']:.4f}"
            ):

                st.markdown(
                    "**Customer:**"
                )

                st.write(
                    case["customer_text"]
                )

                st.markdown(
                    "**AppleSupport:**"
                )

                st.write(
                    case["brand_text"]
                )


    # --------------------------------------------------------
    # Reply
    # --------------------------------------------------------

    st.subheader(
        "3. Generated Reply"
    )

    st.info(
        draft_reply
        if draft_reply
        else "No draft reply generated."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Evidence sufficient",
            str(evidence_sufficient),
        )

    with col2:

        if reply_confidence is not None:

            st.metric(
                "Reply confidence",
                f"{float(reply_confidence):.2f}",
            )

        else:

            st.metric(
                "Reply confidence",
                "N/A",
            )

    with col3:

        st.metric(
            "Retrieved cases",
            len(retrieved_cases),
        )

    if generator_reason:

        st.caption(
            f"Generator reason: {generator_reason}"
        )


    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    st.subheader(
        "4. Agent Decision"
    )

    if decision == "AUTO_HANDLE":

        st.success(
            f"✅ {decision}"
        )

    else:

        st.warning(
            f"⚠️ {decision}"
        )

    st.write(
        f"**Reason:** {decision_reason}"
    )


    # --------------------------------------------------------
    # Technical details
    # --------------------------------------------------------

    with st.expander(
        "Technical details"
    ):

        st.write(
            {
                "intent": intent,
                "intent_confidence": intent_confidence,
                "top_similarity": top_similarity,
                "retrieved_cases": len(
                    retrieved_cases
                ),
                "evidence_sufficient": (
                    evidence_sufficient
                ),
                "reply_confidence": (
                    reply_confidence
                ),
                "agent_decision": decision,
            }
        )