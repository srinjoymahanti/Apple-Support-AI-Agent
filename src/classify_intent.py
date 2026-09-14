from pathlib import Path
import json

from openai import OpenAI


PROMPT_PATH = Path(
    "configs/labeling_prompt.txt"
)

MODEL = "openai/gpt-oss-120b"


VALID_INTENTS = {
    "battery_power",
    "ios_software_issue",
    "app_service_issue",
    "keyboard_input_issue",
    "device_hardware_issue",
    "account_security",
    "activation_setup",
    "feature_how_to",
    "purchase_order",
    "accessories",
    "other_unclear",
}


def load_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def classify_intent(
    client: OpenAI,
    message: str,
) -> dict:

    prompt = (
        load_prompt()
        + "\n\nCustomer message:\n"
        + message
    )

    response = (
        client
        .chat
        .completions
        .create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
            response_format={
                "type": "json_object"
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    result = json.loads(content)

    intent = result.get(
        "intent"
    )

    confidence = result.get(
        "confidence"
    )

    if intent not in VALID_INTENTS:
        raise ValueError(
            f"Invalid intent: {intent}"
        )

    try:
        confidence = float(
            confidence
        )
    except (
        TypeError,
        ValueError,
    ):
        confidence = None

    return {
        "intent": intent,
        "confidence": confidence,
    }