# Apple Support AI Agent

An end-to-end AI customer-support agent built for the Hiver SDE Intern take-home assignment using historical `AppleSupport` interactions from the Customer Support on Twitter dataset.

## Live Demo

**Streamlit:** `https://hiver-support-agent-srinjoy-mahanti.streamlit.app/`

## Overview

For each customer message, the system:

1. Classifies the support intent.
2. Retrieves similar historical AppleSupport cases.
3. Generates a reply grounded in those cases.
4. Decides whether to `AUTO_HANDLE` or `ESCALATE`.

```text
Customer message
      |
      v
GPT-OSS-120B intent classifier
      |
      v
TF-IDF historical retriever
      |
      v
Historical AppleSupport evidence
      |
      v
GPT-OSS-120B grounded reply
      |
      v
AUTO_HANDLE / ESCALATE
```

## Dataset

The project uses the Customer Support on Twitter dataset.

AppleSupport subset:

- Raw dataset: **2,811,774 tweets**
- AppleSupport tweets: **106,860**
- Unique customer tweets with responses: **106,623**
- Corrected customer -> AppleSupport pairs: **106,646**
- Usable historical retrieval cases: **33,374**
- Human-verified training examples: **492**
- Human-labeled golden evaluation examples: **200**

The raw dataset is not required by the deployed Streamlit app.

## Intent Taxonomy

The final taxonomy contains 11 intents:

| Intent | Description |
|---|---|
| `battery_power` | Battery drain, charging, battery percentage, unexpected shutdown related to power |
| `ios_software_issue` | Broad iOS/system/update instability, crashes, freezing, failed updates |
| `app_service_issue` | Specific app or service malfunction |
| `keyboard_input_issue` | Typing, autocorrect, keyboard, character/symbol rendering |
| `device_hardware_issue` | Physical/device-level malfunction, including failure to power on |
| `account_security` | Apple ID, iCloud account access, authentication, security concerns |
| `activation_setup` | Activation, initial setup, registration, configuration |
| `feature_how_to` | How to use, enable, disable, configure, or locate a feature |
| `purchase_order` | Purchases, orders, payment/purchasing problems |
| `accessories` | Chargers, AirPods, cables, keyboards, and other accessories |
| `other_unclear` | Vague, incomplete, link-only, image-only, or underspecified requests |

## Intent Classification

A TF-IDF + Logistic Regression model provides a lightweight baseline.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 47.97% | 0.421 | 0.451 |
| GPT-OSS-120B | **74.00%** | **0.703** | **0.737** |

GPT-OSS-120B was evaluated on the 200-example human-labeled golden set.

Relative to the baseline, GPT-OSS-120B improved:

- Accuracy by **26.03 percentage points**
- Macro-F1 by **0.282**
- Weighted-F1 by **0.286**

## Historical Retrieval

Historical customer/support pairs are indexed with TF-IDF and ranked by cosine similarity.

A 50-example human evaluation measured whether the top-ranked historical case was relevant.

**Recall@1: 62.0%**

31 of 50 top-ranked cases were judged relevant.

> Recall@1 measures retrieval relevance only; it is not end-to-end agent accuracy.

## Grounded Reply Generation

The reply generator receives:

- Customer message
- Predicted intent
- Retrieved historical AppleSupport cases

The generation prompt restricts responses to the supplied historical evidence and discourages invented troubleshooting steps, guarantees, and unsupported claims.

Results from a 30-example human evaluation:

| Metric | Result |
|---|---:|
| Intent accuracy on this sample | 66.67% |
| Fully grounded | **100.0%** |
| Helpful | **100.0%** |
| Addresses customer problem | **100.0%** |
| Unsupported-claim rate | **3.33%** |

The unsupported-claim rate was 1 of 30 replies.

The 66.67% intent accuracy is specific to this end-to-end sample; the dedicated 200-example intent result is 74.0% accuracy / 0.703 macro-F1.

## Auto-Handle vs Escalation

The routing policy is:

```text
intent confidence >= 0.80
AND
top retrieval similarity >= 0.30
AND
evidence_sufficient == True
    -> AUTO_HANDLE

otherwise
    -> ESCALATE
```

30-example human routing evaluation:

| Metric | Result |
|---|---:|
| Routing agreement | **86.67%** |
| Escalation precision | **90.00%** |
| Escalation recall | **75.00%** |
| Human AUTO_HANDLE | 18 |
| Human ESCALATE | 12 |
| Agent AUTO_HANDLE | 20 |
| Agent ESCALATE | 10 |

Confusion matrix:

| Human \ Agent | AUTO_HANDLE | ESCALATE |
|---|---:|---:|
| AUTO_HANDLE | 17 | 1 |
| ESCALATE | 3 | 9 |

The main safety failure was 3 cases that humans wanted escalated but the agent auto-handled.

## Human Agreement

A second reviewer independently labeled 50 examples using the same 11-intent taxonomy.

- **Raw agreement: 100.0%**
- **Cohen's kappa: 1.00**
- **50/50 labels agreed**

This assumes the second review was completed without access to the first reviewer's labels.

## Top Failure Modes

### 1. Underspecified follow-ups can be overconfidently auto-handled

Messages such as “same problem” or “you didn't fix it” can receive high confidence and strong retrieval similarity even when the current message lacks enough context for safe automation.

### 2. Retrieval similarity is not evidence sufficiency

TF-IDF similarity measures textual overlap, not whether the retrieved support response is sufficient to resolve the current case.

### 3. A hard similarity threshold can create false escalations

The 0.30 threshold can reject useful borderline evidence. One evaluated case had high intent confidence but similarity around 0.287 and was escalated even though a human preferred auto-handling.

### 4. Semantically overlapping intents are difficult to separate

Important overlaps include:

```text
app_service_issue <-> ios_software_issue
ios_software_issue <-> device_hardware_issue
keyboard_input_issue <-> other_unclear
```

### 5. Grounded generation can still make unsupported additions

Even with evidence-restricted generation, 1 of 30 evaluated replies contained an unsupported claim.

## What Is Misleading About the Headline Number?

A statement such as:

> “The support agent achieves 74% accuracy.”

would be misleading.

The **74.0%** figure is specifically intent-classification accuracy on a 200-example golden set. It is not the accuracy of the full support agent.

The other components were evaluated separately on smaller samples:

- Retrieval: 50 examples
- Reply generation: 30 examples
- Routing: 30 examples

A more accurate summary is:

> **GPT-OSS-120B achieved 0.703 macro-F1 for intent classification on a 200-example human-labeled golden set; the full prototype achieved 86.7% human routing agreement and 100% judged reply groundedness on a 30-example evaluation sample.**

These are prototype evaluation results, not production-level guarantees.

## Project Structure

```text
hiver-support-agent/
|
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
|
├── configs/
|   ├── labeling_prompt.txt
|   └── reply_generation_prompt.txt
|
├── src/
|   ├── agent.py
|   ├── classify_intent.py
|   ├── retrieve.py
|   ├── generate_reply.py
|   ├── train_baseline.py
|   ├── evaluate_intent.py
|   ├── evaluate_retrieval.py
|   ├── evaluate_replies.py
|   └── evaluate_escalation.py
|
└── data/
    └── processed/
        ├── retrieval_index_data.csv
        ├── retrieval_tfidf_vectorizer.joblib
        └── retrieval_tfidf_matrix.joblib
```

## Setup

### 1. Clone

```bash
git clone https://github.com/srinjoymahanti/hiver-support-agent.git
cd hiver-support-agent
```

### 2. Install dependencies

With `uv`:

```bash
uv sync
```

Or:

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

For local development, create `.env` in the project root:

```text
GROQ_API_KEY=your_api_key
```

Never commit `.env` or expose the API key in source code.

## Run the CLI Agent

```bash
uv run python src/agent.py
```

## Run the Streamlit App

```bash
uv run streamlit run app.py
```

The Streamlit UI exposes:

- Predicted intent
- Intent confidence
- Historical evidence
- Generated reply
- Evidence sufficiency
- Reply confidence
- Final routing decision
- Routing reason

## Streamlit Deployment

The app can be deployed from GitHub using Streamlit Community Cloud.

Add the API key in Streamlit Secrets:

```toml
GROQ_API_KEY = "your_api_key"
```

Do not store the API key in GitHub.

The deployed app needs the processed retrieval artifacts, but not the original raw dataset.

## Evaluation Artifacts

```text
data/processed/golden_set.csv
data/processed/golden_intent_predictions.csv
data/processed/retrieval_evaluation.csv
data/processed/reply_evaluation.csv
data/processed/escalation_evaluation.csv
data/processed/human_agreement_review.csv
data/processed/human_agreement_answer_key.csv
```

## Design Decisions

### Why AppleSupport?

AppleSupport interactions provided relatively specific troubleshooting and resolution patterns suitable for historical evidence retrieval.

### Why TF-IDF?

TF-IDF is simple, fast, deterministic, and easy to inspect, making it a practical baseline for a take-home prototype.

### Why historical grounding?

The goal is not unrestricted chatbot generation. Replies are constrained by real historical AppleSupport interactions to reduce unsupported troubleshooting.

### Why explicit routing thresholds?

The routing policy is transparent and auditable rather than being an opaque LLM-only decision.

## Limitations and Future Improvements

- Short and noisy social-media messages make intent and retrieval difficult.
- Many customer messages are vague follow-ups.
- Some taxonomy classes have limited evaluation support.
- Retrieval evaluation currently reports Recall@1; Recall@3, Recall@5, and MRR would provide a richer picture.
- Routing evaluations use relatively small samples.

Potential improvements:

1. Add embedding-based semantic retrieval.
2. Add an explicit “problem sufficiently specified” gate before auto-handling.
3. Calibrate routing thresholds on a larger validation set.
4. Rerank historical cases using intent compatibility in addition to similarity.

