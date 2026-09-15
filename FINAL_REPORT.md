# Hiver Support Agent — Final Evaluation Report

## 1. Problem and approach

This project builds a prototype AI customer-support agent for the `AppleSupport` brand from the Customer Support on Twitter dataset.

For each inbound customer message, the system:
1. classifies the message into a small, human-defined intent taxonomy;
2. retrieves similar historical AppleSupport customer/support interactions;
3. generates a reply constrained to the retrieved historical evidence; and
4. decides whether to `AUTO_HANDLE` or `ESCALATE`.

### Architecture

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
Top historical AppleSupport cases
      |
      v
GPT-OSS-120B grounded reply generator
      |
      v
AUTO_HANDLE / ESCALATE
```

## 2. Dataset and labeling

The raw dataset contains **2,811,774 tweets** with customer/brand interaction metadata.

For this project, `AppleSupport` was selected because its interactions provided relatively specific troubleshooting and resolution patterns.

AppleSupport contains approximately:
- **106,860** AppleSupport tweets
- **106,623** unique customer tweets with corresponding brand responses
- **106,646** corrected direct customer -> AppleSupport response pairs

A filtered historical-response dataset produced **33,374** usable retrieval cases.

### Intent taxonomy

The final taxonomy contains 11 intents:

| Intent | Meaning |
|---|---|
| `battery_power` | Battery drain, charging, battery percentage, unexpected shutdown related to power |
| `ios_software_issue` | Broad iOS/system/update instability, crashes, freezing, failed updates |
| `app_service_issue` | Specific app/service malfunction |
| `keyboard_input_issue` | Typing, autocorrect, keyboard, character/symbol rendering |
| `device_hardware_issue` | Physical/device-level malfunction, including failure to power on |
| `account_security` | Apple ID, iCloud account access, authentication, security concerns |
| `activation_setup` | Activation, initial setup, device registration/configuration |
| `feature_how_to` | How to use, enable, disable, configure, or locate a feature |
| `purchase_order` | Purchases, orders, payment/purchasing problems |
| `accessories` | Chargers, AirPods, cables, keyboards, and other accessories |
| `other_unclear` | Vague, incomplete, link-only, image-only, or otherwise underspecified requests |

The training set contains **492 human-verified examples**.

An independently human-labeled **200-example golden set** was created for intent evaluation.

## 3. Baselines and intent classification

A TF-IDF + Logistic Regression classifier was used as a lightweight reference baseline.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 47.97% | 0.421 | 0.451 |
| GPT-OSS-120B | **74.00%** | **0.703** | **0.737** |

GPT-OSS-120B improved over the TF-IDF baseline by:
- **26.03 percentage points** in accuracy
- **0.282** in macro-F1
- **0.286** in weighted-F1

The GPT-OSS-120B figures come from the **200-example human-labeled golden set**.

Because some classes have small support in the golden set, per-class F1 values should be interpreted cautiously. Macro-F1 is more informative than raw accuracy for this taxonomy.

## 4. Historical retrieval

Historical customer/support pairs are indexed with TF-IDF and ranked by cosine similarity.

A **50-example human evaluation** judged whether the highest-ranked historical case (`Result 1`) was relevant to the current customer problem.

**Recall@1: 62.0%**

The top retrieved historical case was judged relevant for **31 of 50** evaluated queries.

This shows that historical interactions can provide grounding material, but lexical retrieval remains imperfect.

## 5. Grounded reply generation

The reply generator receives:
- the customer message;
- the predicted intent; and
- retrieved historical AppleSupport cases.

Its prompt explicitly restricts it to historical evidence and prohibits unsupported troubleshooting steps, guarantees, or invented capabilities.

A **30-example human evaluation** produced:

| Metric | Result |
|---|---:|
| Intent accuracy on this sample | 66.67% |
| Fully grounded | **100.0%** |
| Helpful | **100.0%** |
| Directly addresses problem | **100.0%** |
| Unsupported-claim rate | **3.33%** |

The **3.33% unsupported-claim rate corresponds to 1 of 30 replies**.

The 66.67% intent accuracy is only for the 30-example end-to-end reply sample. It does not replace the dedicated 200-example intent result of 74.0% accuracy / 0.703 macro-F1.

## 6. Auto-handle vs escalation

The prototype currently auto-handles only when all three conditions are satisfied:

```text
intent confidence >= 0.80
AND
top retrieval similarity >= 0.30
AND
evidence_sufficient == True
```

Otherwise it escalates.

A **30-example human routing evaluation** produced:

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

The main safety failure is that **3 of 12 human-escalation cases were incorrectly auto-handled**.

## 7. Human agreement

A second reviewer labeled a **50-example agreement subset** using the same 11-intent taxonomy.

Reported agreement:
- **Raw agreement: 100.0%**
- **Cohen's kappa: 1.00**
- **50/50 labels agreed**

This result should only be reported as independent human agreement if the second reviewer made all 50 decisions without seeing the first reviewer's labels or answer key.

## 8. Top 5 failure modes

### 1. Underspecified follow-ups can be auto-handled too confidently
Messages such as “same problem” or “you didn't fix it” can receive high classifier confidence and strong retrieval similarity even though the current message does not state enough information to safely resolve the issue.

### 2. Retrieval similarity is not the same as resolution sufficiency
A high TF-IDF similarity score can indicate wording overlap rather than genuinely sufficient evidence for solving the current problem.

### 3. A hard similarity threshold creates false escalations
The current `0.30` threshold can reject useful borderline evidence. One evaluated case had strong intent confidence but a similarity of about `0.287`, leading to escalation even though the human reviewer preferred auto-handling.

### 4. Semantically overlapping intents cause classification errors
The most important observed overlaps include:
- `app_service_issue` <-> `ios_software_issue`
- `ios_software_issue` <-> `device_hardware_issue`
- `keyboard_input_issue` <-> `other_unclear`

### 5. Grounded generation can still add unsupported details
Even with an evidence-only prompt, one of 30 evaluated replies was judged to contain an unsupported claim.

## 9. What is misleading about my headline number?

A headline such as:

> “The support agent achieves 74% accuracy.”

would be misleading.

The **74.0%** number is specifically the intent-classification accuracy on a **200-example human-labeled golden set**. It is not the accuracy of the full support agent, and it does not measure retrieval, response quality, or routing.

The other measurements use smaller human-evaluation samples:
- 50 retrieval evaluations
- 30 reply evaluations
- 30 routing evaluations

A more honest headline is:

> **GPT-OSS-120B achieved 0.703 macro-F1 for intent classification on a 200-example human-labeled golden set; the full prototype achieved 86.7% human routing agreement and 100% judged reply groundedness on a 30-example evaluation sample.**

## 10. Decision log

### Brand selection
Selected `AppleSupport` after comparing candidate support brands and observing relatively specific troubleshooting/resolution interactions.

### Intent taxonomy
Merged ambiguous or duplicate categories into an 11-intent taxonomy to keep the label space small enough for practical evaluation.

### Training labels
Used model-assisted labeling followed by human verification, with extra review for underrepresented intents.

### Golden set
Created a separate 200-example golden set from examples excluded from the human-verified training data.

### Retrieval
Used TF-IDF retrieval because it is simple, fast, deterministic, and easy to inspect.

### Reply grounding
Restricted generation to historical AppleSupport evidence rather than allowing free-form troubleshooting.

### Escalation
Used a transparent threshold policy requiring strong intent confidence and sufficiently similar historical evidence.

## 11. Limitations and next improvements

The dataset consists of short, noisy social-media messages and contains many vague follow-ups.

The current retrieval evaluation measures only `Recall@1`; a stronger follow-up would report Recall@3, Recall@5, and MRR.

The routing sample is small, so threshold tuning should be treated as exploratory rather than statistically definitive.

Potential next improvements:
1. add semantic embedding retrieval;
2. add an explicit “problem sufficiently specified” gate before auto-handling;
3. calibrate routing thresholds on a larger validation set; and
4. rerank retrieved cases using intent compatibility in addition to lexical similarity.

## 12. How to run

Configure `GROQ_API_KEY` in `.env`, then run:

```powershell
uv run python src/agent.py
```

Key evaluation scripts:

```text
src/train_baseline.py
src/evaluate_intent.py
src/evaluate_retrieval.py
src/evaluate_replies.py
src/evaluate_escalation.py
```

Key evaluation artifacts:

```text
data/processed/golden_set.csv
data/processed/golden_intent_predictions.csv
data/processed/retrieval_evaluation.csv
data/processed/reply_evaluation.csv
data/processed/escalation_evaluation.csv
data/processed/human_agreement_review.csv
data/processed/human_agreement_answer_key.csv
```
