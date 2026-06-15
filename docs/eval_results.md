# ATHENA-R1 Evaluation Results

Headline benchmark results for the **`mims-harvard/ATHENA-R1-Qwen3-8B`** release
model, as reported in the paper. Models are evaluated in an open-ended setting:
each question is answered free-form, then mapped to one of the original answer
choices.

## DrugPC — open-ended drug reasoning

3,168 treatment questions derived from FDA drug labels, spanning 11 categories
(indications, dosing, warnings and safety, pharmacology, etc.). Drugs approved
by the FDA in 2024 are held out from training to reduce pretraining leakage.

| Model | Accuracy |
|---|---|
| **ATHENA-R1** | **94.7%** |
| GPT-5 | 76.9% |
| DeepSeek-R1 (671B) | 68.8% |
| Qwen3 | 48.7% |

ATHENA-R1 exceeds GPT-5 by 17.8, DeepSeek-R1 by 25.9, and Qwen3 by 46.0
percentage points.

## TreatmentPC — patient-specific treatment selection

456 questions in which the correct treatment depends on patient context —
pregnancy, comorbidity, dosing constraints or a contraindicated co-medication.

| Model | Accuracy |
|---|---|
| **ATHENA-R1** | **82.9%** |
| GPT-5 | 72.2% |
| DeepSeek-R1 (671B) | 67.5% |
| Qwen3 | 39.2% |
| ToolACE-8B (tool-use) | 13.4% |
| WattTool-8B (tool-use) | 5.9% |

ATHENA-R1 exceeds GPT-5 by 10.7 and DeepSeek-R1 by 15.4 percentage points.
Access to the tool library alone is not enough: tool-use LLMs given the full
library score far lower, and GPT-5 with tool access does not close the gap.

## Effect of two-level self-learning

Contribution of each training level on TreatmentPC. The model answers free-form,
then selects the option that best matches its own response:

| Stage | Accuracy |
|---|---|
| Qwen3-8B base | 39.2% |
| + supervised fine-tuning | 66.5% (+27.3) |
| + reinforcement learning | 74.8% (+8.3) |

## Inference settings

| Parameter | Value |
|---|---|
| temperature | 0.7 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 0 |
| max_round | 40 |

## Reproducibility

These results use the ATHENA-R1-Qwen3-8B release model served with vLLM and the
ToolUniverse tool server. To run the benchmarks:

1. Launch the ToolUniverse server (`scripts/launch_tooluniverse.sh`).
2. Launch vLLM with the release model (`scripts/launch_vllm.sh`).
3. Run the benchmark scripts in `examples/`.

See [`examples/eval_mcq.py`](../examples/eval_mcq.py) for the per-question
pattern; for bulk evaluation against an MCQ dataset, loop over `agent.answer()`
+ `agent.map_to_option()`.
