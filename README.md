# Behavior Priming for Agentic Search

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.06534-b31b1b.svg)](https://arxiv.org/abs/2510.06534)
[![arXiv](https://img.shields.io/badge/arXiv-2507.05495-b31b1b.svg)](https://arxiv.org/abs/2507.05495)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

This repository contains the agent scaffold introduced in the **[Deep Research Comparator paper](https://arxiv.org/abs/2507.05495)**, and the behavior-priming, SFT and evaluation toolkit in the **[Behavior Priming paper](https://arxiv.org/abs/2510.06534)**. For the RL training in the Behavior Priming paper, please refer to [this repository](https://github.com/cxcscmu/verl-agent-deepresearch).

## Overview

We study **what reasoning behaviors make search agents succeed** and how to reliably instill them via **behavior-targeted post-training**.

Key components in this repo:
- DeepResearch agent scaffold. The long-report mode is used in the **[Deep Research Comparator paper](https://arxiv.org/abs/2507.05495)**, and the short-answer mode is used in the **[Behavior Priming paper](https://arxiv.org/abs/2510.06534)**.
- Behavior analysis pipeline for identifying reasoning behaviors from trajectories.
- SFT recipes for injecting reasoning behaviors into models.
- Evaluation suite on web agent tasks and multi-hop QA tasks.

## Setup

### Agent Scaffold, Evaluation and Behavior Priming

#### Option 1: Using Python venv

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Conda

```bash
# Create a conda environment
conda create -n deepresearch python=3.10
conda activate deepresearch

# Install dependencies
pip install -r requirements.txt
```

#### Environment Variables

Create a `keys.env` file in the root directory with your API keys:

```bash
# Model Deployment Keys
GEMINI_API_KEY=your_gemini_api_key                    # Required if using Gemini models as the underlying LLM (default option in our agent scaffold)
AWS_ACCESS_KEY_ID=your_aws_access_key_id              # Required if using Amazon Bedrok models as the underlying LLM 
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key       # Required if using Amazon Bedrok models as the underlying LLM 
AWS_BEARER_TOKEN_BEDROCK=your_aws_bearer_token        # Required if using Amazon Bedrok models as the underlying LLM

# Evaluation Keys
OPENAI_API_KEY=your_openai_api_key                    # Required for evaluation script

# Search Engine Keys
SERPER_API_KEY=your_serper_api_key                     # Required if using --search_engine serper
CLUEWEB_API_KEY=your_clueweb_api_key                   # Required if using --search_engine clueweb
```

### SFT

Please refer to the setup in the official [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository.

## Run the Agent

### 1. Prepare Data

Please provide the agent input datasets following the schema used in `evaluation/short/gaia/test.json`. The required fields are:

```json
{
  "id": "123",
  "question": "What is the capital of ...?",
  "answer": "Paris"
}
```
### 2. Model Deployment

We recommend use vLLM to serve the underlying model of agent scaffold. For Qwen3 series model, please enable it's internal thinking mode. For models without built-in reasoning, you should add `use_explicit_thinking` flag when inference. See the next section for more details.

Example scripts:
- [Here](https://github.com/cxcscmu/Behavior_Priming_For_Agentic_Search/blob/main/scripts/serve_qwen.sh) is an example script to serve the Qwen3 model.
- [Here](https://github.com/cxcscmu/Behavior_Priming_For_Agentic_Search/blob/main/scripts/serve_llama.sh) is an example script to serve Llama3.2 model.

---

### 3. Launch Agent Generation

The agent supports both long-report and short-answer generation modes.
- In long-report mode, the agent is prompted to produce a comprehensive research report for the given question.
- In short-answer mode, the agent provides a concise response.

```bash
python3 main_parallel.py \
    --batch_file <path_to_dataset> \
    --answer_dir <output_answers> \
    --log_dir <output_logs> \
    [--is_qwen | --is_llama | --is_bedrock] \
    [--use_explicit_thinking] \
    --search_engine <clueweb|serper>
    --max_turns <max_turns_for_agent>
```

Important flags:

| Flag | Description |
| ---- | --- |
| `--batch_file` | Input dataset |
| `--answer_dir` | Output directories for answers |
|  `--log_dir`   | Output directories for logs (*.jsonl files and *.md files for agent trajectories, and search logs) |
| `--long_report` | Enables long-report mode; otherwise defaults to short-answer mode. |
| `--is_qwen` / `--is_llama`/ `--is_bedrock` | If not using Gemini model (default), selecting other underlying LLM for agent scaffold. |
| `--use_explicit_thinking` | CoT prompting for models without built-in reasoning (e.g., Llama3.2-3B-Instruct) |
| `--search_engine` | Search backend (e.g., clueweb, serper, etc.) |
| `--url` | Custom vLLM server endpoint for local deployment of underlyLLM (required if choosing `--is_qwen` or `--is_llama`). |
| `--max_turns` | Maximum number of agent interaction turns. |

Example scripts:
- [Here](https://github.com/cxcscmu/Behavior_Priming_For_Agentic_Search/blob/main/scripts/run_qwen.sh) is an example script to run the Qwen3 model.
- [Here](https://github.com/cxcscmu/Behavior_Priming_For_Agentic_Search/blob/main/scripts/run_llama.sh) is an example script to run Llama3.2 model.

---

## Behavior Analysis Pipeline 


### 1. Behavior Identification

Step 1. **Reasoner** – compare successful vs failed runs.

```bash
python behaviour_analysis/reasoner.py \
  --results_dir results/... \
  --logs_dir logs/... \
  --question_file evaluation/...json \
  --use_llm
```

Step 2. **Extractor** – mine behavior candidates per question.

```bash
python behaviour_analysis/extractor.py \
  --input_suffix 2 \
  --output_suffix 2
```

Step 3. **Merger** – consolidate behaviors across the corpus.

```bash
python behaviour_analysis/merger.py
```

### 2.Behavior Frequency Analysis 

```bash
bash behaviour_analysis/analysis.sh
```
---

## SFT Training 

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning (SFT).

1.	Collect logs from agent runs (the *.json files under logs_dir).
2.	Filter for the targeted trajectories.
3.	Merge and convert the *.json files into the SFT format:

```bash
python train/sft/data_reorganize.py
```

4.	Configure the training settings (train/sft/sft*.yaml) and launch training:
  
```bash
bash scripts/train.sh
```

For more configuration options, please refer to the official [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository.

---

## Evaluation 

```bash
python evaluation/short/evaluation.py \
  --data_path evaluation/short/webwalkerqa/test.json \
  --results_dir results/webwalkerqa/run1 \
  [--mhqa] \
  [--enable_hack_detection] \
  [--rerun]
```

- `--mhqa`: For multi-hop QA scoring.  
- `--enable_hack_detection`: Activates LLM-based hack detection. Any answer flagged as hacking will receive a score of 0.
- `--rerun`: Recomputes all metrics on cached outputs.

Example scripts:
- [Here](https://github.com/cxcscmu/Behavior_Priming_For_Agentic_Search/blob/main/evaluation/short/evaluation.sh) is an example script to evaluate on the web agent benchmarks and multi-hop QA benchmarks used in the Behavior Priming paper.

---


## Citation

If you find this work helpful, please consider citing:

- **Deep Research Comparator**

```bibtext
@misc{chandrahasan2025deepresearchcomparatorplatform,
      title={Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents}, 
      author={Prahaladh Chandrahasan and Jiahe Jin and Zhihan Zhang and Tevin Wang and Andy Tang and Lucy Mo and Morteza Ziyadi and Leonardo F. R. Ribeiro and Zimeng Qiu and Markus Dreyer and Akari Asai and Chenyan Xiong},
      year={2025},
      eprint={2507.05495},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.05495}, 
}
```
- **Beneficial Reasoning Behaviors in Agentic Search and Effective Post-Training to Obtain Them** 

```bibtex
@article{jin2025beneficial,
  title   = {Beneficial Reasoning Behaviors in Agentic Search and Effective Post-Training to Obtain Them},
  author  = {Jiahe Jin and Abhijay Paladugu and Chenyan Xiong},
  year    = {2025},
  journal = {arXiv preprint arXiv:2510.06534},
  url     = {https://arxiv.org/abs/2510.06534}
}
```
