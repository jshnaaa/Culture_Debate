# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the implementation of the ACL 2025 paper "Multiple LLM Agents Debate for Equitable Cultural Alignment". The codebase evaluates how multiple LLM agents can collaborate through debate to improve cultural alignment across 75 countries using the NORMAD-ETI dataset.

## Architecture

### Core Components

**Multi-Agent Debate Framework (`multi_llm/`)**
- Each file represents a specific model pair combination (e.g., `gemma_llama3.py`, `aya_seallm.py`)
- Implements a 6-step debate protocol: initial decisions → feedback exchange → final decisions
- All combinations follow the same pattern but load different model pairs

**Single LLM Baselines (`single_llm/`)**
- `single_model/`: Direct inference with/without Rule-of-Thumb (RoT) prompting
- `self_reflection/`: 3-step self-reflection process (initial → reflection → final)

**Evaluation (`evaluate/`)**
- `accuracy_multi.py`: Evaluates multi-agent debate results
- `accuracy_single.py`: Evaluates single model baselines
- Both compute overall accuracy and per-culture group metrics

### Key Shared Components

**Prompt Templates (`*/prompt.py`)**
- Multi-agent: `make_initial_decision`, `give_feedback`, `make_final_decision`
- Self-reflection: `make_initial_decision`, `generate_self_reflection`, `make_final_decision`
- Single model: `make_decision_with_rot`, `make_decision_without_rot`

**Utilities (`*/utils.py`)**
- Country name mappings (75 countries)
- Response parsing functions
- ISO language code mappings

## Development Commands

### Running Multi-Agent Debates
```bash
python -u multi_llm/{model1}_{model2}.py \
  --input_path data/normad.jsonl \
  --output_path output/{model1}_{model2}_results.jsonl
```

### Running Single Model Baselines
```bash
# With Rule-of-Thumb
python -u single_llm/single_model/{model}.py \
  --input_path data/normad.jsonl \
  --output_path output/{model}_with_rot.jsonl \
  --type with_rot

# Without Rule-of-Thumb
python -u single_llm/single_model/{model}.py \
  --input_path data/normad.jsonl \
  --output_path output/{model}_without_rot.jsonl \
  --type without_rot
```

### Running Self-Reflection
```bash
python -u single_llm/self_reflection/{model}.py \
  --input_path data/normad.jsonl \
  --output_path output/{model}_reflection.jsonl
```

### Evaluation
```bash
# Multi-agent results (modify FIRST_MODEL and SECOND_MODEL variables)
python evaluate/accuracy_multi.py

# Single model results (modify MODEL_NAMES variable)
python evaluate/accuracy_single.py
```

## Model Configuration

### Supported Models
- `aya`: Aya model for multilingual cultural understanding
- `seallm`: SeaLLM for Southeast Asian contexts
- `gemma`: Google Gemma-2-9B-IT
- `llama3`: Meta LLaMA-3-8B-Instruct
- `internlm`: InternLM from Shanghai AI Lab
- `qwen`: Alibaba Qwen series
- `yi`: Yi model from 01.AI
- `exaone`: LG AI Research Exaone

### Hardware Requirements
- GPU with sufficient VRAM for loading two 8-9B models simultaneously (multi-agent)
- Models use `torch.bfloat16` precision and `device_map="auto"`
- HuggingFace cache directory configurable via `own_cache_dir` variable

## Data Format

**Input (NORMAD-ETI dataset)**
```json
{
  "ID": 0,
  "Country": "egypt",
  "Background": "### Basic Etiquette...",
  "Rule-of-Thumb": "It is respectful to greet everyone present...",
  "Story": "At a gathering...",
  "Gold Label": "yes"
}
```

**Output Format**
- Multi-agent: Includes all 6 debate steps plus final decisions from both models
- Single model: Includes initial decision and final answer
- Self-reflection: Includes initial decision, reflection, and final decision

## Authentication

All model scripts require HuggingFace authentication:
- Set `hf_token` variable in each script
- Or configure HuggingFace CLI: `huggingface-cli login`

## Key Implementation Details

### Debate Protocol
1. **Model1 Initial Decision**: First model evaluates cultural scenario
2. **Model2 Initial Decision**: Second model evaluates independently
3. **Model1 Feedback**: First model responds to second model's decision
4. **Model2 Feedback**: Second model responds to first model's decision
5. **Model1 Final**: First model makes final decision based on full discussion
6. **Model2 Final**: Second model makes final decision based on full discussion

### Response Constraints
- Initial decisions: Yes/No/Neither + explanation (≤3 sentences)
- Feedback: ≤3 sentences
- Final decisions: Yes/No/Neither only

### Evaluation Metrics
- Overall accuracy across all 2,633 test cases
- Cultural group parity using ISO language code groupings
- Per-model accuracy in multi-agent settings