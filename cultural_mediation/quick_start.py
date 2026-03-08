"""
Quick Start Example: Cultural Conflict Mediation System (Unified Model)
All agents share the same model (Llama3.1-8B or Qwen3-8B)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.unified_model_manager import UnifiedModelManager
from utils.weight_learner import CountryWeightLearner
from utils.normad_loader import NORMADLoader
from sentence_transformers import SentenceTransformer


def parse_agent_answer(response: str) -> str:
    """
    Parse agent response to extract answer (1/2/3)

    Args:
        response: Agent's generated response

    Returns:
        "1", "2", or "3"
    """
    # Look for explicit answer format
    response_lower = response.lower()

    # Check for "answer: 1/2/3" pattern
    if "answer:" in response_lower:
        answer_part = response_lower.split("answer:")[-1].strip()
        if "1" in answer_part[:10]:
            return "1"
        elif "2" in answer_part[:10]:
            return "2"
        elif "3" in answer_part[:10]:
            return "3"

    # Check for first occurrence of 1/2/3
    for char in response[:100]:
        if char in ["1", "2", "3"]:
            return char

    # Default to neutral
    return "3"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Cultural Conflict Mediation System - Quick Start"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "qwen"],
        help="Model to use: llama (Llama3.1-8B) or qwen (Qwen3-8B)"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to test (default: 0)"
    )
    parser.add_argument(
        "--no_batch",
        action="store_true",
        help="Generate one agent at a time instead of batch (saves ~3-4 GB VRAM)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Cultural Conflict Mediation System - Quick Start")
    print(f"Model: {args.model.upper()}")
    print("=" * 80)

    # ============================================================================
    # Step 1: Initialize Model Manager
    # ============================================================================
    print("\n[Step 1] Initializing model manager...")

    model_mgr = UnifiedModelManager(model_name=args.model)

    # Load weight learner
    weight_path = Path(__file__).parent / "data" / "country_weights_init.json"
    weight_learner = CountryWeightLearner(init_weights_path=weight_path)
    print(f"✓ Loaded weights for {weight_learner.num_countries} countries")

    # ============================================================================
    # Step 2: Load NORMAD Sample
    # ============================================================================
    print("\n[Step 2] Loading NORMAD dataset...")

    dataset_path = "/root/autodl-fs/normad_merge_gen.json"
    loader = NORMADLoader(dataset_path)

    # Get sample
    sample = loader.get_sample(args.sample_idx)
    sample_dict = sample.to_dict()

    print(f"✓ Testing sample #{args.sample_idx}")
    print(f"✓ Country: {sample.country}")
    print(f"✓ Gold label: {sample.gold_label} (1=yes, 2=no, 3=neutral)")
    print(f"✓ Story: {sample.story[:100]}...")

    # ============================================================================
    # Step 3: Get Country Weights
    # ============================================================================
    print("\n[Step 3] Getting country weights...")

    country_weights = weight_learner.get_country_weight(sample.country)
    print(f"✓ Weights: {country_weights.tolist()}")
    print(f"  [Autonomy, Order&Security, Tradition, Care&Universalism, Achievement&Power]")

    # ============================================================================
    # Step 4: Generate Cultural Agent Responses
    # ============================================================================
    print(f"\n[Step 4] Generating cultural agent responses ({args.model})...")

    dimensions = ["Autonomy", "Order & Security", "Tradition",
                 "Care & Universalism", "Achievement & Power"]

    # Build prompts for all 5 agents
    prompts = []
    for i, dimension in enumerate(dimensions):
        weight = country_weights[i].item()

        prompt = f"""You are a cultural perspective agent with the following value priorities:
- Autonomy (self-direction, independence): {country_weights[0]:.2f}
- Order & Security (stability, safety): {country_weights[1]:.2f}
- Tradition (respect for customs): {country_weights[2]:.2f}
- Care & Universalism (welfare, equality): {country_weights[3]:.2f}
- Achievement & Power (success, influence): {country_weights[4]:.2f}

Your PRIMARY focus is on: {dimension} (weight: {weight:.2f})

Cultural Context:
{sample_dict['background']}

Rule-of-Thumb:
{sample_dict['rule_of_thumb']}

Scenario:
{sample_dict['story']}

Based on your value priorities (especially {dimension}), is the action socially acceptable?
Answer with one of the following:
1 - yes (socially acceptable)
2 - no (not socially acceptable)
3 - neutral (depends on context)

Provide your answer and a brief explanation (≤3 sentences, focusing on {dimension}):
Answer:"""

        prompts.append(prompt)

    # Generate responses: batch (faster) or sequential (lower VRAM)
    if args.no_batch:
        print("  Generating 5 agent responses sequentially (low VRAM mode)...")
        agent_responses_raw = [
            model_mgr.generate(p, max_new_tokens=200, temperature=0.0)
            for p in prompts
        ]
    else:
        print("  Generating batch of 5 agent responses...")
        agent_responses_raw = model_mgr.batch_generate(
            prompts,
            max_new_tokens=200,
            temperature=0.0
        )

    # Parse answers
    agent_responses = []
    for i, (dimension, response_raw) in enumerate(zip(dimensions, agent_responses_raw)):
        answer = parse_agent_answer(response_raw)

        agent_responses.append({
            "dimension": dimension,
            "weight": country_weights[i].item(),
            "answer": answer,
            "explanation": response_raw
        })

        answer_text = {"1": "yes", "2": "no", "3": "neutral"}[answer]
        print(f"  Agent {i+1} ({dimension}, w={country_weights[i].item():.2f}): "
              f"{answer} ({answer_text})")

    # ============================================================================
    # Step 5: Compute Conflict Score (Simplified)
    # ============================================================================
    print("\n[Step 5] Analyzing conflict...")

    # Simple conflict: check answer variance
    answers = [resp["answer"] for resp in agent_responses]
    unique_answers = set(answers)
    conflict_score = (len(unique_answers) - 1) / 2.0  # 0 if all same, 1 if all different

    print(f"✓ Answer distribution: {dict((a, answers.count(a)) for a in unique_answers)}")
    print(f"✓ Conflict score: {conflict_score:.3f} (0=consensus, 1=max disagreement)")

    # ============================================================================
    # Step 6: Final Decision (Weighted Voting)
    # ============================================================================
    print("\n[Step 6] Computing final decision...")

    # Weighted voting
    votes = {"1": 0.0, "2": 0.0, "3": 0.0}
    for i, resp in enumerate(agent_responses):
        votes[resp["answer"]] += country_weights[i].item()

    final_answer = max(votes, key=votes.get)

    print(f"✓ Vote distribution:")
    for answer, weight in votes.items():
        answer_text = {"1": "yes", "2": "no", "3": "neutral"}[answer]
        print(f"  {answer} ({answer_text}): {weight:.3f}")

    print(f"\n✓ Final answer: {final_answer} ({{'1': 'yes', '2': 'no', '3': 'neutral'}}[final_answer]})")
    print(f"✓ Gold label: {sample.gold_label} ({{'1': 'yes', '2': 'no', '3': 'neutral'}}[sample.gold_label]})")

    # Check correctness
    correct = final_answer == sample.gold_label
    print(f"✓ Prediction: {'✓ CORRECT' if correct else '✗ INCORRECT'}")

    # ============================================================================
    # Step 7: Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Sample: #{args.sample_idx}")
    print(f"Country: {sample.country}")
    print(f"Conflict Score: {conflict_score:.3f}")
    print(f"Final Answer: {final_answer}")
    print(f"Gold Label: {sample.gold_label}")
    print(f"Accuracy: {'✓ Correct' if correct else '✗ Incorrect'}")

    # Memory usage
    print("\n" + "=" * 80)
    print("GPU Memory Usage")
    print("=" * 80)
    model_mgr.print_memory_usage()

    print("\n" + "=" * 80)
    print("✓ Quick start completed successfully!")
    print("=" * 80)

    return correct


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
