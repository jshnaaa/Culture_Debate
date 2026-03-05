"""
Quick Start Example: Cultural Conflict Mediation System (Dual-GPU Version)
Demonstrates the complete pipeline on a single NORMAD sample
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.dual_gpu_manager import DualGPUModelManager
from utils.weight_learner import CountryWeightLearner
from utils.value_extractor import batch_extract_value_tags, compute_value_distance
from utils.normad_loader import NORMADLoader
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 80)
    print("Cultural Conflict Mediation System - Quick Start (Dual-GPU)")
    print("=" * 80)

    # ============================================================================
    # Step 1: Load Configuration and Models
    # ============================================================================
    print("\n[Step 1] Initializing dual-GPU model manager...")

    # Initialize dual-GPU manager (loads both models)
    model_mgr = DualGPUModelManager()

    # Load weight learner
    weight_path = Path(__file__).parent / "data" / "country_weights_init.json"
    weight_learner = CountryWeightLearner(init_weights_path=weight_path)
    print(f"✓ Loaded weights for {weight_learner.num_countries} countries")

    # ============================================================================
    # Step 2: Load NORMAD Sample
    # ============================================================================
    print("\n[Step 2] Loading NORMAD dataset...")

    # Load dataset
    dataset_path = "/root/autodl-fs/normad_merge_gen.json"
    loader = NORMADLoader(dataset_path)

    # Get first sample for testing
    sample = loader.get_sample(0)
    sample_dict = sample.to_dict()

    print(f"✓ Loaded sample from country: {sample.country}")
    print(f"✓ Gold label: {sample.gold_label} (output: {sample.output})")
    print(f"✓ Story preview: {sample.story[:100]}...")

    # ============================================================================
    # Step 3: Get Country Weights
    # ============================================================================
    print("\n[Step 3] Getting country weights...")

    country_weights = weight_learner.get_country_weight(sample.country)
    print(f"✓ Country: {sample.country}")
    print(f"✓ Weights: {country_weights.tolist()}")
    print(f"  [Autonomy, Order&Security, Tradition, Care&Universalism, Achievement&Power]")

    # ============================================================================
    # Step 4: Generate Cultural Agent Responses (GPU0)
    # ============================================================================
    print("\n[Step 4] Generating cultural agent responses (GPU0: Llama3.1-8B)...")

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

Based on your value priorities (especially {dimension}), is the action appropriate?
Answer: [Yes/No/Neither]
Explanation (≤3 sentences, focusing on {dimension}):"""

        prompts.append(prompt)

    # Generate all 5 responses in batch
    print("  Generating batch of 5 agent responses...")
    agent_responses_raw = model_mgr.generate_cultural_responses(
        prompts,
        max_new_tokens=200,
        temperature=0.0
    )

    # Parse answers
    agent_responses = []
    for i, (dimension, response_raw) in enumerate(zip(dimensions, agent_responses_raw)):
        # Parse answer from response
        if "Yes" in response_raw[:100]:
            answer = "Yes"
        elif "No" in response_raw[:100]:
            answer = "No"
        else:
            answer = "Neither"

        agent_responses.append({
            "dimension": dimension,
            "weight": country_weights[i].item(),
            "answer": answer,
            "explanation": response_raw
        })

        print(f"  Agent {i+1} ({dimension}, w={country_weights[i].item():.2f}): {answer}")

    # ============================================================================
    # Step 5: Extract Value Tags (GPU1)
    # ============================================================================
    print("\n[Step 5] Extracting value tags (GPU1: Qwen2.5-14B)...")

    explanations = [resp["explanation"] for resp in agent_responses]
    value_tags_list = batch_extract_value_tags(explanations, model_mgr)

    for i, tags in enumerate(value_tags_list):
        print(f"  Agent {i+1} value emphasis:")
        for dim, score in tags.items():
            print(f"    {dim}: {score:.2f}")

    # ============================================================================
    # Step 6: Analyze Conflict (GPU1)
    # ============================================================================
    print("\n[Step 6] Analyzing conflict (GPU1: Qwen2.5-14B)...")

    # Load embedding model
    print("  Loading sentence transformer for embeddings...")
    embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = embed_model.encode(explanations)

    # Compute pairwise conflicts
    conflicts = []
    for i in range(5):
        for j in range(i+1, 5):
            # Semantic distance
            sem_dist = 1 - (embeddings[i] @ embeddings[j]) / \
                      (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

            # Value distance
            val_dist = compute_value_distance(value_tags_list[i], value_tags_list[j])

            # Polarity distance
            polarity_map = {"Yes": 1, "No": -1, "Neither": 0}
            pol_i = polarity_map[agent_responses[i]["answer"]]
            pol_j = polarity_map[agent_responses[j]["answer"]]
            pol_dist = abs(pol_i - pol_j) / 2

            # Combined conflict (α=0.4, β=0.4, γ=0.2)
            conflict_ij = 0.4 * sem_dist + 0.4 * val_dist + 0.2 * pol_dist
            conflicts.append(conflict_ij)

    overall_conflict = np.mean(conflicts)
    print(f"✓ Overall conflict score: {overall_conflict:.3f}")
    print(f"  (Threshold for mediation: 0.6)")

    # ============================================================================
    # Step 7: Mediation (if needed) (GPU1)
    # ============================================================================
    threshold = 0.6
    if overall_conflict > threshold:
        print(f"\n[Step 7] Conflict exceeds threshold, initiating mediation (GPU1: Qwen2.5-14B)...")

        # Build mediation prompt
        responses_text = "\n".join([
            f"Agent {i+1} ({resp['dimension']}, weight={resp['weight']:.2f}): "
            f"{resp['answer']} - {resp['explanation'][:150]}..."
            for i, resp in enumerate(agent_responses)
        ])

        mediation_prompt = f"""You are a neutral mediator facilitating cultural dialogue.

Current Situation:
{responses_text}

Conflict Analysis:
- Overall conflict score: {overall_conflict:.2f}
- Main disagreements: Different emphasis on cultural values

Your Task:
1. Identify common ground across all perspectives
2. Propose a balanced resolution respecting all value priorities
3. Suggest specific adjustments for each agent (≤2 sentences each)

Constraints:
- Do NOT favor any single cultural perspective
- Preserve diversity while reducing unnecessary conflict
- Focus on bridging core value differences

Output Format:
Common Ground: ...
Proposed Resolution: ...
Agent Adjustments:
- Agent 1: ...
- Agent 2: ...
- Agent 3: ...
- Agent 4: ...
- Agent 5: ..."""

        print("  Generating mediation...")
        mediation = model_mgr.generate_with_qwen(
            mediation_prompt,
            max_new_tokens=500,
            temperature=0.0
        )

        print(f"✓ Mediation generated")
        print(f"\nMediation Output:")
        print("-" * 80)
        print(mediation[:500] + "..." if len(mediation) > 500 else mediation)
        print("-" * 80)

    else:
        print(f"\n[Step 7] Conflict below threshold ({threshold:.2f}), no mediation needed")

    # ============================================================================
    # Step 8: Final Decision
    # ============================================================================
    print("\n[Step 8] Computing final decision...")

    # Weighted voting
    votes = {"Yes": 0.0, "No": 0.0, "Neither": 0.0}
    for i, resp in enumerate(agent_responses):
        votes[resp["answer"]] += country_weights[i].item()

    final_answer = max(votes, key=votes.get)

    print(f"✓ Vote distribution:")
    for answer, weight in votes.items():
        print(f"  {answer}: {weight:.3f}")

    print(f"\n✓ Final answer: {final_answer}")
    print(f"✓ Gold label: {sample.gold_label}")

    # Check correctness
    correct = final_answer.lower() == sample.gold_label.lower()
    print(f"✓ Prediction: {'✓ CORRECT' if correct else '✗ INCORRECT'}")

    # ============================================================================
    # Step 9: Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Country: {sample.country}")
    print(f"Conflict Score: {overall_conflict:.3f}")
    print(f"Mediation: {'Yes' if overall_conflict > threshold else 'No'}")
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


if __name__ == "__main__":
    main()
