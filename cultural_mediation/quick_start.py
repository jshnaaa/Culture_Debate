"""
Quick Start Example: Cultural Conflict Mediation System
Demonstrates the complete pipeline on a single NORMAD sample
"""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.model_manager import ModelManager
from utils.weight_learner import CountryWeightLearner
from utils.value_extractor import extract_value_tags, batch_extract_value_tags


def main():
    print("=" * 80)
    print("Cultural Conflict Mediation System - Quick Start")
    print("=" * 80)

    # ============================================================================
    # Step 1: Configuration
    # ============================================================================
    print("\n[Step 1] Loading configuration...")

    # Model paths (UPDATE THESE TO YOUR LOCAL PATHS)
    model_paths = {
        "cultural_agent": "/path/to/llama3.1-8b",  # TODO: Update
        "conflict_analyzer": "/path/to/qwen2.5-7b",  # TODO: Update
        "mediator": "/path/to/qwen2.5-14b"  # TODO: Update
    }

    # Check if paths are configured
    if "/path/to/" in str(model_paths):
        print("\n⚠️  WARNING: Please update model paths in quick_start.py")
        print("Edit the 'model_paths' dictionary with your local model paths")
        return

    device = "cuda:0"
    print(f"✓ Device: {device}")
    print(f"✓ Models configured")

    # ============================================================================
    # Step 2: Initialize Components
    # ============================================================================
    print("\n[Step 2] Initializing components...")

    # Model manager
    model_mgr = ModelManager(model_paths, device=device)
    print("✓ Model manager initialized")

    # Weight learner
    weight_path = Path(__file__).parent / "data" / "country_weights_init.json"
    weight_learner = CountryWeightLearner(init_weights_path=weight_path)
    print(f"✓ Loaded weights for {weight_learner.num_countries} countries")

    # ============================================================================
    # Step 3: Load Sample
    # ============================================================================
    print("\n[Step 3] Loading NORMAD sample...")

    # Example sample (replace with actual NORMAD data)
    sample = {
        "ID": 0,
        "Country": "egypt",
        "Background": """### Basic Etiquette
- Egyptians are generally very warm, sociable and open when among friends and family.
- Greetings are usually prolonged as each person enquires about the other's health, children, family, etc.
- It is considered rude to rush off after a greeting.
- Hospitality is highly valued in Egyptian culture.""",
        "Rule-of-Thumb": "It is respectful to greet everyone present when entering a room and spend time in conversation before discussing business matters.",
        "Story": "At a business meeting in Cairo, Ahmed arrives and immediately sits down to discuss the contract terms without greeting the other attendees individually.",
        "Gold Label": "no"
    }

    print(f"✓ Country: {sample['Country']}")
    print(f"✓ Scenario: {sample['Story'][:100]}...")
    print(f"✓ Gold Label: {sample['Gold Label']}")

    # ============================================================================
    # Step 4: Generate Cultural Agent Responses
    # ============================================================================
    print("\n[Step 4] Generating cultural agent responses...")

    # Get country weights
    country_weights = weight_learner.get_country_weight(sample['Country'])
    print(f"✓ Country weights: {country_weights.tolist()}")

    # Load cultural agent model
    model_mgr.load_model("cultural_agent")

    # Dimension names
    dimensions = ["Autonomy", "Order & Security", "Tradition",
                 "Care & Universalism", "Achievement & Power"]

    # Generate 5 agent responses
    agent_responses = []
    for i, dimension in enumerate(dimensions):
        weight = country_weights[i].item()

        # Build prompt
        prompt = f"""You are a cultural perspective agent with the following value priorities:
- Autonomy (self-direction, independence): {country_weights[0]:.2f}
- Order & Security (stability, safety): {country_weights[1]:.2f}
- Tradition (respect for customs): {country_weights[2]:.2f}
- Care & Universalism (welfare, equality): {country_weights[3]:.2f}
- Achievement & Power (success, influence): {country_weights[4]:.2f}

Your PRIMARY focus is on: {dimension} (weight: {weight:.2f})

Cultural Context:
{sample['Background']}

Rule-of-Thumb:
{sample['Rule-of-Thumb']}

Scenario:
{sample['Story']}

Based on your value priorities (especially {dimension}), is the action appropriate?
Answer: [Yes/No/Neither]
Explanation (≤3 sentences, focusing on {dimension}):"""

        # Generate response
        response = model_mgr.generate(prompt, max_new_tokens=200, temperature=0.0)

        # Parse answer
        if "Yes" in response[:50]:
            answer = "Yes"
        elif "No" in response[:50]:
            answer = "No"
        else:
            answer = "Neither"

        agent_responses.append({
            "dimension": dimension,
            "weight": weight,
            "answer": answer,
            "explanation": response
        })

        print(f"  Agent {i+1} ({dimension}, w={weight:.2f}): {answer}")

    # ============================================================================
    # Step 5: Extract Value Tags
    # ============================================================================
    print("\n[Step 5] Extracting value tags...")

    model_mgr.load_model("conflict_analyzer")

    explanations = [resp["explanation"] for resp in agent_responses]
    value_tags_list = batch_extract_value_tags(explanations, model_mgr)

    for i, tags in enumerate(value_tags_list):
        print(f"  Agent {i+1} value emphasis: {tags}")

    # ============================================================================
    # Step 6: Analyze Conflict
    # ============================================================================
    print("\n[Step 6] Analyzing conflict...")

    # Compute pairwise conflicts (simplified)
    from utils.value_extractor import compute_value_distance
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = embed_model.encode(explanations)

    # Compute conflict scores
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

            # Combined conflict
            conflict_ij = 0.4 * sem_dist + 0.4 * val_dist + 0.2 * pol_dist
            conflicts.append(conflict_ij)

    overall_conflict = np.mean(conflicts)
    print(f"✓ Overall conflict score: {overall_conflict:.3f}")

    # ============================================================================
    # Step 7: Mediation (if needed)
    # ============================================================================
    threshold = 0.6
    if overall_conflict > threshold:
        print(f"\n[Step 7] Conflict exceeds threshold ({threshold:.2f}), initiating mediation...")

        model_mgr.load_model("mediator")

        # Build mediation prompt
        responses_text = "\n".join([
            f"Agent {i+1} ({resp['dimension']}): {resp['answer']} - {resp['explanation'][:100]}..."
            for i, resp in enumerate(agent_responses)
        ])

        mediation_prompt = f"""You are a neutral mediator facilitating cultural dialogue.

Current Situation:
{responses_text}

Conflict Analysis:
- Overall conflict score: {overall_conflict:.2f}
- Main disagreements: Different emphasis on tradition vs. individual autonomy

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
..."""

        mediation = model_mgr.generate(mediation_prompt, max_new_tokens=500, temperature=0.0)
        print(f"✓ Mediation generated")
        print(f"\nMediation Output:\n{mediation}")

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
    print(f"✓ Vote distribution: {votes}")
    print(f"✓ Final answer: {final_answer}")
    print(f"✓ Gold label: {sample['Gold Label']}")
    print(f"✓ Correct: {final_answer.lower() == sample['Gold Label'].lower()}")

    # ============================================================================
    # Cleanup
    # ============================================================================
    print("\n[Cleanup] Releasing resources...")
    model_mgr.cleanup()
    print("✓ Done")

    print("\n" + "=" * 80)
    print("Quick start completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import numpy as np  # Import here to avoid top-level import issues
    main()
