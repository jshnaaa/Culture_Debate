"""
Value Tag Extraction Module
Extracts cultural value emphasis from agent responses using LLM
"""

import json
import re
from typing import Dict, List


VALUE_EXTRACTION_PROMPT = """Analyze the following cultural perspective and rate how much it emphasizes each value dimension on a 0-1 scale (0=not emphasized, 1=strongly emphasized).

Response: {agent_response}

Value Dimensions:
1. Autonomy: self-direction, independence, personal choice
2. Order & Security: stability, safety, rules, protection
3. Tradition: respect for customs, cultural heritage, continuity
4. Care & Universalism: welfare, equality, justice, compassion
5. Achievement & Power: success, influence, status, competence

Output JSON format (no explanation):
{{
  "autonomy": 0.0-1.0,
  "order_security": 0.0-1.0,
  "tradition": 0.0-1.0,
  "care_universalism": 0.0-1.0,
  "achievement_power": 0.0-1.0
}}"""


def parse_value_tags(response_text: str) -> Dict[str, float]:
    """
    Parse LLM response to extract value tags

    Args:
        response_text: LLM-generated response containing JSON

    Returns:
        Dictionary with value dimensions and scores
    """
    # Try to extract JSON
    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if json_match:
        try:
            tags = json.loads(json_match.group(0))
            # Validate keys
            expected_keys = ["autonomy", "order_security", "tradition",
                           "care_universalism", "achievement_power"]
            if all(k in tags for k in expected_keys):
                # Ensure values are in [0, 1]
                tags = {k: max(0.0, min(1.0, float(v))) for k, v in tags.items()}
                return tags
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: uniform distribution
    print(f"Warning: Failed to parse value tags, using uniform distribution")
    return {
        "autonomy": 0.2,
        "order_security": 0.2,
        "tradition": 0.2,
        "care_universalism": 0.2,
        "achievement_power": 0.2
    }


def extract_value_tags(agent_response: str, model_manager) -> Dict[str, float]:
    """
    Extract value tags from an agent response using LLM

    Args:
        agent_response: The cultural agent's response text
        model_manager: ModelManager instance with conflict_analyzer loaded

    Returns:
        Dictionary with value dimensions and scores
    """
    prompt = VALUE_EXTRACTION_PROMPT.format(agent_response=agent_response)

    # Generate extraction
    extraction_text = model_manager.generate(
        prompt,
        max_new_tokens=150,
        temperature=0.0
    )

    # Parse result
    tags = parse_value_tags(extraction_text)

    return tags


def batch_extract_value_tags(agent_responses: List[str], model_manager) -> List[Dict[str, float]]:
    """
    Extract value tags from multiple agent responses in batch

    Args:
        agent_responses: List of agent response texts
        model_manager: ModelManager instance with conflict_analyzer loaded

    Returns:
        List of value tag dictionaries
    """
    prompts = [
        VALUE_EXTRACTION_PROMPT.format(agent_response=resp)
        for resp in agent_responses
    ]

    # Batch generate
    extraction_texts = model_manager.batch_generate(
        prompts,
        max_new_tokens=150,
        temperature=0.0
    )

    # Parse all results
    tags_list = [parse_value_tags(text) for text in extraction_texts]

    return tags_list


def compute_value_distance(tags1: Dict[str, float], tags2: Dict[str, float]) -> float:
    """
    Compute Jaccard-like distance between two value tag sets

    Args:
        tags1, tags2: Value tag dictionaries

    Returns:
        Distance in [0, 1] (0=identical, 1=completely different)
    """
    # Convert to vectors
    dimensions = ["autonomy", "order_security", "tradition",
                 "care_universalism", "achievement_power"]
    v1 = [tags1[d] for d in dimensions]
    v2 = [tags2[d] for d in dimensions]

    # Compute Jaccard distance: 1 - (intersection / union)
    # For continuous values, use min/max as proxy
    intersection = sum(min(v1[i], v2[i]) for i in range(5))
    union = sum(max(v1[i], v2[i]) for i in range(5))

    if union == 0:
        return 0.0

    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


# Example usage
if __name__ == "__main__":
    # Test parsing
    sample_response = """
    {
      "autonomy": 0.8,
      "order_security": 0.3,
      "tradition": 0.2,
      "care_universalism": 0.6,
      "achievement_power": 0.4
    }
    """
    tags = parse_value_tags(sample_response)
    print(f"Parsed tags: {tags}")

    # Test distance
    tags2 = {
        "autonomy": 0.2,
        "order_security": 0.7,
        "tradition": 0.8,
        "care_universalism": 0.3,
        "achievement_power": 0.5
    }
    distance = compute_value_distance(tags, tags2)
    print(f"Value distance: {distance:.3f}")
