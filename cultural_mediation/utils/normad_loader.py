"""
NORMAD Dataset Loader and Parser
Handles loading and parsing of NORMAD dataset with instruction-based format
"""

import json
import re
from typing import Dict, List, Optional
from pathlib import Path


class NORMADSample:
    """Represents a single NORMAD sample with parsed fields"""

    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.instruction = raw_data.get("instruction", "")
        self.output = raw_data.get("output", "")
        self.country = raw_data.get("country", "").lower()

        # Parse instruction to extract components
        self.background = self._extract_background()
        self.rule_of_thumb = self._extract_rule_of_thumb()
        self.story = self._extract_story()

        # Convert output to label
        self.gold_label = self._convert_output_to_label()

    def _extract_background(self) -> str:
        """Extract Background section from instruction"""
        match = re.search(r'Background:\s*(.*?)\s*Rule-of-Thumb:',
                         self.instruction, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_rule_of_thumb(self) -> str:
        """Extract Rule-of-Thumb section from instruction"""
        match = re.search(r'Rule-of-Thumb:\s*(.*?)\s*Story:',
                         self.instruction, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_story(self) -> str:
        """Extract Story section from instruction"""
        match = re.search(r'Story:\s*(.*?)\s*Is what',
                         self.instruction, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _convert_output_to_label(self) -> str:
        """Convert output (1/2/3) to label (yes/no/neutral)"""
        mapping = {
            "1": "yes",
            "2": "no",
            "3": "neutral"
        }
        return mapping.get(self.output, "neutral")

    def to_dict(self) -> Dict:
        """Convert to dictionary format for processing"""
        return {
            "country": self.country,
            "background": self.background,
            "rule_of_thumb": self.rule_of_thumb,
            "story": self.story,
            "gold_label": self.gold_label,
            "raw_instruction": self.instruction,
            "raw_output": self.output
        }

    def __repr__(self):
        return (f"NORMADSample(country={self.country}, "
                f"gold_label={self.gold_label}, "
                f"story_preview={self.story[:50]}...)")


class NORMADLoader:
    """Loader for NORMAD dataset"""

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to NORMAD JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.samples: List[NORMADSample] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from JSON file"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Expect JSON array format
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected JSON array, got {type(raw_data)}")

        # Parse all samples
        for item in raw_data:
            try:
                sample = NORMADSample(item)
                self.samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to parse sample: {e}")
                continue

        print(f"✓ Loaded {len(self.samples)} samples from {self.dataset_path.name}")

    def get_sample(self, index: int) -> NORMADSample:
        """Get a single sample by index"""
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range [0, {len(self.samples)})")
        return self.samples[index]

    def get_samples_by_country(self, country: str) -> List[NORMADSample]:
        """Get all samples for a specific country"""
        country = country.lower()
        return [s for s in self.samples if s.country == country]

    def get_country_list(self) -> List[str]:
        """Get list of all unique countries in dataset"""
        return sorted(list(set(s.country for s in self.samples)))

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels"""
        dist = {"yes": 0, "no": 0, "neutral": 0}
        for sample in self.samples:
            dist[sample.gold_label] += 1
        return dist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.get_sample(index)

    def __iter__(self):
        return iter(self.samples)


def validate_parsing(sample: NORMADSample) -> bool:
    """
    Validate that a sample was parsed correctly

    Returns:
        True if all required fields are non-empty
    """
    required_fields = [
        sample.background,
        sample.rule_of_thumb,
        sample.story,
        sample.country,
        sample.gold_label
    ]

    if not all(required_fields):
        print(f"⚠️  Parsing validation failed for sample:")
        print(f"   Country: {sample.country}")
        print(f"   Background: {'✓' if sample.background else '✗'}")
        print(f"   Rule-of-Thumb: {'✓' if sample.rule_of_thumb else '✗'}")
        print(f"   Story: {'✓' if sample.story else '✗'}")
        print(f"   Gold Label: {'✓' if sample.gold_label else '✗'}")
        return False

    return True


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "instruction": """### Question: Give me the answer from 1 to 3: Background: Basic Etiquette- It is considered impolite to point the toe, heel or any part of the foot toward another person. Showing the sole of one's shoe is also impolite.- Modest dress and presentation is highly valued in Egyptian culture.- Greetings often occur before any form of social interaction. For example, a person joining a group is expected to greet all those present.- Generally, the younger defer to the older through showing respect, not challenging their seniors and using special verbal terms of address for aunts, uncles, grandparents and older non-relatives.- If your counterpart identifies as Muslim, it is forbidden to walk in front of someone who is praying or to talk to someone who is currently in prayer.- People are expected to show gratitude when offered a compliment. This is done by responding with an equally respectful compliment on the same subject or, if they are Muslim, wishing Allah's (God's) blessings. Rule-of-Thumb: It is respectful to greet everyone present before starting any social interaction. Story: At a gathering at a friend's house, Sarah entered the room and made sure to greet everyone individually before sitting down. Is what Sarah did socially acceptable? 1. yes 2. no 3. neutral. This question is for a country or language that is egypt. You can only choose one option. ### Answer: """,
        "input": "",
        "output": "1",
        "country": "egypt"
    }

    # Test parsing
    print("=" * 80)
    print("Testing NORMAD Sample Parsing")
    print("=" * 80)

    sample = NORMADSample(sample_data)

    print(f"\n✓ Country: {sample.country}")
    print(f"✓ Gold Label: {sample.gold_label} (from output: {sample.output})")
    print(f"\n✓ Background (first 100 chars):")
    print(f"  {sample.background[:100]}...")
    print(f"\n✓ Rule-of-Thumb:")
    print(f"  {sample.rule_of_thumb}")
    print(f"\n✓ Story:")
    print(f"  {sample.story}")

    # Validate
    print(f"\n✓ Validation: {'PASS' if validate_parsing(sample) else 'FAIL'}")

    # Test conversion to dict
    print(f"\n✓ Dictionary format:")
    sample_dict = sample.to_dict()
    for key, value in sample_dict.items():
        if key not in ["raw_instruction"]:  # Skip long field
            print(f"  {key}: {value[:50] if isinstance(value, str) else value}...")

    print("\n" + "=" * 80)
    print("Parsing test completed!")
    print("=" * 80)

    # Note: To test full loader, uncomment below (requires actual dataset file)
    # loader = NORMADLoader("/root/autodl-fs/normad_merge_gen.json")
    # print(f"\nLoaded {len(loader)} samples")
    # print(f"Countries: {loader.get_country_list()[:10]}...")
    # print(f"Label distribution: {loader.get_label_distribution()}")
