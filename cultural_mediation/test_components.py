"""
Component Testing Script
Tests each component independently before running the full pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


def test_normad_loader():
    """Test NORMAD dataset loading and parsing"""
    print("\n" + "=" * 80)
    print("TEST 1: NORMAD Loader")
    print("=" * 80)

    try:
        from utils.normad_loader import NORMADLoader, validate_parsing

        # Load dataset
        dataset_path = "/root/autodl-fs/normad_merge_gen.json"
        loader = NORMADLoader(dataset_path)

        print(f"✓ Loaded {len(loader)} samples")

        # Test first sample
        sample = loader.get_sample(0)
        print(f"✓ Sample 0:")
        print(f"  Country: {sample.country}")
        print(f"  Gold Label: {sample.gold_label}")
        print(f"  Story: {sample.story[:100]}...")

        # Validate parsing
        is_valid = validate_parsing(sample)
        print(f"✓ Parsing validation: {'PASS' if is_valid else 'FAIL'}")

        # Get country list
        countries = loader.get_country_list()
        print(f"✓ Number of countries: {len(countries)}")
        print(f"  First 10: {countries[:10]}")

        # Label distribution
        dist = loader.get_label_distribution()
        print(f"✓ Label distribution: {dist}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_learner():
    """Test country weight learner"""
    print("\n" + "=" * 80)
    print("TEST 2: Weight Learner")
    print("=" * 80)

    try:
        from utils.weight_learner import CountryWeightLearner, project_to_simplex
        import numpy as np

        # Test simplex projection
        w = np.array([0.5, 0.3, -0.1, 0.4, 0.2])
        w_proj = project_to_simplex(w)
        print(f"✓ Simplex projection:")
        print(f"  Before: {w}")
        print(f"  After: {w_proj}")
        print(f"  Sum: {w_proj.sum():.6f} (should be 1.0)")
        print(f"  All non-negative: {(w_proj >= 0).all()}")

        # Load weight learner
        weight_path = Path(__file__).parent / "data" / "country_weights_init.json"
        learner = CountryWeightLearner(init_weights_path=weight_path)

        print(f"\n✓ Loaded {learner.num_countries} countries")

        # Test specific country
        egypt_weights = learner.get_country_weight('egypt')
        print(f"✓ Egypt weights: {egypt_weights.tolist()}")
        print(f"  Sum: {egypt_weights.sum().item():.6f}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_paths():
    """Test model paths configuration"""
    print("\n" + "=" * 80)
    print("TEST 3: Model Paths")
    print("=" * 80)

    try:
        import json
        import os

        config_path = Path(__file__).parent / "config" / "model_paths.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"✓ Loaded config from {config_path}")

        # Check Cultural Agent path
        cultural_path = config["cultural_agent"]["path"]
        print(f"\n✓ Cultural Agent path: {cultural_path}")
        if os.path.exists(cultural_path):
            print(f"  ✓ Path exists")
        else:
            print(f"  ✗ Path NOT found!")
            return False

        # Check Qwen path
        qwen_path = config["qwen_unified"]["path"]
        print(f"\n✓ Qwen path: {qwen_path}")
        if os.path.exists(qwen_path):
            print(f"  ✓ Path exists")
        else:
            print(f"  ✗ Path NOT found!")
            return False

        # Check NORMAD dataset
        normad_path = config["_data_paths"]["normad_dataset"]
        print(f"\n✓ NORMAD dataset path: {normad_path}")
        if os.path.exists(normad_path):
            print(f"  ✓ Path exists")
        else:
            print(f"  ✗ Path NOT found!")
            return False

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_model_manager():
    """Test unified model manager (WARNING: Loads model, takes time)"""
    print("\n" + "=" * 80)
    print("TEST 4: Unified Model Manager (Model Loading)")
    print("=" * 80)
    print("⚠️  This test will load model (~30 seconds)")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Skipped.")
        return None

    # Ask which model to test
    print("\nWhich model to test?")
    print("  1. llama (Llama3.1-8B)")
    print("  2. qwen (Qwen3-8B)")
    model_choice = input("Choice (1/2, default=1): ").strip()
    model_name = "qwen" if model_choice == "2" else "llama"

    try:
        from utils.unified_model_manager import UnifiedModelManager

        # Initialize manager (loads model)
        print(f"\nInitializing manager with {model_name}...")
        manager = UnifiedModelManager(model_name=model_name)

        # Test single generation
        print("\n✓ Testing single generation...")
        test_prompt = "What is cultural alignment?"
        response = manager.generate(
            test_prompt,
            max_new_tokens=50,
            temperature=0.0
        )
        print(f"  Response: {response[:100]}...")

        # Test batch generation
        print("\n✓ Testing batch generation...")
        test_prompts = [
            "What is cultural alignment?",
            "How do values differ?"
        ]
        responses = manager.batch_generate(
            test_prompts,
            max_new_tokens=50,
            temperature=0.0
        )
        print(f"  Response 1: {responses[0][:50]}...")
        print(f"  Response 2: {responses[1][:50]}...")

        # Print memory usage
        manager.print_memory_usage()

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_value_extractor():
    """Test value tag extraction (requires model loaded)"""
    print("\n" + "=" * 80)
    print("TEST 5: Value Extractor")
    print("=" * 80)
    print("⚠️  This test requires Qwen model loaded (skipped if not available)")

    try:
        from utils.value_extractor import parse_value_tags, compute_value_distance

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
        print(f"✓ Parsed tags: {tags}")

        # Test distance computation
        tags2 = {
            "autonomy": 0.2,
            "order_security": 0.7,
            "tradition": 0.8,
            "care_universalism": 0.3,
            "achievement_power": 0.5
        }
        distance = compute_value_distance(tags, tags2)
        print(f"✓ Value distance: {distance:.3f}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("Component Testing Suite")
    print("=" * 80)

    results = {}

    # Test 1: NORMAD Loader
    results['normad_loader'] = test_normad_loader()

    # Test 2: Weight Learner
    results['weight_learner'] = test_weight_learner()

    # Test 3: Model Paths
    results['model_paths'] = test_model_paths()

    # Test 4: Unified Model Manager (optional, slow)
    results['unified_model_manager'] = test_unified_model_manager()

    # Test 5: Value Extractor
    results['value_extractor'] = test_value_extractor()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
            symbol = "⊘"
        elif result:
            status = "PASS"
            symbol = "✓"
        else:
            status = "FAIL"
            symbol = "✗"

        print(f"{symbol} {test_name}: {status}")

    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n✓ All tests passed! Ready to run quick_start.py")
    else:
        print("\n✗ Some tests failed. Please fix before proceeding.")


if __name__ == "__main__":
    main()
