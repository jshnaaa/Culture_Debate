"""
Unified Model Manager for Single-Model Architecture
All agents (Cultural, Conflict Analyzer, Mediator) share the same model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import json
from pathlib import Path


class UnifiedModelManager:
    """
    Manages a single model for all agent tasks
    Supports both Llama3.1-8B and Qwen3-8B
    """

    def __init__(self, model_name: str = "llama", config_path: Optional[str] = None):
        """
        Args:
            model_name: "llama" or "qwen"
            config_path: Path to model_paths.json config file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "model_paths.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Validate model name
        if model_name not in self.config["models"]:
            raise ValueError(f"Invalid model name: {model_name}. "
                           f"Must be one of: {list(self.config['models'].keys())}")

        self.model_name = model_name
        self.model_config = self.config["models"][model_name]
        self.model_path = self.model_config["path"]
        self.device = self.model_config["device"]

        # Model and tokenizer
        self.model = None
        self.tokenizer = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        print("=" * 80)
        print(f"Loading {self.model_config['name']}...")
        print(f"Path: {self.model_path}")
        print(f"Device: {self.device}")
        print("=" * 80)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )

        # Print memory usage
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"\n✓ Model loaded successfully!")
            print(f"✓ GPU memory allocated: {mem_allocated:.2f} GB")
        else:
            print(f"\n✓ Model loaded successfully on CPU!")

        print("=" * 80)

    def generate(self,
                prompt: str,
                max_new_tokens: int = 512,
                temperature: float = 0.0) -> str:
        """
        Generate response for a single prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            Generated text (without prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()

        return generated_text

    def batch_generate(self,
                      prompts: List[str],
                      max_new_tokens: int = 512,
                      temperature: float = 0.0) -> List[str]:
        """
        Generate responses for multiple prompts in batch

        Args:
            prompts: List of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode all outputs
        responses = []
        for i, output in enumerate(outputs):
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove prompt (approximate)
            prompt_len = len(prompts[i])
            generated_text = full_text[prompt_len:].strip()
            responses.append(generated_text)

        return responses

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0}

        return {
            "allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1e9
        }

    def print_memory_usage(self):
        """Print current GPU memory usage"""
        mem = self.get_memory_usage()
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
        print(f"  Reserved: {mem['reserved_gb']:.2f} GB")

    def cleanup(self):
        """Release resources"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("✓ Model unloaded and GPU memory cleared")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama",
                       choices=["llama", "qwen"],
                       help="Model to use (llama or qwen)")
    args = parser.parse_args()

    # Initialize manager
    print(f"\nTesting UnifiedModelManager with model: {args.model}")
    manager = UnifiedModelManager(model_name=args.model)

    # Test single generation
    print("\n" + "=" * 80)
    print("Testing single generation")
    print("=" * 80)

    test_prompt = "What is cultural alignment?"
    response = manager.generate(test_prompt, max_new_tokens=50)
    print(f"\nPrompt: {test_prompt}")
    print(f"Response: {response[:100]}...")

    # Test batch generation
    print("\n" + "=" * 80)
    print("Testing batch generation")
    print("=" * 80)

    test_prompts = [
        "What is cultural alignment?",
        "How do values differ across cultures?"
    ]

    responses = manager.batch_generate(test_prompts, max_new_tokens=50)
    for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {response[:100]}...")

    # Print memory usage
    manager.print_memory_usage()

    print("\n" + "=" * 80)
    print("✓ UnifiedModelManager test completed!")
    print("=" * 80)
