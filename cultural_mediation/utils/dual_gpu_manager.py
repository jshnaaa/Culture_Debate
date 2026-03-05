"""
Dual-GPU Model Manager for Pipeline Parallelism
Manages Llama3.1-8B on GPU0 and Qwen2.5-14B on GPU1
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import json
from pathlib import Path


class DualGPUModelManager:
    """
    Manages two models on separate GPUs for pipeline parallelism:
    - GPU0 (cuda:0): Llama3.1-8B for Cultural Agents
    - GPU1 (cuda:1): Qwen2.5-14B for Conflict Analysis and Mediation
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to model_paths.json config file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "model_paths.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Model paths and devices
        self.cultural_agent_path = self.config["cultural_agent"]["path"]
        self.qwen_path = self.config["qwen_unified"]["path"]
        self.device_cultural = self.config["cultural_agent"]["device"]
        self.device_qwen = self.config["qwen_unified"]["device"]

        # Models and tokenizers
        self.cultural_model = None
        self.cultural_tokenizer = None
        self.qwen_model = None
        self.qwen_tokenizer = None

        # Load both models
        self._load_models()

    def _load_models(self):
        """Load both models onto their respective GPUs"""
        print("=" * 80)
        print("Loading models for dual-GPU pipeline parallelism...")
        print("=" * 80)

        # Load Llama3.1-8B on GPU0
        print(f"\n[GPU0] Loading Cultural Agent: {self.cultural_agent_path}")
        self.cultural_tokenizer = AutoTokenizer.from_pretrained(
            self.cultural_agent_path,
            trust_remote_code=True
        )
        self.cultural_model = AutoModelForCausalLM.from_pretrained(
            self.cultural_agent_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_cultural,
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            mem_gpu0 = torch.cuda.memory_allocated(0) / 1e9
            print(f"✓ GPU0 memory allocated: {mem_gpu0:.2f} GB")

        # Load Qwen2.5-14B on GPU1
        print(f"\n[GPU1] Loading Qwen (Conflict Analyzer + Mediator): {self.qwen_path}")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_path,
            trust_remote_code=True
        )
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            self.qwen_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_qwen,
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            mem_gpu1 = torch.cuda.memory_allocated(1) / 1e9
            print(f"✓ GPU1 memory allocated: {mem_gpu1:.2f} GB")

        print("\n" + "=" * 80)
        print("✓ Both models loaded successfully!")
        print("=" * 80)

    def generate_cultural_responses(self,
                                   prompts: List[str],
                                   max_new_tokens: int = 256,
                                   temperature: float = 0.0) -> List[str]:
        """
        Generate responses from Cultural Agent (Llama3.1-8B on GPU0)

        Args:
            prompts: List of prompts (one per agent)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            List of generated responses
        """
        # Tokenize with padding
        inputs = self.cultural_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device_cultural)

        # Generate
        with torch.no_grad():
            outputs = self.cultural_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.cultural_tokenizer.eos_token_id
            )

        # Decode
        responses = []
        for i, output in enumerate(outputs):
            full_text = self.cultural_tokenizer.decode(output, skip_special_tokens=True)
            # Remove prompt (approximate)
            prompt_len = len(prompts[i])
            generated_text = full_text[prompt_len:].strip()
            responses.append(generated_text)

        return responses

    def generate_with_qwen(self,
                          prompt: str,
                          max_new_tokens: int = 512,
                          temperature: float = 0.0) -> str:
        """
        Generate response using Qwen2.5-14B on GPU1
        Used for value extraction, conflict analysis, and mediation

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        inputs = self.qwen_tokenizer(prompt, return_tensors="pt").to(self.device_qwen)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.qwen_tokenizer.eos_token_id
            )

        full_text = self.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()

        return generated_text

    def batch_generate_with_qwen(self,
                                 prompts: List[str],
                                 max_new_tokens: int = 512,
                                 temperature: float = 0.0) -> List[str]:
        """
        Batch generate responses using Qwen2.5-14B on GPU1

        Args:
            prompts: List of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        inputs = self.qwen_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device_qwen)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.qwen_tokenizer.eos_token_id
            )

        responses = []
        for i, output in enumerate(outputs):
            full_text = self.qwen_tokenizer.decode(output, skip_special_tokens=True)
            prompt_len = len(prompts[i])
            generated_text = full_text[prompt_len:].strip()
            responses.append(generated_text)

        return responses

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"gpu0": 0.0, "gpu1": 0.0}

        return {
            "gpu0_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "gpu0_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "gpu1_allocated_gb": torch.cuda.memory_allocated(1) / 1e9,
            "gpu1_reserved_gb": torch.cuda.memory_reserved(1) / 1e9
        }

    def print_memory_usage(self):
        """Print current GPU memory usage"""
        mem = self.get_memory_usage()
        print("\nGPU Memory Usage:")
        print(f"  GPU0: {mem['gpu0_allocated_gb']:.2f} GB allocated, "
              f"{mem['gpu0_reserved_gb']:.2f} GB reserved")
        print(f"  GPU1: {mem['gpu1_allocated_gb']:.2f} GB allocated, "
              f"{mem['gpu1_reserved_gb']:.2f} GB reserved")

    def cleanup(self):
        """Release all resources (optional, usually not needed)"""
        if self.cultural_model is not None:
            del self.cultural_model
            del self.cultural_tokenizer
        if self.qwen_model is not None:
            del self.qwen_model
            del self.qwen_tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ Models unloaded and GPU memory cleared")


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = DualGPUModelManager()

    # Test cultural agent generation
    print("\n" + "=" * 80)
    print("Testing Cultural Agent (GPU0)")
    print("=" * 80)

    test_prompts = [
        "What is cultural alignment?",
        "How do values differ across cultures?"
    ]

    responses = manager.generate_cultural_responses(
        test_prompts,
        max_new_tokens=50,
        temperature=0.0
    )

    for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {response[:100]}...")

    # Test Qwen generation
    print("\n" + "=" * 80)
    print("Testing Qwen (GPU1)")
    print("=" * 80)

    qwen_prompt = "Analyze the following conflict: Agent A says yes, Agent B says no."
    qwen_response = manager.generate_with_qwen(qwen_prompt, max_new_tokens=50)

    print(f"\nPrompt: {qwen_prompt}")
    print(f"Response: {qwen_response[:100]}...")

    # Print memory usage
    manager.print_memory_usage()

    print("\n" + "=" * 80)
    print("✓ Dual-GPU manager test completed!")
    print("=" * 80)
