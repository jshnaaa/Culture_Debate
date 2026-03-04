"""
Model Manager for Sequential Loading
Manages loading/unloading of LLMs to optimize GPU memory usage
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


class ModelManager:
    """
    Manages multiple LLMs with sequential loading to save GPU memory
    """
    def __init__(self,
                 model_paths,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        """
        Args:
            model_paths: dict mapping model names to local paths
                {
                    "cultural_agent": "path/to/llama3.1-8b",
                    "conflict_analyzer": "path/to/qwen2.5-7b",
                    "mediator": "path/to/qwen2.5-14b"
                }
            device: GPU device (e.g., "cuda:0")
            dtype: Model dtype (torch.bfloat16 or torch.float16)
        """
        self.model_paths = model_paths
        self.device = device
        self.dtype = dtype

        self.current_model_name = None
        self.current_model = None
        self.current_tokenizer = None

        # Cache tokenizers (lightweight)
        self.tokenizer_cache = {}
        for name, path in model_paths.items():
            self.tokenizer_cache[name] = AutoTokenizer.from_pretrained(path)

    def load_model(self, model_name):
        """
        Load a specific model, unloading the current one if different

        Args:
            model_name: Name of model to load ("cultural_agent", "conflict_analyzer", "mediator")

        Returns:
            (model, tokenizer) tuple
        """
        # If already loaded, return cached
        if self.current_model_name == model_name:
            return self.current_model, self.current_tokenizer

        # Unload current model
        if self.current_model is not None:
            print(f"Unloading {self.current_model_name}...")
            del self.current_model
            torch.cuda.empty_cache()
            gc.collect()

        # Load new model
        print(f"Loading {model_name}...")
        model_path = self.model_paths[model_name]
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.current_tokenizer = self.tokenizer_cache[model_name]
        self.current_model_name = model_name

        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")

        return self.current_model, self.current_tokenizer

    def generate(self, prompt, max_new_tokens=512, temperature=0.0, **kwargs):
        """
        Generate text using current model

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (without prompt)
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        inputs = self.current_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.current_tokenizer.eos_token_id,
                **kwargs
            )

        # Decode and remove prompt
        full_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()

        return generated_text

    def batch_generate(self, prompts, max_new_tokens=512, temperature=0.0, **kwargs):
        """
        Generate text for multiple prompts in batch

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Tokenize with padding
        inputs = self.current_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Adjust based on model context length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.current_tokenizer.eos_token_id,
                **kwargs
            )

        # Decode all outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            full_text = self.current_tokenizer.decode(output, skip_special_tokens=True)
            # Remove prompt (approximate, since batch padding may vary)
            generated_text = full_text[len(prompts[i]):].strip()
            generated_texts.append(generated_text)

        return generated_texts

    def cleanup(self):
        """Release all resources"""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        self.current_model_name = None


# Example usage
if __name__ == "__main__":
    # Define model paths (update these to your local paths)
    model_paths = {
        "cultural_agent": "/path/to/llama3.1-8b",
        "conflict_analyzer": "/path/to/qwen2.5-7b",
        "mediator": "/path/to/qwen2.5-14b"
    }

    # Initialize manager
    manager = ModelManager(model_paths, device="cuda:0")

    # Load cultural agent
    model, tokenizer = manager.load_model("cultural_agent")

    # Generate response
    prompt = "What is cultural alignment?"
    response = manager.generate(prompt, max_new_tokens=100)
    print(f"Response: {response}")

    # Switch to conflict analyzer
    model, tokenizer = manager.load_model("conflict_analyzer")
    response = manager.generate("Analyze this conflict...", max_new_tokens=50)
    print(f"Conflict analysis: {response}")

    # Cleanup
    manager.cleanup()
