#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import os

def main():
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
    cache_dir = "/models"
    
    print(f"Downloading model: {model_name}")
    
    # Download all model files (including AWQ specific files)
    model_files = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=["*.json", "*.model", "*.safetensors", "*.py"],
        ignore_patterns=["*.bin", "*.h5"]  # Exclude unnecessary formats
    )
    
    # Verify critical files exist
    required_files = {
        "config.json",
        "model.safetensors",
        "quantization_config.json"
    }
    
    model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}", "snapshots")
    snapshot_id = next(os.listdir(model_path))
    full_path = os.path.join(model_path, snapshot_id)
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(full_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    # Download tokenizer (with same cache directory)
    print("Downloading tokenizer...")
    AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    print(f"Model and tokenizer downloaded successfully to: {full_path}")

if __name__ == "__main__":
    main()