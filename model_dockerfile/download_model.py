#!/usr/bin/env python3
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

def main():
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
    cache_dir = "/models"
    
    print(f"Downloading model files: {model_name}")
    
    # Download model files only (no instantiation)
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True
    )
    
    # Download tokenizer files only
    AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    print("✅ Files downloaded successfully!")

if __name__ == "__main__":
    main()