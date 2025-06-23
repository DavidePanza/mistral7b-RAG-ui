#!/usr/bin/env python3
from huggingface_hub import snapshot_download

def main():
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
    cache_dir = "/models"
    
    print(f"Downloading AWQ model files: {model_name}")
    
    # Download only the model files (no model instantiation)
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True
    )
    
    print("AWQ model files downloaded successfully!")

if __name__ == "__main__":
    main()