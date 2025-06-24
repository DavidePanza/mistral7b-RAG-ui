#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    cache_dir = "/models"
    
    print(f"Downloading and quantizing model: {model_name}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Download and cache tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Download model files (this will cache the original model files)
    print("Downloading model files...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("Model downloaded and quantized successfully!")
    print(f"Model device: {model.device}")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Save the quantized model to disk for faster loading
    print("Saving quantized model...")
    model.save_pretrained(f"{cache_dir}/quantized_model")
    tokenizer.save_pretrained(f"{cache_dir}/quantized_model")
    
    print("Quantized model saved successfully!")

if __name__ == "__main__":
    main()