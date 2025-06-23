import runpod
from transformers import AutoTokenizer
from autoawq import AutoAWQForCausalLM
import torch
import os

def find_model_path():
    for root, dirs, files in os.walk("/models"):
        if "config.json" in files:
            return root
    return None

def load_model():
    model_path = find_model_path()
    if not model_path:
        raise RuntimeError("Model not found in /models")
    
    print(f"Loading AWQ 4-bit model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load AWQ quantized model
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        fuse_layers=True,
        trust_remote_code=True,
        safetensors=True
    )
    
    print("✅ AWQ 4-bit model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    return model, tokenizer

# Load model at startup
model, tokenizer = load_model()

def handler(job):
    try:
        inputs = job["input"]
        prompt = inputs.get("prompt", "")
        max_tokens = inputs.get("max_tokens", 100)
        temperature = inputs.get("temperature", 0.7)
        
        if not prompt:
            return {"error": "Empty prompt provided"}
        
        # Format prompt for Mistral Instruct
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize input
        input_ids = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][input_ids['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return {
            "response": response.strip(),
            "model": "Mistral-7B-AWQ-4bit",
            "device": str(model.device),
            "memory_usage_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})