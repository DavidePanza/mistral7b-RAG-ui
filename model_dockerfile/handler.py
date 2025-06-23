import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Find the correct model path in cache
def find_model_path():
    base_cache = "/models"
    
    # Look for the AWQ model directory
    for root, dirs, files in os.walk(base_cache):
        if "TheBloke" in root and "Mistral-7B-Instruct-v0.1-AWQ" in root:
            # Find the snapshot directory
            if "snapshots" in root and os.listdir(root):
                snapshot_dir = os.path.join(root, os.listdir(root)[0])
                if os.path.exists(os.path.join(snapshot_dir, "config.json")):
                    return snapshot_dir
    
    # Fallback: look for any model with config.json
    for root, dirs, files in os.walk(base_cache):
        if "config.json" in files:
            return root
    
    return None

MODEL_PATH = find_model_path()

def load_model():
    if not MODEL_PATH:
        available_paths = []
        for root, dirs, files in os.walk("/models"):
            if files:
                available_paths.append(root)
        raise RuntimeError(f"Model not found. Available paths: {available_paths}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available - GPU required")

    print(f"Loading model from: {MODEL_PATH}")
    
    # Load AWQ model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_safetensors=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

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
        
        input_ids = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
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
            "model": "Mistral-7B-Instruct-v0.1-AWQ",
            "device": str(model.device),
            "model_path": MODEL_PATH
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})