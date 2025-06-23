import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MODEL_DIR = "/models/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots"
MODEL_ID = os.listdir(MODEL_DIR)[0] if os.path.exists(MODEL_DIR) else None
MODEL_PATH = f"{MODEL_DIR}/{MODEL_ID}" if MODEL_ID else None

def load_model():
    if not MODEL_PATH:
        raise RuntimeError("Model not found in /models directory")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available - GPU required")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_model()

def handler(job):
    try:
        inputs = job["input"]
        prompt = inputs.get("prompt", "")
        max_tokens = inputs.get("max_tokens", 100)
        
        if not prompt:
            return {"error": "Empty prompt provided"}
        
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return {
            "response": tokenizer.decode(outputs[0], skip_special_tokens=True),
            "model": "Mistral-7B-Instruct-v0.1",
            "device": str(model.device)
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})