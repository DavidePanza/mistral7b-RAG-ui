import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

def find_model_path():
    # First check for quantized model
    quantized_path = "/models/quantized_model"
    if os.path.exists(quantized_path) and os.path.exists(os.path.join(quantized_path, "config.json")):
        return quantized_path
    
    # Fallback to searching for original model
    for root, dirs, files in os.walk("/models"):
        if "config.json" in files:
            return root
    return None

def load_model():
    model_path = find_model_path()
    if not model_path:
        raise RuntimeError("Model not found in /models")
    
    print(f"Loading 4-bit quantized model from: {model_path}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("4-bit quantized model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
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
            "model": "Mistral-7B-4bit-BitsAndBytes",
            "device": str(model.device),
            "memory_usage_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})