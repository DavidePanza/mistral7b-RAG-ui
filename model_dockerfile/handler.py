import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_model_path():
    """Find the downloaded model path"""
    # First check for the direct download path
    direct_path = "/models/Mistral-7B-Instruct-v0.1-GPTQ"
    if os.path.exists(direct_path) and os.path.exists(os.path.join(direct_path, "config.json")):
        logger.info(f"Found model at direct path: {direct_path}")
        return direct_path
    
    # Check marker file
    marker_file = "/models/downloaded_model.txt"
    if os.path.exists(marker_file):
        with open(marker_file, "r") as f:
            content = f.read().strip()
            for line in content.split('\n'):
                if line.startswith('local_path:'):
                    path = line.split('local_path:')[1].strip()
                    if os.path.exists(path):
                        logger.info(f"Found model from marker: {path}")
                        return path
    
    # Fallback: search for any config.json
    for root, dirs, files in os.walk("/models"):
        if "config.json" in files and "quantize_config.json" in files:
            logger.info(f"Found quantized model at: {root}")
            return root
    
    # If nothing found locally, use the model name (will download at runtime)
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
    logger.info(f"No local model found, will use: {model_name}")
    return model_name

def load_model():
    """Load the pre-quantized GPTQ model"""
    model_path = find_model_path()
    
    logger.info(f"Loading pre-quantized GPTQ model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,  # As per TheBloke's docs
        use_fast=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the pre-quantized model
    # Note: No quantization_config needed - it's already quantized
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=False,  # As per TheBloke's docs
        revision="main"  # Use main branch
    )
    
    logger.info("Pre-quantized GPTQ model loaded successfully!")
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model dtype: {model.dtype}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return model, tokenizer

# Load model at startup
logger.info("Loading model at startup...")
model, tokenizer = load_model()
logger.info("Model loaded and ready for inference!")

def handler(job):
    """Handle inference requests"""
    try:
        inputs = job["input"]
        prompt = inputs.get("prompt", "")
        max_tokens = inputs.get("max_tokens", 100)
        temperature = inputs.get("temperature", 0.7)
        top_p = inputs.get("top_p", 0.95)
        top_k = inputs.get("top_k", 40)
        
        if not prompt:
            return {"error": "Empty prompt provided"}
        
        logger.info(f"Processing prompt (length: {len(prompt)} chars)")
        
        # Format prompt according to Mistral's instruction format
        # As shown in TheBloke's documentation
        prompt_template = f"<s>[INST] {prompt} [/INST]\n"
        
        # Tokenize input
        input_ids = tokenizer(
            prompt_template, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).input_ids.to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (excluding input)
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        logger.info("Response generated successfully")
        
        return {
            "response": response.strip(),
            "model": "Mistral-7B-Instruct-GPTQ-4bit",
            "device": str(model.device),
            "memory_usage_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "input_tokens": input_ids.shape[1],
            "output_tokens": outputs[0].shape[0] - input_ids.shape[1],
            "prompt_template_used": True
        }
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})