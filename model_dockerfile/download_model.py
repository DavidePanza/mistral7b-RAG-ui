#!/usr/bin/env python3
import subprocess
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, show_progress=False):
    """Run a shell command and log output"""
    logger.info(f"Running: {cmd}")
    
    if show_progress:
        # For long-running commands, show real-time output
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                logger.info(f"DOWNLOAD: {line.strip()}")
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with code {process.returncode}: {cmd}")
            
    else:
        # For quick commands, capture all output
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
        
        return result

def main():
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
    model_dir = "/models/Mistral-7B-Instruct-v0.1-GPTQ"
    
    logger.info(f"Downloading pre-quantized model: {model_name}")
    logger.info("Using huggingface-cli for reliable download (CPU-only, no GPU needed)")
    
    # Create models directory
    os.makedirs("/models", exist_ok=True)
    
    # Remove existing directory if it exists
    if os.path.exists(model_dir):
        logger.info(f"Removing existing directory: {model_dir}")
        run_command(f"rm -rf {model_dir}", show_progress=False)
    
    # Create the model directory
    run_command(f"mkdir -p {model_dir}", show_progress=False)
    
    # Download using huggingface-cli (much more reliable than Python API)
    download_cmd = (
        f"huggingface-cli download {model_name} "
        f"--local-dir {model_dir} "
        f"--local-dir-use-symlinks False"
    )
    
    try:
        logger.info("Starting download with huggingface-cli...")
        logger.info("This may take several minutes for a 7B model (~3-4GB)...")
        logger.info("Download progress will be shown below:")
        
        # Use show_progress=True for the download command
        run_command(download_cmd, show_progress=True)
        logger.info("✓ Model download completed successfully!")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Trying fallback download with specific branch...")
        
        # Try with specific branch as fallback
        fallback_cmd = (
            f"huggingface-cli download {model_name} "
            f"--revision gptq-4bit-32g-actorder_True "
            f"--local-dir {model_dir} "
            f"--local-dir-use-symlinks False"
        )
        
        try:
            logger.info("Starting fallback download...")
            run_command(fallback_cmd, show_progress=True)
            logger.info("✓ Fallback download completed successfully!")
        except Exception as e2:
            logger.error(f"Fallback download also failed: {e2}")
            raise RuntimeError("Both primary and fallback downloads failed")
    
    # Verify download
    logger.info("Verifying downloaded files...")
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        logger.info(f"Downloaded files: {files}")
        
        # Check for required files
        required_files = ["config.json", "tokenizer.json", "quantize_config.json"]
        missing_files = []
        
        for file in required_files:
            if file in files:
                logger.info(f"✓ {file} found")
            else:
                missing_files.append(file)
                logger.warning(f"⚠ {file} not found")
        
        # Check for model weight files
        weight_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]
        if weight_files:
            logger.info(f"✓ Model weights found: {weight_files}")
        else:
            logger.warning("⚠ No model weight files found")
        
        # Create marker file with model info
        marker_file = "/models/downloaded_model.txt"
        with open(marker_file, "w") as f:
            f.write(f"{model_name}\n")
            f.write(f"local_path: {model_dir}\n")
            f.write(f"files: {', '.join(files)}\n")
        
        logger.info(f"✓ Model marker created: {marker_file}")
        
        if missing_files:
            logger.warning(f"Some files missing: {missing_files}")
        else:
            logger.info("✓ All required files present!")
            
    else:
        raise RuntimeError(f"Model directory not found: {model_dir}")
    
    logger.info("Pre-quantized model download completed successfully!")
    logger.info("Model ready for GPU loading at runtime")

if __name__ == "__main__":
    main()