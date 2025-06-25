# Stage 1: Builder - CPU-only for downloading pre-quantized models
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system packages for building (no CUDA needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install minimal requirements for downloading
COPY requirements-builder.txt .
RUN pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements-builder.txt

# Add Hugging Face authentication
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Authenticate with Hugging Face if token provided
RUN if [ -n "$HF_TOKEN" ]; then \
        python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"; \
    fi

# Copy and run model download script (CPU-only operation)
COPY download_model.py .
RUN python3 download_model.py

# Stage 2: Runtime - GPU-enabled for model inference
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install runtime system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy downloaded models from builder stage
COPY --from=builder /models /models

# Install runtime requirements (includes GPU support)
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements-runtime.txt

# Copy application code
COPY handler.py .

# Set environment variables
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (optional, for debugging)
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

CMD ["python3", "-u", "handler.py"]