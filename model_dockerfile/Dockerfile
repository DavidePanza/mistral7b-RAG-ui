# Stage 1: Builder
FROM nvidia/cuda:12.1-base-ubuntu22.04 as builder

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-builder.txt .
RUN pip install --no-cache-dir -r requirements-builder.txt

# Download model
RUN python3 -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained( \
    'mistralai/Mistral-7B-Instruct-v0.1', \
    cache_dir='/models', \
    load_in_4bit=True, \
    device_map='auto')"

# Stage 2: Runtime
FROM nvidia/cuda:12.1-base-ubuntu22.04

WORKDIR /app

# Install Python runtime only
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 && \
    rm -rf /var/lib/apt/lists/*

# Copy artifacts
COPY --from=builder /models /models
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install runtime deps
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]