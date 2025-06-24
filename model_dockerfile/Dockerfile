# Stage 1: Builder - AMD64-optimized  
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install system packages for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install builder requirements
COPY requirements-builder.txt .
RUN pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements-builder.txt

# Copy and run model download script
COPY download_model.py .
RUN python3 download_model.py

# Stage 2: Runtime - Minimal image
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install runtime system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy models and Python packages from builder
COPY --from=builder /models /models
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime requirements
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy application code
COPY handler.py .

# Set Python path to ensure packages are found
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH

CMD ["python3", "-u", "handler.py"]