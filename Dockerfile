# Use CUDA 12.1 to match llama-cpp-python wheel
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    wget \
    git \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip

# Download the model
RUN wget -O model.gguf \
    "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Copy application code
COPY app.py .

# No need to EXPOSE port — RunPod handles networking
# Listen on $PORT (set by RunPod at runtime)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
