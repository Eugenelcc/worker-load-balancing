# CUDA 12.1 image (use devel so build tools exist if any wheel falls back to source)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# (Optional but commonly needed on RunPod) ensure CUDA compat libs are registered
RUN ldconfig /usr/local/cuda-12.1/compat/ || true

# System deps: python, pip, wget, build tools (safe even if wheels are used)
RUN apt-get update -y && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    git \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# ---- Model download (same idea as Dockerfile #2) ----
# If you want a different model, change this URL.
RUN wget -O /app/model.gguf \
  "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# ---- Python deps ----
# Install llama-cpp-python CUDA wheel for cu121 + your API deps
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Copy app
COPY app.py /app/app.py

# RunPod typically sets PORT, your app.py uses PORT default 5000 anyway
ENV PORT=5000
ENV MODEL_PATH=/app/model.gguf

# Start like Dockerfile #1 (works with RunPod)
CMD ["python3", "app.py"]
