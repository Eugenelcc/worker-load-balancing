import os
import time
from typing import List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from llama_cpp import Llama

app = FastAPI(title="RunPod FastAPI + llama.cpp")

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = os.getenv("MODEL_PATH", "./model.gguf")

N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
GPU_LAYER_COUNT = int(os.getenv("GPU_LAYER_COUNT", "30"))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are AskVox, a safe educational AI tutor.")

# -----------------------
# GLOBALS
# -----------------------
llm: Optional[Llama] = None
model_ready = False
model_error: Optional[str] = None

request_count = 0

# -----------------------
# SCHEMAS
# -----------------------
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []
    model_config = ConfigDict(extra="allow")

class ChatResponse(BaseModel):
    response: str

# -----------------------
# MODEL LOADING
# -----------------------
def load_model():
    global llm, model_ready, model_error

    try:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")

        print(f"Loading model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=GPU_LAYER_COUNT,
            verbose=False,
        )

        model_ready = True
        model_error = None
        print("✅ Model loaded successfully.")

    except Exception as e:
        model_ready = False
        model_error = str(e)
        print(f"❌ Model failed to load: {model_error}")

@app.on_event("startup")
async def startup_event():
    load_model()

# -----------------------
# RUNPOD REQUIRED ENDPOINT
# -----------------------
@app.get("/ping")
async def health_check():
    """
    RunPod Load Balancer health endpoint.
    200 = healthy/ready
    204 = initializing (model still loading)
    other = unhealthy
    """
    if model_ready:
        return JSONResponse(status_code=200, content={"status": "healthy", "model": "ready"})
    if model_error:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": model_error})
    return JSONResponse(status_code=204, content={"status": "initializing"})

# -----------------------
# OPTIONAL DEBUG ENDPOINTS
# -----------------------
@app.get("/stats")
async def stats():
    return {
        "total_requests": request_count,
        "model_ready": model_ready,
        "model_error": model_error,
    }

@app.get("/")
async def root():
    datacenter_id = os.getenv("RUNPOD_DC_ID")
    return {"message": f"Hello, {datacenter_id if datacenter_id else 'world'}!".strip()}

# -----------------------
# INFERENCE ENDPOINTS
# -----------------------
 
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    History-based endpoint like your App.py #2.
    """
    global request_count, llm

    if not model_ready or llm is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    request_count += 1

    parts = [f"[SYSTEM] {SYSTEM_PROMPT}"]
    for h in req.history[-3:]:
        if h.role == "user":
            parts.append(f"[USER] {h.content}")
        else:
            parts.append(f"[ASSISTANT] {h.content}")

    parts.append(f"[USER] {req.message}")
    parts.append("[ASSISTANT]")
    prompt = "\n".join(parts)

    t0 = time.perf_counter()
    try:
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            echo=False,
        )
        text = output["choices"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        t1 = time.perf_counter()
        print(f"⏱️ /chat time: {t1 - t0:.2f}s (request #{request_count})")

    return ChatResponse(response=text.strip())

# -----------------------
# LOCAL RUN (RunPod uses this too if you start app.py)
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
