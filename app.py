from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Literal
from llama_cpp import Llama
import os
import time

# Initialize FastAPI app
app = FastAPI(title="AskVox LLaMA2 CloudRun")

# --- CONFIGURATION ---
MODEL_PATH = "./model.gguf"
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
GPU_LAYER_COUNT = int(os.getenv("GPU_LAYER_COUNT", "30"))

# Global model instance
llm = None

def load_model():
    global llm
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
    print("✅ Model loaded successfully.")

# Load model at startup
@app.on_event("startup")
async def startup_event():
    load_model()

# ---------- SCHEMAS ----------
class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []

    model_config = ConfigDict(extra="allow")

class ChatResponse(BaseModel):
    response: str

# ---------- ENDPOINTS ----------

# 🔴 REQUIRED for RunPod Load Balancer: /ping
@app.get("/ping")
def ping():
    """Health check endpoint required by RunPod Load Balancer."""
    return {"status": "ok", "message": "Model is ready"}

# Optional: keep /health for your own debugging
@app.get("/health")
def health():
    return {"status": "online", "system": "RunPod GPU"}

# Main inference endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    SYSTEM_PROMPT = "You are AskVox, a safe educational AI tutor."
    parts = [f"[SYSTEM] {SYSTEM_PROMPT}"]

    for h in req.history[-3:]:
        if h.role == "user":
            parts.append(f"[USER] {h.content}")
        else:
            parts.append(f"[ASSISTANT] {h.content}")

    parts.append(f"[USER] {req.message}")
    parts.append("[ASSISTANT]")

    prompt = "\n".join(parts)

    print("🧠 RunPod LLaMA: generating...")
    t0 = time.perf_counter()
    try:
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            echo=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    t1 = time.perf_counter()
    print(f"⏱️ RunPod generation time: {t1 - t0:.2f}s")

    try:
        text = output["choices"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bad model output: {e}")

    return ChatResponse(response=text.strip())