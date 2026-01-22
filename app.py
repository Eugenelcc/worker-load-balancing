import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from llama_cpp import Llama

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
GPU_LAYER_COUNT = int(os.getenv("GPU_LAYER_COUNT", "30"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are AskVox, a safe educational AI tutor.")

llm: Optional[Llama] = None
model_ready = False
model_error: Optional[str] = None
request_count = 0

def load_model():
    global llm, model_ready, model_error
    try:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_model()
    yield
    # shutdown (optional cleanup)
    # if llm: del llm

app = FastAPI(title="RunPod FastAPI + llama.cpp", lifespan=lifespan)

# ---- schemas (same as before) ----
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

@app.get("/ping")
async def ping():
    if model_ready:
        return JSONResponse(status_code=200, content={"status": "healthy", "model": "ready"})
    if model_error:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": model_error})
    return JSONResponse(status_code=204, content={"status": "initializing"})

@app.get("/stats")
async def stats():
    return {"total_requests": request_count, "model_ready": model_ready, "model_error": model_error}

@app.post("/generate", response_model=GenerationResponse)
async def generate(req: GenerationRequest):
    global request_count, llm
    if not model_ready or llm is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    request_count += 1
    out = llm(req.prompt, max_tokens=req.max_tokens, temperature=req.temperature, echo=False)
    return GenerationResponse(generated_text=out["choices"][0]["text"].strip())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

class InputModel(BaseModel):
    input: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: InputModel):
    global request_count, llm, model_ready
    if not model_ready or llm is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    request_count += 1

    # Extract prompt string from input
    prompt = req.input.get("message", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt in input.")

    out = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, echo=False)
    return ChatResponse(response=out["choices"][0]["text"].strip())

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
