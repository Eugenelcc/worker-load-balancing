import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str

request_count = 0

# Required for Runpod to monitor worker health
@app.get("/ping")
async def health_check():
    # The response body doesn't matter, it's all about the status code:
    # 200 - Healthy
    # 204 - Initializing (maybe your model is still loading)
    # Anything else - Unhealthy
    return JSONResponse(status_code=200, content={"status": "healthy"})

# Our custom generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    global request_count
    request_count += 1
    # Simple mock implementation; you'd do your actual work here!
    generated_text = f"Response to: {request.prompt} (request #{request_count})"
    return {"generated_text": generated_text}

@app.get("/stats")
async def stats():
    return {"total_requests": request_count}

@app.get("/")
async def root():
    datacenter_id = os.getenv("RUNPOD_DC_ID")
    return {"message": f"Hello, {datacenter_id if datacenter_id else 'world'}!".strip()}

if __name__ == "__main__":
    import uvicorn

    # When you deploy your endpoint, make sure to expose port 5000
    # And add it as an environment variable in the Runpod console
    port = int(os.getenv("PORT", "5000"))

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)