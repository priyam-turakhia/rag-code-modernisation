from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import API_TITLE, API_VERSION, API_HOST, API_PORT
from schemas import CodeAnalysisRequest, CodeAnalysisResponse, HealthResponse
from rag_service import RAGService

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title = API_TITLE, version = API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None

# Get available models
def get_available_models():
    models_dir = Path(__file__).parent.parent / "fine-tuning" / "models"
    models = []
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                models.append(model_dir.name)
    return models

# Prompt for model selection
def select_model():
    models = get_available_models()
    if not models:
        print("No fine-tuned models found. Using Ollama fallback.")
        return None
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print(f"{len(models) + 1}. Use Ollama (deepseek-coder)")
    
    while True:
        try:
            choice = input("\nSelect model (number): ")
            idx = int(choice) - 1
            if idx == len(models):
                return None
            if 0 <= idx < len(models):
                return models[idx]
        except:
            print("Invalid selection. Try again.")

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status = "healthy",
        chroma_connected = rag.is_connected() if rag else False,
        model_available = rag.is_model_available() if rag else False
    )

@app.post("/analyze", response_model = CodeAnalysisResponse)
async def analyze(req: CodeAnalysisRequest) -> CodeAnalysisResponse:
    logger.info(f"Analyzing code: {len(req.code)} chars, NumPy {req.numpy_version}")
    
    if not rag.is_connected():
        raise HTTPException(status_code = 503, detail = "Vector database unavailable")
    
    try:
        result = rag.analyze_code(req.code, req.numpy_version)
        
        if result.error:
            logger.error(f"Analysis error: {result.error}")
        else:
            logger.info(f"Analysis successful: {len(result.changes)} changes, {len(result.retrieved_context)} functions analyzed")
        
        return result
        
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code = 500, detail = str(e))

@app.on_event("startup")
async def startup() -> None:
    global rag
    selected_model = select_model()
    rag = RAGService(selected_model)
    
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Model: {selected_model or 'Ollama (deepseek-coder)'}")
    logger.info(f"ChromaDB: {'connected' if rag.is_connected() else 'disconnected'}")
    logger.info(f"Model: {'available' if rag.is_model_available() else 'unavailable'}")

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)