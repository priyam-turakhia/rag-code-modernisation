"""
FastAPI application for NumPy code modernization RAG service
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import API_TITLE, API_VERSION, API_HOST, API_PORT
from schemas import (
    CodeAnalysisRequest,
    CodeAnalysisResponse,
    HealthResponse
)
from rag_service import RAGService

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="RAG-based API for modernizing NumPy code by detecting deprecated functions and suggesting alternatives"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        chroma_db_connected=rag_service.is_connected(),
        ollama_available=rag_service.is_model_available()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        chroma_db_connected=rag_service.is_connected(),
        ollama_available=rag_service.is_model_available()
    )


@app.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    try:
        if not rag_service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Vector database is not available"
            )
        
        result = rag_service.analyze_code(
            code=request.code,
            numpy_version=request.numpy_version
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing code: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )