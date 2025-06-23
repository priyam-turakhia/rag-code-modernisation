"""
Pydantic models for request and response schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis"""
    code: str = Field(..., description="The Python code to analyze")
    numpy_version: str = Field(..., description="NumPy version in user's environment (e.g., '1.24.0')")
    context: Optional[str] = Field(None, description="Additional context about the code")


class FunctionInfo(BaseModel):
    """Information about a detected NumPy function"""
    function_name: str = Field(..., description="Name of the NumPy function")
    line_number: int = Field(..., description="Line number where function appears")
    full_call: str = Field(..., description="Full function call including arguments")


class RetrievedChunk(BaseModel):
    """Retrieved documentation chunk from vector DB"""
    content: str = Field(..., description="Content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the chunk")
    similarity_score: float = Field(..., description="Similarity score")


class AugmentedPrompt(BaseModel):
    """Augmented prompt ready for the model"""
    system_prompt: str = Field(..., description="System prompt for the model")
    user_prompt: str = Field(..., description="User prompt with code and context")
    retrieved_context: Dict[str, List[RetrievedChunk]] = Field(
        default_factory=dict, 
        description="Retrieved chunks organized by function name"
    )


class ModelResponse(BaseModel):
    """Simplified model response for testing baseline performance"""
    modernized_code: str = Field(..., description="Modernized code without deprecated functions")
    deprecation_context: str = Field(..., description="Information about deprecated functions")
    success: bool = Field(default=True, description="Whether modernization was successful")


class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis"""
    original_code: str = Field(..., description="Original code submitted")
    detected_functions: List[FunctionInfo] = Field(..., description="Detected NumPy functions")
    augmented_prompt: AugmentedPrompt = Field(..., description="Augmented prompt for model")
    model_response: Optional["ModelResponse"] = Field(None, description="Model's modernization response")
    

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    chroma_db_connected: bool = Field(..., description="ChromaDB connection status")
    ollama_available: bool = Field(..., description="Ollama model availability")