"""
Configuration settings for the RAG service
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

# ChromaDB settings
CHROMA_DB_DIR = str(DOCS_DIR / "chroma_db")
COLLECTION_NAME = "numpy_docs"

# Model settings
OLLAMA_MODEL = "deepseek-coder:6.7b"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_TEMPERATURE = 0.2
MODEL_MAX_TOKENS = 2048

# RAG settings
TOP_K_RESULTS = 3  # Number of top chunks to retrieve per function
CHUNK_SIZE = 500  # Approximate chunk size in characters
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score for relevant chunks

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "NumPy Code Modernization RAG API"
API_VERSION = "1.0.0"

# Numpy function extraction settings
NUMPY_ALIASES = ["np", "numpy"]  # Common numpy import aliases