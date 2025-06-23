"""
RAG service for NumPy code modernization
"""
import ast
import re
from typing import List, Dict, Set, Tuple, Optional, Any
import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    NUMPY_ALIASES
)
from schemas import FunctionInfo, RetrievedChunk, AugmentedPrompt, ModelResponse, CodeAnalysisResponse
from model_service import ModelService


class NumpyFunctionExtractor(ast.NodeVisitor):
    """AST visitor to extract NumPy function calls from Python code"""
    
    def __init__(self, numpy_aliases: Set[str]):
        self.numpy_aliases = numpy_aliases
        self.functions = []
        self.current_line = 0
        
    def visit_Call(self, node):
        """Visit function calls in the AST"""
        func_name = self._get_function_name(node.func)
        
        if func_name and self._is_numpy_call(func_name):
            # Extract the full call as string
            full_call = ast.unparse(node) if hasattr(ast, 'unparse') else self._node_to_string(node)
            
            function_info = FunctionInfo(
                function_name=func_name.split('.')[-1],  # Get just the function name
                line_number=node.lineno,
                full_call=full_call
            )
            self.functions.append(function_info)
            
        self.generic_visit(node)
    
    def _get_function_name(self, node) -> str:
        """Extract function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ""
    
    def _is_numpy_call(self, func_name: str) -> bool:
        """Check if the function call is a NumPy call"""
        parts = func_name.split('.')
        return len(parts) > 1 and parts[0] in self.numpy_aliases
    
    def _node_to_string(self, node) -> str:
        """Fallback method to convert AST node to string"""
        # Simple reconstruction for older Python versions
        if isinstance(node, ast.Call):
            func_str = self._get_function_name(node.func)
            args = ", ".join(str(arg) for arg in node.args)
            return f"{func_str}({args})"
        return ""


class RAGService:
    """Main RAG service for code modernization"""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self._init_chroma()
        self.model_service = ModelService()
    
    def _init_chroma(self):
        """Initialize ChromaDB connection"""
        try:
            # Use sentence transformers embedding function
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            
            embed_fn = SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-base-en-v1.5",
                device="cpu"  # or "cuda" if you have GPU
            )
            
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                # Try to get existing collection with embedding function
                self.collection = self.chroma_client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embed_fn
                )
                print(f"Successfully connected to existing collection: {COLLECTION_NAME}")
            except Exception:
                # Collection doesn't exist, create it
                print(f"Collection {COLLECTION_NAME} not found. Creating new collection...")
                self.collection = self.chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embed_fn,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.collection = None
    
    def extract_numpy_functions(self, code: str) -> List[FunctionInfo]:
        """Extract NumPy function calls from Python code"""
        try:
            # First, detect numpy import aliases in the code
            aliases = self._detect_numpy_imports(code)
            aliases.update(NUMPY_ALIASES)  # Add default aliases
            
            # Parse the code and extract functions
            tree = ast.parse(code)
            extractor = NumpyFunctionExtractor(aliases)
            extractor.visit(tree)
            
            return extractor.functions
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return []
        except Exception as e:
            print(f"Error extracting functions: {e}")
            return []
    
    def _detect_numpy_imports(self, code: str) -> Set[str]:
        """Detect NumPy import aliases from the code"""
        aliases = set()
        
        # Regex patterns for different import styles
        patterns = [
            r'import\s+numpy\s+as\s+(\w+)',  # import numpy as np
            r'from\s+numpy\s+import',  # from numpy import ...
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            aliases.update(matches)
        
        # Always include 'numpy' itself
        if 'from numpy import' in code or 'import numpy' in code:
            aliases.add('numpy')
            
        return aliases
    
    def query_vector_db(self, function_name: str, numpy_version: str) -> List[RetrievedChunk]:
        """Query ChromaDB for relevant documentation chunks"""
        if not self.collection:
            return []
        
        try:
            # Build query with function name and version context
            query_text = f"{function_name} numpy {numpy_version} deprecated deprecation replacement alternative"
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=TOP_K_RESULTS,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to RetrievedChunk objects
            chunks = []
            
            # Safely check if we have results
            documents = results.get('documents', [])
            distances = results.get('distances', [])
            metadatas = results.get('metadatas', [])
            
            if documents and len(documents) > 0 and len(documents[0]) > 0:
                for i in range(len(documents[0])):
                    # Calculate similarity score from distance (assuming cosine distance)
                    # Default to 0 similarity if distances not available
                    if distances and len(distances) > 0 and len(distances[0]) > i:
                        similarity = 1 - distances[0][i]
                    else:
                        similarity = 0.5  # Default similarity if not provided
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        # Extract metadata safely
                        metadata_dict = {}
                        if metadatas and len(metadatas) > 0 and len(metadatas[0]) > i:
                            # Convert metadata to dict if it's not already
                            meta = metadatas[0][i]
                            if meta:
                                metadata_dict = dict(meta) if not isinstance(meta, dict) else meta
                        
                        chunk = RetrievedChunk(
                            content=documents[0][i],
                            metadata=metadata_dict,
                            similarity_score=similarity
                        )
                        chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error querying vector DB for {function_name}: {e}")
            return []
    
    def build_augmented_prompt(
        self,
        code: str,
        numpy_version: str,
        functions: List[FunctionInfo],
        retrieved_chunks: Dict[str, List[RetrievedChunk]]
    ) -> AugmentedPrompt:
        """Build an augmented prompt with retrieved context"""
        
        # Simplified system prompt for baseline testing
        system_prompt = (
            "You are an expert Python developer. Your task is to modernize NumPy code by "
            "replacing deprecated functions with their modern equivalents."
        )
        
        # Build user prompt
        user_prompt_parts = [
            f"The following Python code uses NumPy version {numpy_version}:",
            f"\n```python\n{code}\n```\n",
            "\nDetected NumPy functions:"
        ]
        
        # Add function information
        for func in functions:
            user_prompt_parts.append(f"- {func.function_name} (line {func.line_number})")
        
        user_prompt_parts.append("\nPlease modernize this code.")
        user_prompt = "\n".join(user_prompt_parts)
        
        return AugmentedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieved_context=retrieved_chunks
        )
    
    def analyze_code(self, code: str, numpy_version: str) -> CodeAnalysisResponse:

        functions = self.extract_numpy_functions(code)
        
        retrieved_chunks = {}
        unique_functions = {func.function_name for func in functions}
        
        for func_name in unique_functions:
            chunks = self.query_vector_db(func_name, numpy_version)
            retrieved_chunks[func_name] = chunks
        
        augmented_prompt = self.build_augmented_prompt(
            code=code,
            numpy_version=numpy_version,
            functions=functions,
            retrieved_chunks=retrieved_chunks
        )

        if self.model_service.is_available():
            model_result = self.model_service.call_model(
                code=code,
                numpy_version=numpy_version,
                detected_functions=list(unique_functions),
                retrieved_context=retrieved_chunks
            )

            model_response = ModelResponse(
                modernized_code=model_result["modernized_code"],
                deprecation_context=model_result["deprecation_info"],
                success=model_result["success"]
            )
        else:
            model_response = None

        return CodeAnalysisResponse(
            original_code=code,
            detected_functions=functions,
            augmented_prompt=augmented_prompt,
            model_response=model_response
        )
    
    def is_connected(self) -> bool:
        """Check if ChromaDB is connected"""
        return self.collection is not None
    
    def is_model_available(self) -> bool:
        """Check if Ollama model is available"""
        return self.model_service.is_available()