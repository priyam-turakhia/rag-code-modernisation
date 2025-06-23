"""
Service for interacting with DeepSeek Coder via Ollama
"""
import requests
from typing import Dict, Optional, List
import re
import logging

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_MAX_TOKENS
from schemas import RetrievedChunk

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"

    def create_prompt(
        self, 
        code: str,
        numpy_version: str,
        detected_functions: List[str],
        retrieved_context: Dict[str, List[RetrievedChunk]]
    ) -> str:
        context_parts = []
        for func_name, chunks in retrieved_context.items():
            if chunks and chunks[0].similarity_score > 0.7:
                context_parts.append(f"\n{func_name}: {chunks[0].content[:200]}")

        context = "\n".join(context_parts) if context_parts else ""

        prompt = f"""You are a NumPy expert. Modernize this NumPy {numpy_version} code by replacing deprecated functions.

Code:
```python
{code}
```

Detected functions: {', '.join(detected_functions)}

{context}

Provide:
1. Modernized code
2. What was deprecated and the modern replacement

Response:"""

        return prompt

    def parse_response(self, response_text: str, original_code: str) -> Dict:
        try:
            logger.debug("Raw model response: %s", response_text)

            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response_text, re.DOTALL)
            modernized_code = code_blocks[0].strip() if code_blocks else original_code

            deprecation_info = ""
            explanation_lines = []
            for line in response_text.splitlines():
                if any(keyword in line.lower() for keyword in ["deprecated", "replacement", "use"]):
                    explanation_lines.append(line.strip())
            deprecation_info = " ".join(explanation_lines[:3]) if explanation_lines else "No deprecation explanation."

            deprecated_funcs = list(set(re.findall(r'(\w+) is deprecated', response_text)))

            return {
                "modernized_code": modernized_code,
                "deprecated_functions": deprecated_funcs,
                "deprecation_info": deprecation_info,
                "success": modernized_code != original_code
            }

        except Exception as e:
            logger.exception("Failed to parse model response")
            return {
                "modernized_code": original_code,
                "deprecated_functions": [],
                "deprecation_info": f"Error parsing response: {str(e)}",
                "success": False
            }

    def call_model(
        self,
        code: str,
        numpy_version: str,
        detected_functions: List[str],
        retrieved_context: Dict[str, List[RetrievedChunk]]
    ) -> Dict:
        prompt = self.create_prompt(code, numpy_version, detected_functions, retrieved_context)

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": MODEL_TEMPERATURE,
                        "num_predict": MODEL_MAX_TOKENS
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return self.parse_response(result.get("response", ""), code)
            else:
                logger.error("Model call failed with status code %d: %s", response.status_code, response.text)
                return {
                    "modernized_code": code,
                    "deprecated_functions": [],
                    "deprecation_info": f"Model request failed with status code {response.status_code}",
                    "success": False
                }

        except Exception as e:
            logger.exception("Exception during model call")
            return {
                "modernized_code": code,
                "deprecated_functions": [],
                "deprecation_info": f"Error calling model: {str(e)}",
                "success": False
            }

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
