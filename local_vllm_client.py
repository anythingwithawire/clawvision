"""Local vLLM client for vision + text generation."""

import base64
import json
import logging
from typing import Optional

import requests


class LocalVLLMClient:
    """Client for local vLLM server with vision capabilities."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger('LocalVLLM')
    
    def test_connection(self) -> bool:
        """Test if vLLM server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Connected to local vLLM")
                return True
        except Exception as e:
            self.logger.error(f"❌ Cannot connect to vLLM: {e}")
        return False
    
    def generate_with_image(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from prompt + optional image.
        
        Args:
            prompt: User prompt text
            image_bytes: JPEG image bytes (optional)
            system_prompt: System instructions (optional)
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message
        user_content = []
        
        if image_bytes:
            # Encode image as base64 data URL
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            })
        
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        try:
            self.logger.info(f"🖼️ Sending request to vLLM...")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                self.logger.info("✅ Got response from vLLM")
                return content
            else:
                error_msg = f"vLLM error: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except requests.exceptions.Timeout:
            self.logger.error("vLLM request timed out")
            return "Error: Request timed out (30s)"
        except Exception as e:
            self.logger.error(f"vLLM request failed: {e}")
            return f"Error: {str(e)}"
    
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text only (no image)."""
        return self.generate_with_image(prompt, None, system_prompt)
