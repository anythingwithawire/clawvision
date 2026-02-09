"""
Gemini REST API Client

Uses standard HTTP REST API for multimodal (image + audio) generation.
Falls back from Live API for free tier users.
"""

import base64
import json
import logging
from typing import Optional, Callable

import requests


class GeminiRestClient:
    """
    REST API client for Google Gemini API.
    
    Supports:
    - Image input (JPEG)
    - Audio input (WAV)
    - Text output (for TTS processing)
    - Tool calling via function declarations
    """
    
    # Gemini REST API endpoint
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        on_tool_call: Optional[Callable[[str, dict], dict]] = None,
        on_text_response: Optional[Callable[[str], None]] = None
    ):
        self.api_key = api_key
        self.model = model
        self.on_tool_call = on_tool_call
        self.on_text_response = on_text_response
        
        self.logger = logging.getLogger('GeminiRest')
        self._session = requests.Session()
    
    def _build_request_body(
        self,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        text_prompt: Optional[str] = None
    ) -> dict:
        """
        Build the request body for multimodal generation.
        
        Args:
            image_bytes: JPEG image data
            audio_bytes: WAV audio data
            text_prompt: Additional text prompt
            
        Returns:
            Request body dictionary
        """
        parts = []
        
        # Add image if provided
        if image_bytes:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_b64
                }
            })
        
        # Add audio if provided
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            parts.append({
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_b64
                }
            })
        
        # Add text prompt
        prompt_text = text_prompt or ""
        
        # If no media provided, use a default prompt
        if not parts:
            prompt_text = prompt_text or "Hello, I'm here. What can you see or hear?"
        
        # Add the text prompt
        parts.append({
            "text": prompt_text + """

You are ClawVision, a helpful AI assistant with vision and voice capabilities.
You can see through the user's phone camera and hear their voice.

If the user asks you to do something on their computer (like check files, run commands, 
search the web, send messages, etc.), use the openclaw_execute tool to execute the task.

Be concise but helpful. When you see something interesting, describe it briefly.
"""
        })
        
        body = {
            "contents": [{
                "parts": parts
            }],
            "tools": [
                {"google_search": {}},
                {
                    "function_declarations": [
                        {
                            "name": "openclaw_execute",
                            "description": "Execute a task through the OpenClaw gateway. Use this for ANY computer-related tasks like: checking files, running shell commands, searching the web, sending messages, controlling the system, etc.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "task": {
                                        "type": "string",
                                        "description": "Natural language description of what to do. Be specific about the task."
                                    },
                                    "context": {
                                        "type": "object",
                                        "description": "Optional context object with additional information"
                                    }
                                },
                                "required": ["task"]
                            }
                        }
                    ]
                }
            ],
            "generation_config": {
                "temperature": 0.7,
                "max_output_tokens": 1024
            }
        }
        
        return body
    
    def process_frame_and_audio(
        self,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        text_prompt: Optional[str] = None
    ) -> str:
        """
        Process image and/or audio through Gemini REST API.
        
        Args:
            image_bytes: JPEG image data
            audio_bytes: WAV audio data  
            text_prompt: Optional text prompt
            
        Returns:
            Text response from Gemini
        """
        url = self.API_URL.format(model=self.model)
        
        params = {"key": self.api_key}
        
        body = self._build_request_body(image_bytes, audio_bytes, text_prompt)
        
        self.logger.debug(f"Sending request to {url}")
        
        try:
            response = self._session.post(
                url,
                params=params,
                json=body,
                timeout=60.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            return self._process_response(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return f"Sorry, something went wrong: {str(e)}"
    
    def _process_response(self, data: dict) -> str:
        """
        Process API response and handle tool calls.
        
        Args:
            data: Response JSON from API
            
        Returns:
            Text response
        """
        try:
            # Check for errors
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown error")
                self.logger.error(f"API error: {error_msg}")
                return f"Error: {error_msg}"
            
            # Get candidates
            candidates = data.get("candidates", [])
            if not candidates:
                self.logger.warning("No candidates in response")
                return "I didn't get a response. Please try again."
            
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            full_text = []
            
            for part in parts:
                # Handle text
                if "text" in part:
                    full_text.append(part["text"])
                
                # Handle function calls
                if "function_call" in part:
                    func_call = part["function_call"]
                    name = func_call.get("name", "")
                    args = func_call.get("args", {})
                    
                    self.logger.info(f"Tool call: {name}")
                    
                    if name == "openclaw_execute" and self.on_tool_call:
                        # Execute the tool
                        result = self.on_tool_call(name, args)
                        
                        # Make a follow-up request with the tool result
                        return self._handle_tool_result(result)
            
            response_text = " ".join(full_text) if full_text else "I see. What else can I help you with?"
            
            if self.on_text_response:
                self.on_text_response(response_text)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return "Sorry, I had trouble processing the response."
    
    def _handle_tool_result(self, result: dict) -> str:
        """
        Handle tool execution result and get final response.
        
        Args:
            result: Tool execution result
            
        Returns:
            Final text response
        """
        # Build a follow-up request with the tool result
        status = result.get("status", "unknown")
        result_text = result.get("result", "") if status == "success" else result.get("error", "Unknown error")
        
        follow_up_prompt = f"""
I executed the task for you. Here's what happened:

Status: {status}
Result: {result_text}

Please summarize this for the user in a friendly way.
"""
        
        # Make a simple text-only follow-up request
        url = self.API_URL.format(model=self.model)
        params = {"key": self.api_key}
        
        body = {
            "contents": [{
                "parts": [{"text": follow_up_prompt}]
            }],
            "generation_config": {
                "temperature": 0.7,
                "max_output_tokens": 1024
            }
        }
        
        try:
            response = self._session.post(
                url,
                params=params,
                json=body,
                timeout=30.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if "text" in p]
                return " ".join(text_parts) if text_parts else "Done!"
            
            return "Task completed."
            
        except Exception as e:
            # Fallback response
            return f"Task completed with status: {status}. Result: {result_text[:200]}"
    
    def test_connection(self) -> bool:
        """
        Test API connectivity with a simple request.
        
        Returns:
            True if connection successful
        """
        url = self.API_URL.format(model=self.model)
        params = {"key": self.api_key}
        
        body = {
            "contents": [{
                "parts": [{"text": "Say 'Connected successfully' and nothing else."}]
            }]
        }
        
        try:
            response = self._session.post(
                url,
                params=params,
                json=body,
                timeout=10.0
            )
            
            if response.status_code == 200:
                self.logger.info("✅ API connection test successful")
                return True
            else:
                self.logger.error(f"API test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"API test error: {e}")
            return False
