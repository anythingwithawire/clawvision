"""OpenClaw bridge for executing tool calls."""

import json
import logging
from typing import Any, Dict, Optional

import requests


class OpenClawBridge:
    """Bridge to OpenClaw gateway for executing tool calls."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 18789,
        token: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.token = token
        self.logger = logging.getLogger('OpenClawBridge')
        
        self._base_url = f"http://{host}:{port}"
        self._session = requests.Session()
        
        if token:
            self._session.headers.update({
                "Authorization": f"Bearer {token}"
            })
    
    def health_check(self) -> bool:
        """Check if OpenClaw gateway is accessible.
        
        Returns:
            True if gateway is healthy
        """
        try:
            response = self._session.get(
                f"{self._base_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"OpenClaw health check failed: {e}")
            return False
    
    def execute(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a task via OpenClaw.
        
        This is the main entry point for tool calls from Gemini.
        
        Args:
            task: Natural language task description
            context: Optional context for the task
            
        Returns:
            Result dictionary with status and response
        """
        self.logger.info(f"🔧 Executing task: {task[:80]}...")
        
        try:
            # Use OpenClaw's chat completions endpoint
            # This routes through the agent which can use any configured skill
            payload = {
                "model": "openclaw",
                "messages": [
                    {
                        "role": "user",
                        "content": task
                    }
                ],
                "stream": False
            }
            
            if context:
                # Add context as system message
                context_msg = "Context: " + json.dumps(context)
                payload["messages"].insert(0, {
                    "role": "system",
                    "content": context_msg
                })
            
            response = self._session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response content
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    
                    self.logger.info("✅ Task executed successfully")
                    return {
                        "status": "success",
                        "result": content,
                        "raw": result
                    }
                else:
                    return {
                        "status": "error",
                        "error": "Empty response from OpenClaw"
                    }
            else:
                error_msg = f"OpenClaw error: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            self.logger.error("OpenClaw request timed out")
            return {
                "status": "error",
                "error": "Request timed out (60s)"
            }
        except Exception as e:
            self.logger.error(f"OpenClaw execution error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def list_skills(self) -> Optional[list]:
        """List available OpenClaw skills.
        
        Returns:
            List of skill names, or None if failed
        """
        try:
            # Try to get skills from OpenClaw
            # Note: This endpoint may vary based on OpenClaw version
            response = self._session.get(
                f"{self._base_url}/v1/skills",
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json().get("skills", [])
            
            # Fallback: try alternative endpoint
            response = self._session.get(
                f"{self._base_url}/skills",
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            self.logger.warning(f"Could not list skills: {e}")
        
        return None
    
    def close(self):
        """Close the HTTP session."""
        self._session.close()
