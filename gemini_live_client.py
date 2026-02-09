"""
Gemini Live API WebSocket Client

Handles connection to Gemini Live API, sending audio/video,
and receiving responses including tool calls.
"""

import asyncio
import base64
import json
import logging
import struct
from typing import Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection


class GeminiLiveClient:
    """
    WebSocket client for Google Gemini Live API.
    
    Supports:
    - BidiGenerateContent for real-time streaming
    - Audio input (PCM 16kHz, 16-bit)
    - Video input (JPEG frames)
    - Audio output (PCM 24kHz, 16-bit)
    - Tool calling
    """
    
    # Gemini Live API endpoint
    API_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        on_tool_call: Optional[Callable[[str, dict], dict]] = None,
        on_audio_response: Optional[Callable[[bytes], None]] = None,
        on_text_response: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        self.api_key = api_key
        self.model = model
        self.on_tool_call = on_tool_call
        self.on_audio_response = on_audio_response
        self.on_text_response = on_text_response
        self.on_error = on_error
        
        self.logger = logging.getLogger('GeminiLive')
        self.websocket: Optional[ClientConnection] = None
        self.is_connected = False
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._setup_message: Optional[dict] = None
    
    def _build_setup_message(self) -> dict:
        """Build the initial setup message with configuration."""
        return {
            "setup": {
                "model": f"models/{self.model}",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Puck"  # Options: Puck, Charon, Kore, Fenrir, Aoede
                            }
                        }
                    }
                },
                "system_instruction": {
                    "parts": [{
                        "text": """You are ClawVision, a helpful AI assistant with vision and voice capabilities.
You can see through the user's phone camera and hear their voice.
You can also execute tools on the user's computer through the OpenClaw gateway.

Be concise but helpful. When you see something interesting, describe it briefly.
When the user asks you to do something, use the available tools if appropriate.
"""
                    }]
                },
                "tools": [
                    # Google Search tool (built-in)
                    {"google_search": {}},
                    # OpenClaw function calling - we'll handle these specially
                    {
                        "function_declarations": [
                            {
                                "name": "openclaw_execute",
                                "description": "Execute a tool through the OpenClaw gateway",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "tool_name": {
                                            "type": "string",
                                            "description": "Name of the OpenClaw tool to execute"
                                        },
                                        "parameters": {
                                            "type": "object",
                                            "description": "Parameters to pass to the tool"
                                        }
                                    },
                                    "required": ["tool_name", "parameters"]
                                }
                            }
                        ]
                    }
                ]
            }
        }
    
    async def connect(self):
        """Establish WebSocket connection to Gemini Live API."""
        url = f"{self.API_URL}?key={self.api_key}"
        
        self.logger.info(f"🔌 Connecting to Gemini Live API...")
        
        try:
            self.websocket = await websockets.connect(
                url,
                additional_headers={
                    "Content-Type": "application/json"
                }
            )
            
            # Send setup message
            setup_msg = self._build_setup_message()
            await self.websocket.send(json.dumps(setup_msg))
            
            # Wait for setup completion
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if "setupComplete" in data:
                self.is_connected = True
                self.logger.info("✅ Connected to Gemini Live API")
            else:
                raise ConnectionError(f"Unexpected setup response: {data}")
                
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise
    
    async def disconnect(self):
        """Close the WebSocket connection."""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.logger.info("🔌 Disconnected from Gemini Live API")
    
    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to Gemini.
        
        Args:
            audio_data: PCM 16-bit signed audio data at 16kHz
        """
        if not self.is_connected or not self.websocket:
            return
        
        # Base64 encode the audio
        encoded = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": encoded,
                        "mime_type": "audio/pcm;rate=16000"
                    }
                ]
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send audio: {e}")
    
    async def send_frame(self, jpeg_data: bytes):
        """
        Send a video frame to Gemini.
        
        Args:
            jpeg_data: JPEG encoded image data
        """
        if not self.is_connected or not self.websocket:
            return
        
        # Base64 encode the image
        encoded = base64.b64encode(jpeg_data).decode('utf-8')
        
        message = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": encoded,
                        "mime_type": "image/jpeg"
                    }
                ]
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send frame: {e}")
    
    async def receive_loop(self):
        """Main loop to receive and process responses from Gemini."""
        self.logger.info("🎯 Starting receive loop...")
        
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=30.0
                    )
                    
                    await self._process_message(message)
                    
                except asyncio.TimeoutError:
                    # Send a keepalive or just continue
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Receive loop error: {e}")
            self.is_connected = False
    
    async def _process_message(self, message):
        """Process incoming message from Gemini."""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                # Handle binary messages if any
                data = json.loads(message.decode('utf-8'))
            
            # Check for server content
            if "serverContent" in data:
                content = data["serverContent"]
                
                # Handle text
                if "modelTurn" in content:
                    turn = content["modelTurn"]
                    if "parts" in turn:
                        for part in turn["parts"]:
                            # Text response
                            if "text" in part:
                                text = part["text"]
                                if self.on_text_response:
                                    await self._safe_callback(self.on_text_response, text)
                            
                            # Inline data (audio)
                            if "inlineData" in part:
                                inline = part["inlineData"]
                                if inline.get("mimeType", "").startswith("audio/"):
                                    audio_data = base64.b64decode(inline["data"])
                                    if self.on_audio_response:
                                        await self._safe_callback(self.on_audio_response, audio_data)
                
                # Handle turn completion
                if "turnComplete" in content:
                    self.logger.debug("Turn complete")
            
            # Check for tool calls
            if "toolCall" in data:
                await self._handle_tool_call(data["toolCall"])
            
            # Check for tool call cancellation
            if "toolCallCancellation" in data:
                self.logger.debug("Tool call cancelled")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def _handle_tool_call(self, tool_call_data: dict):
        """Handle incoming tool call from Gemini."""
        if not self.on_tool_call:
            return
        
        try:
            function_calls = tool_call_data.get("functionCalls", [])
            
            for call in function_calls:
                name = call.get("name", "")
                args = call.get("args", {})
                call_id = call.get("id", "")
                
                self.logger.info(f"Tool call received: {name}")
                
                # Execute tool
                result = await self._safe_callback(self.on_tool_call, name, args)
                
                # Send response back to Gemini
                await self._send_tool_response(call_id, result)
                
        except Exception as e:
            self.logger.error(f"Error handling tool call: {e}")
    
    async def _send_tool_response(self, call_id: str, result: dict):
        """Send tool execution result back to Gemini."""
        if not self.websocket:
            return
        
        message = {
            "tool_response": {
                "function_responses": [
                    {
                        "id": call_id,
                        "response": result
                    }
                ]
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            self.logger.debug(f"Tool response sent for {call_id}")
        except Exception as e:
            self.logger.error(f"Failed to send tool response: {e}")
    
    async def _safe_callback(self, callback, *args):
        """Safely invoke a callback, handling both sync and async."""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            self.logger.error(f"Callback error: {e}")
            return {"error": str(e)}
