#!/usr/bin/env python3
"""
ClawVision - Linux-based VisionClaw Alternative

Uses Android phone as camera/mic via IP Webcam app,
streams to Gemini Live API for vision + voice interaction,
routes tool calls to OpenClaw gateway.

Usage:
    python main.py --config config.yaml
    python main.py --phone-url http://192.168.1.100:8080 --gemini-key YOUR_KEY
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from audio_manager import AudioManager
from gemini_live_client import GeminiLiveClient
from openclaw_bridge import OpenClawBridge
from video_capture import VideoCapture


@dataclass
class Config:
    """Application configuration."""
    # Gemini API
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash-exp"
    
    # Phone stream
    phone_url: str = "http://192.168.1.100:8080"
    video_endpoint: str = "/video"
    audio_endpoint: str = "/audio.wav"
    
    # OpenClaw gateway
    openclaw_host: str = "localhost"
    openclaw_port: int = 18789
    openclaw_token: Optional[str] = None
    
    # Audio settings
    audio_input_source: str = "phone"  # "phone" or "local"
    local_mic_device: Optional[int] = None
    output_sample_rate: int = 24000
    input_sample_rate: int = 16000
    
    # Video settings
    fps_target: float = 1.0  # frames per second to send to Gemini
    jpeg_quality: int = 85
    
    # Behavior
    auto_start: bool = True
    debug: bool = False


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )


class ClawVision:
    """Main application orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('ClawVision')
        
        self.video_capture: Optional[VideoCapture] = None
        self.audio_manager: Optional[AudioManager] = None
        self.gemini_client: Optional[GeminiLiveClient] = None
        self.openclaw_bridge: Optional[OpenClawBridge] = None
        
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
    
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("🚀 Initializing ClawVision...")
        
        # Initialize OpenClaw bridge
        self.openclaw_bridge = OpenClawBridge(
            host=self.config.openclaw_host,
            port=self.config.openclaw_port,
            token=self.config.openclaw_token
        )
        
        # Initialize video capture from phone
        video_url = f"{self.config.phone_url}{self.config.video_endpoint}"
        self.video_capture = VideoCapture(
            url=video_url,
            fps_target=self.config.fps_target,
            jpeg_quality=self.config.jpeg_quality
        )
        
        # Initialize audio manager
        audio_url = None
        if self.config.audio_input_source == "phone":
            audio_url = f"{self.config.phone_url}{self.config.audio_endpoint}"
        
        self.audio_manager = AudioManager(
            input_source=self.config.audio_input_source,
            phone_audio_url=audio_url,
            local_device=self.config.local_mic_device,
            input_sample_rate=self.config.input_sample_rate,
            output_sample_rate=self.config.output_sample_rate
        )
        
        # Initialize Gemini Live client
        self.gemini_client = GeminiLiveClient(
            api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
            on_tool_call=self._handle_tool_call,
            on_audio_response=self._handle_audio_response,
            on_text_response=self._handle_text_response
        )
        
        self.logger.info("✅ All components initialized")
    
    async def _handle_tool_call(self, tool_name: str, params: dict) -> dict:
        """Route tool calls to OpenClaw gateway."""
        self.logger.info(f"🔧 Tool call: {tool_name}")
        if self.openclaw_bridge:
            return await self.openclaw_bridge.execute_tool(tool_name, params)
        return {"error": "OpenClaw bridge not available"}
    
    async def _handle_audio_response(self, audio_data: bytes):
        """Play audio response from Gemini."""
        if self.audio_manager:
            await self.audio_manager.play_audio(audio_data)
    
    async def _handle_text_response(self, text: str):
        """Handle text response from Gemini."""
        self.logger.info(f"🤖 Gemini: {text}")
    
    async def run(self):
        """Main run loop."""
        await self.initialize()
        
        # Start components
        self.logger.info("🎬 Starting capture...")
        
        # Connect to Gemini
        await self.gemini_client.connect()
        
        # Start audio capture and playback
        await self.audio_manager.start()
        
        # Create tasks
        self._tasks = [
            asyncio.create_task(self._video_loop()),
            asyncio.create_task(self._audio_loop()),
            asyncio.create_task(self._gemini_loop()),
            asyncio.create_task(self._shutdown_waiter())
        ]
        
        self.logger.info("✨ ClawVision is running! Speak to interact.")
        self.logger.info("   Press Ctrl+C to stop")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cleanup
        await self.shutdown()
    
    async def _video_loop(self):
        """Continuously capture and send video frames."""
        try:
            async for frame in self.video_capture.stream_frames():
                if self._shutdown_event.is_set():
                    break
                
                # Send frame to Gemini
                if self.gemini_client and self.gemini_client.is_connected:
                    await self.gemini_client.send_frame(frame)
        except Exception as e:
            self.logger.error(f"Video loop error: {e}")
    
    async def _audio_loop(self):
        """Continuously capture and send audio."""
        try:
            async for audio_chunk in self.audio_manager.stream_input():
                if self._shutdown_event.is_set():
                    break
                
                # Send audio to Gemini
                if self.gemini_client and self.gemini_client.is_connected:
                    await self.gemini_client.send_audio(audio_chunk)
        except Exception as e:
            self.logger.error(f"Audio loop error: {e}")
    
    async def _gemini_loop(self):
        """Handle Gemini responses."""
        try:
            await self.gemini_client.receive_loop()
        except Exception as e:
            self.logger.error(f"Gemini loop error: {e}")
    
    async def _shutdown_waiter(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
    
    def request_shutdown(self):
        """Request graceful shutdown."""
        self.logger.info("🛑 Shutdown requested...")
        self._shutdown_event.set()
    
    async def shutdown(self):
        """Cleanup resources."""
        self.logger.info("🧹 Cleaning up...")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Shutdown components
        if self.gemini_client:
            await self.gemini_client.disconnect()
        
        if self.audio_manager:
            await self.audio_manager.stop()
        
        if self.video_capture:
            await self.video_capture.close()
        
        self.logger.info("👋 Goodbye!")


def create_sample_config(path: str):
    """Create a sample configuration file."""
    sample = """# ClawVision Configuration

# Gemini API Key (required)
# Get yours at: https://aistudio.google.com/app/apikey
gemini_api_key: "YOUR_API_KEY_HERE"

# Gemini model to use
gemini_model: "gemini-2.0-flash-exp"

# Phone IP Webcam URL (check the app for the actual URL)
phone_url: "http://192.168.1.100:8080"

# Video stream endpoint (usually /video or /videofeed)
video_endpoint: "/video"

# Audio stream endpoint (usually /audio.wav)
audio_endpoint: "/audio.wav"

# OpenClaw Gateway settings
openclaw_host: "localhost"
openclaw_port: 18789
openclaw_token: null  # Set if your gateway requires authentication

# Audio input source: "phone" or "local"
audio_input_source: "phone"

# Local microphone device index (if using local audio)
# Use 'python -m sounddevice' to list devices
local_mic_device: null

# Audio sample rates
input_sample_rate: 16000   # For sending to Gemini
output_sample_rate: 24000  # For receiving from Gemini

# Video settings
fps_target: 1.0       # Frames per second to send to Gemini
jpeg_quality: 85      # JPEG compression quality (0-100)

# Behavior
auto_start: true
debug: false
"""
    with open(path, 'w') as f:
        f.write(sample)
    print(f"Sample config created: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="ClawVision - Vision + Voice AI Assistant"
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--phone-url',
        help='Phone IP Webcam URL (e.g., http://192.168.1.100:8080)'
    )
    parser.add_argument(
        '--gemini-key',
        help='Gemini API key'
    )
    parser.add_argument(
        '--create-config',
        metavar='PATH',
        help='Create a sample configuration file and exit'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config(args.create_config)
        return
    
    # Load or build configuration
    config = None
    
    if args.config:
        config = load_config(args.config)
    else:
        # Build from environment and args
        gemini_key = args.gemini_key or os.environ.get('GEMINI_API_KEY')
        phone_url = args.phone_url or os.environ.get('PHONE_URL', 'http://192.168.1.100:8080')
        
        if not gemini_key:
            print("Error: Gemini API key required. Use --gemini-key or GEMINI_API_KEY env var")
            print(f"\nOr create a config file: python main.py --create-config config.yaml")
            sys.exit(1)
        
        config = Config(
            gemini_api_key=gemini_key,
            phone_url=phone_url,
            debug=args.debug
        )
    
    # Override debug flag
    if args.debug:
        config.debug = True
    
    setup_logging(config.debug)
    
    # Create and run application
    app = ClawVision(config)
    
    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.request_shutdown)
    
    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == '__main__':
    main()
