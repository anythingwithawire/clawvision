#!/usr/bin/env python3
"""
ClawVision - Linux-based VisionClaw Alternative (REST API Version)

Uses Android phone as camera, local microphone for audio,
sends to Gemini REST API for vision + voice interaction,
routes tool calls to OpenClaw gateway.

Usage:
    python main.py --config config.yaml
    python main.py --phone-url http://192.168.1.100:8080 --gemini-key YOUR_KEY
"""

import argparse
import asyncio
import io
import logging
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml

# Handle gTTS import gracefully
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from gemini_rest_client import GeminiRestClient
from openclaw_bridge import OpenClawBridge
from video_capture import VideoCapture, VideoConfig


@dataclass
class Config:
    """Application configuration."""
    # Gemini API
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    
    # Phone stream
    phone_url: str = "http://192.168.1.100:8080"
    video_endpoint: str = "/video"
    
    # OpenClaw gateway
    openclaw_host: str = "localhost"
    openclaw_port: int = 18789
    openclaw_token: Optional[str] = None
    
    # Audio settings
    local_mic_device: Optional[int] = None
    output_sample_rate: int = 24000
    input_sample_rate: int = 16000
    
    # Video settings
    jpeg_quality: int = 85
    
    # Behavior
    auto_start: bool = True
    debug: bool = False
    
    # Silence detection
    silence_threshold: float = 0.01  # Audio level threshold
    silence_duration: float = 1.5    # Seconds of silence to trigger processing
    min_speech_duration: float = 0.5  # Minimum speech before processing


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


class AudioRecorder:
    """Records audio from microphone with silence detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        self.logger = logging.getLogger('AudioRecorder')
        self._recording = False
        self._frames = []
        self._stream = None
        self._silence_start = None
        self._speech_start = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            self.logger.warning(f"Audio status: {status}")
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(indata**2))
        
        # Check if speaking
        is_speaking = rms > self.silence_threshold
        
        if is_speaking:
            if self._speech_start is None:
                self._speech_start = time.time()
                self.logger.debug("🎤 Speech detected")
            self._silence_start = None
            self._frames.append(indata.copy())
        else:
            if self._speech_start is not None:
                if self._silence_start is None:
                    self._silence_start = time.time()
                
                # Continue recording during brief silence
                self._frames.append(indata.copy())
    
    async def record_until_silence(self) -> Optional[bytes]:
        """
        Record audio until silence is detected.
        
        Returns:
            WAV audio data as bytes, or None if no speech detected
        """
        self.logger.info("🎤 Listening... (speak now)")
        
        self._frames = []
        self._speech_start = None
        self._silence_start = None
        self._recording = True
        
        # Start recording stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            callback=self._audio_callback
        )
        
        self._stream.start()
        
        try:
            while self._recording:
                await asyncio.sleep(0.05)
                
                # Check if we have enough silence to stop
                if self._speech_start and self._silence_start:
                    silence_elapsed = time.time() - self._silence_start
                    speech_duration = time.time() - self._speech_start
                    
                    if silence_elapsed > self.silence_duration:
                        if speech_duration > self.min_speech_duration:
                            self.logger.info(f"✅ Speech recorded ({speech_duration:.1f}s)")
                            break
                        else:
                            # Too short, reset
                            self.logger.debug("Speech too short, resetting")
                            self._frames = []
                            self._speech_start = None
                            self._silence_start = None
        
        finally:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._recording = False
        
        if not self._frames:
            return None
        
        # Concatenate frames
        audio_data = np.concatenate(self._frames, axis=0)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, self.sample_rate, format='WAV')
        buffer.seek(0)
        
        return buffer.getvalue()


class TTSManager:
    """Text-to-speech manager using gTTS."""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger('TTSManager')
    
    async def speak(self, text: str):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
        """
        if not GTTS_AVAILABLE:
            self.logger.warning("gTTS not available. Install with: pip install gtts")
            print(f"🤖 (TTS unavailable): {text}")
            return
        
        try:
            self.logger.info("🔊 Generating speech...")
            
            # Create TTS in executor to not block
            loop = asyncio.get_event_loop()
            
            def generate_and_play():
                # Generate TTS audio
                tts = gTTS(text=text, lang='en', slow=False)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_path = fp.name
                    tts.save(temp_path)
                
                try:
                    # Load and play
                    data, sr = sf.read(temp_path, dtype='float32')
                    sd.play(data, samplerate=sr, blocking=True)
                finally:
                    os.unlink(temp_path)
            
            await loop.run_in_executor(None, generate_and_play)
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            print(f"🤖: {text}")


class ClawVision:
    """Main application orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('ClawVision')
        
        self.video_capture: Optional[VideoCapture] = None
        self.audio_recorder: Optional[AudioRecorder] = None
        self.tts_manager: Optional[TTSManager] = None
        self.gemini_client: Optional[GeminiRestClient] = None
        self.openclaw_bridge: Optional[OpenClawBridge] = None
        
        self._shutdown_event = asyncio.Event()
        self._current_frame: Optional[bytes] = None
    
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
        video_config = VideoConfig(
            url=video_url,
            target_fps=0.5,  # Low FPS for REST API to save quota
            jpeg_quality=self.config.jpeg_quality
        )
        self.video_capture = VideoCapture(video_config)
        
        # Initialize audio recorder
        self.audio_recorder = AudioRecorder(
            sample_rate=self.config.input_sample_rate,
            silence_threshold=self.config.silence_threshold,
            silence_duration=self.config.silence_duration,
            min_speech_duration=self.config.min_speech_duration
        )
        
        # Initialize TTS manager
        self.tts_manager = TTSManager(sample_rate=self.config.output_sample_rate)
        
        # Initialize Gemini REST client
        self.gemini_client = GeminiRestClient(
            api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
            on_tool_call=self._handle_tool_call,
            on_text_response=self._handle_text_response
        )
        
        # Test API connection
        self.logger.info("🔌 Testing Gemini API connection...")
        if not self.gemini_client.test_connection():
            raise ConnectionError("Failed to connect to Gemini API. Check your API key.")
        
        self.logger.info("✅ All components initialized")
    
    async def _handle_tool_call(self, tool_name: str, params: dict) -> dict:
        """Route tool calls to OpenClaw gateway."""
        self.logger.info(f"🔧 Tool call: {tool_name}")
        
        if tool_name == "openclaw_execute" and self.openclaw_bridge:
            task = params.get("task", "")
            context = params.get("context", {})
            return self.openclaw_bridge.execute(task, context)
        
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}
    
    async def _handle_text_response(self, text: str):
        """Handle text response from Gemini."""
        self.logger.info(f"🤖 Gemini: {text[:100]}...")
    
    async def _video_loop(self):
        """Continuously capture video frames."""
        self.logger.info("📹 Starting video capture...")
        
        async def frame_handler(frame: bytes):
            self._current_frame = frame
        
        try:
            await self.video_capture.start(frame_handler)
            
            while not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Video loop error: {e}")
        finally:
            await self.video_capture.stop()
    
    async def _interaction_loop(self):
        """Main interaction loop: record audio -> send to Gemini -> speak response."""
        self.logger.info("🎤 Starting interaction loop...")
        self.logger.info("   Say 'Hey' or speak to interact!")
        
        await asyncio.sleep(2)  # Wait for video to warm up
        
        try:
            while not self._shutdown_event.is_set():
                # Record audio until silence
                audio_data = await self.audio_recorder.record_until_silence()
                
                if not audio_data:
                    continue
                
                if self._shutdown_event.is_set():
                    break
                
                # Get current frame
                frame = self._current_frame
                
                # Send to Gemini
                self.logger.info("🤔 Processing...")
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.gemini_client.process_frame_and_audio(
                        image_bytes=frame,
                        audio_bytes=audio_data
                    )
                )
                
                if self._shutdown_event.is_set():
                    break
                
                # Speak the response
                if response:
                    await self.tts_manager.speak(response)
                
                # Small pause before next recording
                await asyncio.sleep(0.5)
                
        except Exception as e:
            self.logger.error(f"Interaction loop error: {e}")
    
    async def run(self):
        """Main run loop."""
        await self.initialize()
        
        self.logger.info("✨ ClawVision is running!")
        self.logger.info("   Speak to interact. Press Ctrl+C to stop.")
        
        # Run video and interaction loops concurrently
        await asyncio.gather(
            self._video_loop(),
            self._interaction_loop(),
            self._shutdown_waiter(),
            return_exceptions=True
        )
        
        await self.shutdown()
    
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
        
        if self.video_capture:
            await self.video_capture.stop()
        
        if self.openclaw_bridge:
            self.openclaw_bridge.close()
        
        self.logger.info("👋 Goodbye!")


def create_sample_config(path: str):
    """Create a sample configuration file."""
    sample = """# ClawVision Configuration (REST API Version)

# Gemini API Key (required)
# Get yours at: https://aistudio.google.com/app/apikey
gemini_api_key: "YOUR_API_KEY_HERE"

# Gemini model to use
# Use "gemini-2.0-flash" for best results on free tier
gemini_model: "gemini-2.0-flash"

# Phone IP Webcam URL (check the app for the actual URL)
phone_url: "http://192.168.1.100:8080"

# Video stream endpoint (usually /video or /videofeed)
video_endpoint: "/video"

# OpenClaw Gateway settings
openclaw_host: "localhost"
openclaw_port: 18789
openclaw_token: null  # Set if your gateway requires authentication

# Local microphone device index
# Use 'python -m sounddevice' to list devices
local_mic_device: null

# Audio sample rates
input_sample_rate: 16000   # For sending to Gemini
output_sample_rate: 24000  # For TTS output

# Video settings
jpeg_quality: 85      # JPEG compression quality (0-100)

# Silence detection settings
silence_threshold: 0.01   # Audio level threshold (0.0-1.0)
silence_duration: 1.5     # Seconds of silence to end recording
min_speech_duration: 0.5  # Minimum speech before processing

# Behavior
auto_start: true
debug: false
"""
    with open(path, 'w') as f:
        f.write(sample)
    print(f"Sample config created: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="ClawVision - Vision + Voice AI Assistant (REST API Version)"
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
    
    # Check gTTS availability
    if not GTTS_AVAILABLE:
        logging.warning("gTTS not installed. Text responses will be printed only.")
        logging.warning("Install with: pip install gtts")
    
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
