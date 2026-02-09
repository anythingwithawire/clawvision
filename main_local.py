#!/usr/bin/env python3
"""
ClawVision Local - Vision + Voice AI using local vLLM

Uses phone as camera (IP Webcam), local microphone,
and local Qwen2-VL model via vLLM for completely offline operation.
"""

import argparse
import asyncio
import base64
import io
import logging
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from gtts import gTTS

from local_vllm_client import LocalVLLMClient
from video_capture import VideoCapture, VideoConfig


@dataclass
class Config:
    """Application configuration."""
    # Phone stream
    phone_url: str = "http://192.168.86.74:8080"
    video_endpoint: str = "/video"
    
    # vLLM settings
    vllm_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Audio settings
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    silence_duration: float = 2.0  # seconds of silence to end recording
    max_record_seconds: float = 30.0
    
    # Video settings
    fps: float = 0.5  # frames per minute (one every 2 seconds)
    
    # System prompt
    system_prompt: str = """You are ClawVision, a helpful AI assistant with vision capabilities.
You can see through the user's phone camera and respond to their questions about what you see.
Be concise but helpful. Describe what you see when asked."""


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )


class AudioRecorder:
    """Record audio from microphone with silence detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('AudioRecorder')
        self.recording = False
        self.audio_buffer = []
        
    def record_until_silence(self) -> Optional[np.ndarray]:
        """Record audio until silence detected or max duration.
        
        Returns:
            Recorded audio as numpy array, or None if no speech detected
        """
        self.logger.info("🎤 Recording... (speak now)")
        self.recording = True
        self.audio_buffer = []
        
        silence_start = None
        speech_detected = False
        start_time = time.time()
        
        def callback(indata, frames, time_info, status):
            if status:
                self.logger.warning(f"Audio status: {status}")
            if self.recording:
                self.audio_buffer.append(indata.copy())
        
        # Start recording
        stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=callback
        )
        
        with stream:
            while self.recording:
                time.sleep(0.1)
                
                # Check max duration
                elapsed = time.time() - start_time
                if elapsed > self.config.max_record_seconds:
                    self.logger.info("⏱️ Max recording duration reached")
                    self.recording = False
                    break
                
                # Check for silence
                if len(self.audio_buffer) > 10:  # At least 1 second of audio
                    recent_audio = np.concatenate(self.audio_buffer[-10:])
                    volume = np.sqrt(np.mean(recent_audio**2))
                    
                    if volume > self.config.silence_threshold:
                        speech_detected = True
                        silence_start = None
                    elif speech_detected:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.config.silence_duration:
                            self.logger.info("🔇 Silence detected, stopping recording")
                            self.recording = False
                            break
        
        if not speech_detected:
            self.logger.info("🚫 No speech detected")
            return None
        
        # Concatenate audio
        if self.audio_buffer:
            audio = np.concatenate(self.audio_buffer)
            duration = len(audio) / self.config.sample_rate
            self.logger.info(f"✅ Recorded {duration:.1f}s of audio")
            return audio
        
        return None
    
    def save_to_file(self, audio: np.ndarray, filepath: str):
        """Save audio to WAV file."""
        sf.write(filepath, audio, self.config.sample_rate)


class TTSManager:
    """Text-to-speech using gTTS."""
    
    def __init__(self):
        self.logger = logging.getLogger('TTS')
    
    def speak(self, text: str):
        """Convert text to speech and play it."""
        self.logger.info(f"🔊 Speaking: {text[:80]}...")
        
        try:
            # Generate TTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
                tts.save(temp_path)
            
            # Play audio
            data, samplerate = sf.read(temp_path)
            sd.play(data, samplerate)
            sd.wait()
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            # Fallback: just print
            print(f"\n🤖 ClawVision: {text}\n")


class ClawVisionLocal:
    """Main application using local vLLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('ClawVision')
        
        self.vllm = LocalVLLMClient(
            base_url=config.vllm_url,
            model=config.vllm_model
        )
        self.audio_recorder = AudioRecorder(config)
        self.tts = TTSManager()
        self.video_capture: Optional[VideoCapture] = None
        
        self._shutdown = False
    
    async def initialize(self):
        """Initialize components."""
        self.logger.info("🚀 Initializing ClawVision Local...")
        
        # Test vLLM connection
        if not self.vllm.test_connection():
            raise ConnectionError("Cannot connect to vLLM. Is it running?")
        
        # Initialize video capture
        video_url = f"{self.config.phone_url}{self.config.video_endpoint}"
        video_config = VideoConfig(
            url=video_url,
            target_fps=self.config.fps,
            jpeg_quality=85
        )
        self.video_capture = VideoCapture(video_config)
        
        self.logger.info("✅ Ready! Press Ctrl+C to exit.")
        self.logger.info("🎤 Say something to start...")
    
    async def run(self):
        """Main loop."""
        try:
            while not self._shutdown:
                # Record audio
                audio = self.audio_recorder.record_until_silence()
                
                if audio is None:
                    continue
                
                # Get latest video frame
                frame_b64 = self.video_capture.get_frame_base64()
                frame_bytes = base64.b64decode(frame_b64) if frame_b64 else None
                
                # Transcribe audio (simple approach: just use as prompt text)
                # For now, we'll skip STT and assume the user wants us to describe
                prompt = "What do you see? Describe the scene briefly."
                
                # Generate response with vision
                self.logger.info("🤖 Thinking...")
                response = self.vllm.generate_with_image(
                    prompt=prompt,
                    image_bytes=frame_bytes,
                    system_prompt=self.config.system_prompt
                )
                
                # Speak response
                self.tts.speak(response)
                
                self.logger.info("🎤 Listening...")
                
        except KeyboardInterrupt:
            self.logger.info("👋 Shutting down...")
    
    def shutdown(self):
        """Graceful shutdown."""
        self._shutdown = True
        if self.video_capture:
            asyncio.create_task(self.video_capture.stop())


def main():
    parser = argparse.ArgumentParser(description="ClawVision Local")
    parser.add_argument("--phone-url", default="http://192.168.86.74:8080",
                        help="IP Webcam URL")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM API URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    config = Config(
        phone_url=args.phone_url,
        vllm_url=args.vllm_url
    )
    
    app = ClawVisionLocal(config)
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.shutdown)
    
    try:
        loop.run_until_complete(app.initialize())
        loop.run_until_complete(app.run())
    except ConnectionError as e:
        logging.error(f"Startup failed: {e}")
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
