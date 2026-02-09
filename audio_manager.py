"""
Audio Manager

Handles audio capture from phone stream or local microphone,
and audio playback for Gemini responses.
"""

import asyncio
import io
import logging
from typing import AsyncGenerator, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf


class AudioManager:
    """
    Manages audio input and output for ClawVision.
    
    Supports:
    - Phone audio stream (via HTTP)
    - Local microphone input
    - Audio playback to default output device
    """
    
    def __init__(
        self,
        input_source: str = "phone",  # "phone" or "local"
        phone_audio_url: Optional[str] = None,
        local_device: Optional[int] = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        chunk_duration: float = 0.1,  # 100ms chunks
        channels: int = 1
    ):
        self.input_source = input_source
        self.phone_audio_url = phone_audio_url
        self.local_device = local_device
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels
        
        self.logger = logging.getLogger('AudioManager')
        
        self._input_stream = None
        self._output_stream = None
        self._running = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        
        # Calculate chunk size
        self.chunk_samples = int(self.input_sample_rate * self.chunk_duration)
    
    async def start(self):
        """Start audio capture."""
        self._running = True
        
        if self.input_source == "local":
            self._start_local_capture()
        
        self.logger.info(f"🎤 Audio started (source: {self.input_source})")
    
    async def stop(self):
        """Stop audio capture and playback."""
        self._running = False
        
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        
        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
        
        self.logger.info("🎤 Audio stopped")
    
    def _start_local_capture(self):
        """Start capturing from local microphone."""
        def callback(indata, frames, time_info, status):
            if status:
                self.logger.warning(f"Audio status: {status}")
            # Convert to bytes and put in queue
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            asyncio.create_task(self._audio_queue.put(audio_bytes))
        
        self._input_stream = sd.InputStream(
            device=self.local_device,
            channels=self.channels,
            samplerate=self.input_sample_rate,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=callback
        )
        self._input_stream.start()
    
    async def stream_input(self) -> AsyncGenerator[bytes, None]:
        """
        Stream audio input continuously.
        
        Yields:
            Audio data as PCM 16-bit signed bytes
        """
        if self.input_source == "phone":
            async for chunk in self._stream_phone_audio():
                yield chunk
        else:
            # Local microphone
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=1.0
                    )
                    yield chunk
                except asyncio.TimeoutError:
                    continue
    
    async def _stream_phone_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Stream audio from phone's HTTP endpoint.
        
        IP Webcam typically serves audio as WAV stream.
        """
        if not self.phone_audio_url:
            self.logger.error("Phone audio URL not set")
            return
        
        self.logger.info(f"Streaming audio from {self.phone_audio_url}")
        
        # We'll read the WAV stream in chunks
        loop = asyncio.get_event_loop()
        
        try:
            # Run blocking HTTP request in executor
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    self.phone_audio_url,
                    stream=True,
                    timeout=30
                )
            )
            response.raise_for_status()
            
            # Buffer for accumulating audio data
            buffer = bytearray()
            target_bytes = self.chunk_samples * 2  # 16-bit = 2 bytes per sample
            
            # Read stream in chunks
            for chunk in response.iter_content(chunk_size=4096):
                if not self._running:
                    break
                
                buffer.extend(chunk)
                
                # Process complete chunks
                while len(buffer) >= target_bytes:
                    # Extract one chunk
                    audio_chunk = bytes(buffer[:target_bytes])
                    buffer = buffer[target_bytes:]
                    
                    # Convert to target sample rate if needed
                    audio_chunk = await self._resample_if_needed(
                        audio_chunk,
                        8000  # IP Webcam usually streams at 8kHz
                    )
                    
                    yield audio_chunk
                    
        except requests.RequestException as e:
            self.logger.error(f"Phone audio stream error: {e}")
            # Fall back to silence
            while self._running:
                await asyncio.sleep(0.1)
                yield b'\x00' * target_bytes
    
    async def _resample_if_needed(self, audio_data: bytes, source_rate: int) -> bytes:
        """
        Resample audio data to target sample rate.
        
        Simple linear resampling for demonstration.
        For production, consider using librosa or scipy.signal.resample.
        """
        if source_rate == self.input_sample_rate:
            return audio_data
        
        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate resampling ratio
        ratio = self.input_sample_rate / source_rate
        new_length = int(len(samples) * ratio)
        
        # Simple linear interpolation
        indices = np.linspace(0, len(samples) - 1, new_length)
        indices_floor = indices.astype(np.int32)
        indices_ceil = np.minimum(indices_floor + 1, len(samples) - 1)
        fractions = indices - indices_floor
        
        resampled = (
            samples[indices_floor] * (1 - fractions) +
            samples[indices_ceil] * fractions
        ).astype(np.int16)
        
        return resampled.tobytes()
    
    async def play_audio(self, audio_data: bytes):
        """
        Play audio data through default output device.
        
        Args:
            audio_data: PCM 16-bit signed audio data at 24kHz (Gemini output)
        """
        try:
            # Convert bytes to numpy array
            samples = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to float32 (-1.0 to 1.0)
            audio_float = samples.astype(np.float32) / 32767.0
            
            # Run blocking audio playback in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: sd.play(
                    audio_float,
                    samplerate=self.output_sample_rate,
                    blocking=True
                )
            )
            
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
    
    async def play_audio_buffer(self, audio_buffer: io.BytesIO, format: str = "wav"):
        """
        Play audio from a buffer (supports WAV, etc.).
        
        Args:
            audio_buffer: BytesIO containing audio data
            format: Audio format ("wav", "raw", etc.)
        """
        try:
            # Read audio file
            audio_buffer.seek(0)
            data, samplerate = sf.read(audio_buffer, dtype='float32')
            
            # Play audio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: sd.play(data, samplerate=samplerate, blocking=True)
            )
            
        except Exception as e:
            self.logger.error(f"Buffer playback error: {e}")


def list_audio_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())


if __name__ == "__main__":
    list_audio_devices()
