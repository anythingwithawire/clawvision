"""Capture audio from phone microphone via IP Webcam Pro."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import requests


class PhoneAudioCapture:
    """Capture audio from phone mic via IP Webcam Pro stream."""
    
    def __init__(self, base_url: str = "http://192.168.86.74:8080"):
        self.base_url = base_url
        self.audio_endpoint = "/audio.wav"
        self.logger = logging.getLogger('PhoneAudio')
    
    def capture_speech(self, duration_seconds: float = 5.0) -> Optional[Path]:
        """Record audio from phone microphone.
        
        Args:
            duration_seconds: How long to record
            
        Returns:
            Path to WAV file, or None if failed
        """
        url = f"{self.base_url}{self.audio_endpoint}"
        self.logger.info(f"🎤 Recording {duration_seconds}s from phone mic...")
        
        try:
            # Stream audio from phone
            response = requests.get(url, stream=True, timeout=duration_seconds + 5)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get audio stream: HTTP {response.status_code}")
                return None
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                wav_path = Path(f.name)
                
                # Read stream for specified duration
                bytes_read = 0
                chunk_size = 4096
                target_bytes = int(16000 * 2 * duration_seconds)  # 16kHz, 16-bit mono
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    bytes_read += len(chunk)
                    
                    if bytes_read >= target_bytes:
                        break
                
                self.logger.info(f"✅ Recorded {bytes_read} bytes ({duration_seconds}s)")
                return wav_path
                
        except requests.exceptions.Timeout:
            self.logger.error("Audio stream timeout")
        except Exception as e:
            self.logger.error(f"Audio capture error: {e}")
        
        return None
