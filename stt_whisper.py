"""Speech-to-text using local Whisper (via faster-whisper)."""

import logging
from pathlib import Path
from typing import Optional


class WhisperSTT:
    """Local speech-to-text using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """Initialize Whisper STT.
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
                       Use "base" for good speed/accuracy balance on RTX 4090
        """
        self.model_size = model_size
        self.logger = logging.getLogger('WhisperSTT')
        self._model = None
        self._available = False
        
        self._try_import()
    
    def _try_import(self):
        """Try to import faster_whisper."""
        try:
            from faster_whisper import WhisperModel
            self.logger.info(f"🎙️ Loading Whisper {self.model_size} model...")
            
            # Use GPU if available
            self._model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="float16"
            )
            self._available = True
            self.logger.info("✅ Whisper model loaded")
            
        except ImportError:
            self.logger.error("❌ faster-whisper not installed. Run: pip install faster-whisper")
            self._available = False
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Whisper is ready."""
        return self._available
    
    def transcribe(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            Transcribed text, or None if failed
        """
        if not self._available or self._model is None:
            self.logger.error("Whisper not available")
            return None
        
        try:
            self.logger.info("📝 Transcribing...")
            
            segments, info = self._model.transcribe(
                str(audio_path),
                beam_size=5,
                language="en"
            )
            
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            full_text = " ".join(text_parts).strip()
            
            if full_text:
                self.logger.info(f"📝 Heard: \"{full_text}\"")
                return full_text
            else:
                self.logger.info("📝 No speech detected")
                return None
                
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
