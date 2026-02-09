#!/usr/bin/env python3
"""
ClawVision Voice - Vision + Voice AI using phone camera and microphone

Uses:
- Phone camera (IP Webcam Pro)
- Phone microphone (IP Webcam Pro audio stream)
- Local Whisper for speech-to-text
- Local vLLM (Qwen2-VL) for vision+text
- gTTS for text-to-speech
"""

import logging
import tempfile
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from gtts import gTTS

from audio_from_phone import PhoneAudioCapture
from local_vllm_client import LocalVLLMClient
from stt_whisper import WhisperSTT
from video_capture import VideoCapture, VideoConfig


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )


class TTSManager:
    """Text-to-speech using gTTS."""
    
    def __init__(self):
        self.logger = logging.getLogger('TTS')
    
    def speak(self, text: str):
        """Convert text to speech and play it."""
        self.logger.info(f"🔊 Speaking...")
        
        try:
            # Generate TTS
            tts = gTTS(text=text[:500], lang='en', slow=False)  # Limit length
            
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
            print(f"\n🤖 ClawVision: {text}\n")


def main():
    setup_logging()
    logger = logging.getLogger('ClawVision')
    
    print("\n🚀 ClawVision Voice")
    print("=" * 50)
    print("Phone camera + Phone mic + Local AI")
    print("Press Ctrl+C to exit\n")
    
    # Initialize components
    phone_url = "http://192.168.86.74:8080"
    
    # Phone audio capture
    logger.info("🎤 Initializing phone audio capture...")
    phone_audio = PhoneAudioCapture(phone_url)
    
    # Whisper STT
    logger.info("🎙️ Initializing Whisper (base model)...")
    whisper = WhisperSTT(model_size="base")
    if not whisper.is_available():
        logger.error("❌ Whisper not available. Install: pip install faster-whisper")
        print("\nInstall whisper dependencies:")
        print("  pip install faster-whisper")
        return
    
    # vLLM vision
    logger.info("🤖 Initializing vLLM...")
    vllm = LocalVLLMClient()
    if not vllm.test_connection():
        logger.error("❌ vLLM not running. Start with:")
        logger.error("  vllm serve Qwen/Qwen2-VL-7B-Instruct --gpu-memory-utilization 0.75")
        return
    
    # Video capture
    logger.info("📱 Connecting to phone camera...")
    video = VideoCapture(VideoConfig(
        url=f"{phone_url}/video",
        target_fps=0.5
    ))
    
    # TTS
    tts = TTSManager()
    
    print("\n✅ Ready!")
    print("Just start speaking to your phone - I'll detect your voice and respond")
    print("Press Ctrl+C to exit\n")
    
    system_prompt = """You are ClawVision, a helpful AI assistant with vision and hearing capabilities.
You can see through the user's phone camera and hear them speak.
Respond naturally to their questions about what you see and hear.
Be concise but helpful."""
    
    # Initial volume calibration
    print("🎤 Calibrating background noise... (stay quiet for 2 seconds)")
    time.sleep(2)
    print("✅ Listening for your voice...\n")
    
    while True:
        try:
            # Auto-trigger: Record from phone mic
            print("🎤 Listening...")
            audio_file = phone_audio.capture_speech(duration_seconds=6.0)
            
            if not audio_file:
                continue  # Keep listening
            
            # Transcribe
            transcribed = whisper.transcribe(audio_file)
            
            # Cleanup audio file
            audio_file.unlink(missing_ok=True)
            
            if not transcribed:
                continue  # No speech, keep listening
            
            print(f"🗣️ You said: \"{transcribed}\"")
            
            # Capture video frame
            print("📸 Looking...")
            frame_b64 = video.get_frame_base64()
            
            if not frame_b64:
                print("❌ Failed to capture image")
                continue
            
            import base64
            frame_bytes = base64.b64decode(frame_b64)
            
            # Generate response with vision
            print("🤖 Thinking...")
            start_time = time.time()
            
            response = vllm.generate_with_image(
                prompt=transcribed,
                image_bytes=frame_bytes,
                system_prompt=system_prompt
            )
            
            elapsed = time.time() - start_time
            
            print(f"\n🤖 ClawVision ({elapsed:.1f}s):")
            print(response)
            print()
            
            # Speak response
            tts.speak(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
