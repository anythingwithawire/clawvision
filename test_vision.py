#!/usr/bin/env python3
"""Quick test of local vLLM vision without audio input."""

import base64
import sys
import time

from local_vllm_client import LocalVLLMClient
from video_capture import VideoCapture, VideoConfig


def main():
    print("🚀 ClawVision Quick Test (no audio required)")
    print("=" * 50)
    
    # Connect to vLLM
    vllm = LocalVLLMClient()
    if not vllm.test_connection():
        print("❌ Cannot connect to vLLM on localhost:8000")
        print("   Make sure vLLM is running: vllm serve Qwen/Qwen2-VL-7B-Instruct")
        sys.exit(1)
    
    # Connect to phone camera
    phone_url = "http://192.168.86.74:8080"
    video_config = VideoConfig(
        url=f"{phone_url}/video",
        target_fps=0.5,
        jpeg_quality=85
    )
    
    print(f"📱 Connecting to phone camera at {phone_url}...")
    video = VideoCapture(video_config)
    
    print("\n✅ Ready! Type your questions about what the camera sees.")
    print("   Type 'quit' to exit.\n")
    
    system_prompt = """You are ClawVision, a helpful AI assistant with vision capabilities.
Describe what you see when asked. Be concise but helpful."""
    
    while True:
        # Get user input
        prompt = input("🎤 You: ").strip()
        
        if prompt.lower() in ('quit', 'exit', 'q'):
            print("👋 Goodbye!")
            break
        
        if not prompt:
            prompt = "What do you see? Describe the scene."
        
        # Capture frame
        print("📸 Capturing image...")
        frame_b64 = video.get_frame_base64()
        
        if not frame_b64:
            print("❌ Could not capture image from camera")
            continue
        
        frame_bytes = base64.b64decode(frame_b64)
        print(f"   Image captured: {len(frame_bytes)} bytes")
        
        # Generate response
        print("🤖 Thinking...")
        start_time = time.time()
        
        response = vllm.generate_with_image(
            prompt=prompt,
            image_bytes=frame_bytes,
            system_prompt=system_prompt
        )
        
        elapsed = time.time() - start_time
        print(f"\n🤖 ClawVision ({elapsed:.1f}s): {response}\n")


if __name__ == "__main__":
    main()
