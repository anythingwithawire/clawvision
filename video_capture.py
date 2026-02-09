"""Video capture from IP webcam (phone camera)."""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class VideoConfig:
    """Video capture configuration."""
    url: str  # Full URL to MJPEG stream (e.g., http://192.168.1.42:8080/video)
    target_fps: float = 1.0  # Frames per second to capture
    jpeg_quality: int = 85  # JPEG encoding quality (0-100)
    resize_width: Optional[int] = 640  # Resize to this width (None = keep original)
    resize_height: Optional[int] = None  # Resize height (None = auto maintain aspect)


class VideoCapture:
    """Capture video frames from IP webcam stream."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.logger = logging.getLogger('VideoCapture')
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_callback: Optional[Callable[[bytes], None]] = None
        self._task: Optional[asyncio.Task] = None
        
        # Frame timing
        self._frame_interval = 1.0 / config.target_fps
        self._last_frame_time = 0.0
    
    async def start(self, frame_callback: Callable[[bytes], None]) -> bool:
        """Start video capture.
        
        Args:
            frame_callback: Called with JPEG-encoded frame bytes
            
        Returns:
            True if started successfully
        """
        self._frame_callback = frame_callback
        
        # OpenCV VideoCapture in async context
        loop = asyncio.get_event_loop()
        self._cap = await loop.run_in_executor(
            None, 
            lambda: cv2.VideoCapture(self.config.url)
        )
        
        if not self._cap.isOpened():
            self.logger.error(f"Failed to open video stream: {self.config.url}")
            return False
        
        self.logger.info(f"📹 Video stream opened: {self.config.url}")
        
        # Get stream properties
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"   Resolution: {width}x{height} @ {fps:.1f}fps")
        self.logger.info(f"   Capture rate: {self.config.target_fps}fps")
        
        self._running = True
        self._task = asyncio.create_task(self._capture_loop())
        
        return True
    
    async def stop(self):
        """Stop video capture."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self._cap:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._cap.release)
            self._cap = None
        
        self.logger.info("📹 Video capture stopped")
    
    async def _capture_loop(self):
        """Main capture loop."""
        self.logger.info("🎬 Video capture loop started")
        
        while self._running:
            try:
                # Capture frame
                frame = await self._capture_frame()
                
                if frame is not None and self._frame_callback:
                    # Encode and send
                    jpeg_bytes = self._encode_frame(frame)
                    if jpeg_bytes:
                        self._frame_callback(jpeg_bytes)
                
                # Maintain target FPS
                await asyncio.sleep(self._frame_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                await asyncio.sleep(1.0)  # Back off on error
        
        self.logger.info("🎬 Video capture loop ended")
    
    async def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the stream.
        
        Returns:
            Frame as numpy array, or None if failed
        """
        if not self._cap:
            return None
        
        loop = asyncio.get_event_loop()
        
        ret, frame = await loop.run_in_executor(
            None,
            self._cap.read
        )
        
        if not ret or frame is None:
            return None
        
        # Resize if configured
        if self.config.resize_width or self.config.resize_height:
            frame = self._resize_frame(frame)
        
        return frame
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        
        if self.config.resize_width and self.config.resize_height:
            # Exact size specified
            new_size = (self.config.resize_width, self.config.resize_height)
        elif self.config.resize_width:
            # Width specified, calculate height
            ratio = self.config.resize_width / width
            new_height = int(height * ratio)
            new_size = (self.config.resize_width, new_height)
        elif self.config.resize_height:
            # Height specified, calculate width
            ratio = self.config.resize_height / height
            new_width = int(width * ratio)
            new_size = (new_width, self.config.resize_height)
        else:
            return frame
        
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    
    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame to JPEG bytes.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            JPEG-encoded bytes
        """
        try:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Encode to JPEG
            buffer = io.BytesIO()
            pil_image.save(
                buffer, 
                format='JPEG', 
                quality=self.config.jpeg_quality,
                optimize=True
            )
            
            return buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Frame encoding error: {e}")
            return None
    
    def get_frame_base64(self) -> Optional[str]:
        """Get current frame as base64-encoded JPEG.
        
        Returns:
            Base64-encoded JPEG string
        """
        # Initialize capture if not already done
        if not self._cap:
            self._cap = cv2.VideoCapture(self.config.url)
            if not self._cap.isOpened():
                self.logger.error(f"Failed to open video stream: {self.config.url}")
                return None
            self.logger.info(f"📹 Video stream opened: {self.config.url}")
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        if self.config.resize_width or self.config.resize_height:
            frame = self._resize_frame(frame)
        
        jpeg_bytes = self._encode_frame(frame)
        if jpeg_bytes:
            return base64.b64encode(jpeg_bytes).decode('utf-8')
        
        return None
