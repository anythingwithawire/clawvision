# ClawVision - Linux-based VisionClaw Alternative

Use your Android phone as a camera and microphone for Gemini Live API, with tool execution through OpenClaw gateway.

## Architecture

```
┌─────────────┐     HTTP/RTSP      ┌──────────────┐     WebSocket      ┌─────────────┐
│ Android     │ ───────────────────│ ClawVision   │ ──────────────────│ Gemini Live │
│ IP Webcam   │  (video + audio)   │   (Linux)    │  (vision + voice) │    API      │
│    App      │                    │              │                   │             │
└─────────────┘                    └──────────────┘                   └─────────────┘
                                          │
                                          │ HTTP
                                          ▼
                                   ┌──────────────┐
                                   │ OpenClaw     │
                                   │  Gateway     │
                                   └──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Clone/navigate to the project
cd ~/.openclaw/workspace/clawvision

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install system dependencies for audio
sudo apt-get install libportaudio2  # Ubuntu/Debian
# Or: sudo dnf install portaudio     # Fedora
```

### 2. Get a Gemini API Key

1. Visit https://aistudio.google.com/app/apikey
2. Create a new API key
3. Copy it for use in configuration

### 3. Set Up IP Webcam on Your Phone

1. Install **IP Webcam** app from Play Store (or similar app)
2. Connect your phone to the same WiFi as your computer
3. Open the app and tap "Start server"
4. Note the IP address shown (e.g., `http://192.168.1.100:8080`)

### 4. Configure ClawVision

Create a configuration file:

```bash
python main.py --create-config config.yaml
```

Edit `config.yaml` with your settings:

```yaml
gemini_api_key: "YOUR_ACTUAL_API_KEY_HERE"
phone_url: "http://192.168.1.100:8080"  # Your phone's IP from step 3
openclaw_host: "localhost"
openclaw_port: 18789
```

### 5. Run ClawVision

```bash
# Using config file
python main.py --config config.yaml

# Or using command line arguments
python main.py \
    --gemini-key YOUR_API_KEY \
    --phone-url http://192.168.1.100:8080

# Enable debug logging
python main.py --config config.yaml --debug
```

## Usage

Once running:

1. **Speak to Gemini** - Your voice is captured from the phone (or local mic)
2. **Show Gemini things** - Point your phone camera at objects, text, etc.
3. **Ask for actions** - Gemini can execute OpenClaw tools:
   - "Take a screenshot and save it"
   - "Send a message to John saying I'll be late"
   - "Search the web for Python async tutorials"

### Example Commands

- "What do you see?" - Gemini describes the camera view
- "Can you read this text?" - OCR through vision
- "Take a screenshot of my desktop" - Executes screenshot tool
- "Turn off the lights" - If you have home automation tools

## Configuration Options

### `config.yaml` Reference

```yaml
# Required
gemini_api_key: "your-key-here"

# Phone connection
phone_url: "http://192.168.1.100:8080"
video_endpoint: "/video"        # MJPEG stream endpoint
audio_endpoint: "/audio.wav"    # Audio stream endpoint

# OpenClaw gateway
openclaw_host: "localhost"
openclaw_port: 18789
openclaw_token: null            # Set if gateway requires auth

# Audio settings
audio_input_source: "phone"     # "phone" or "local"
local_mic_device: null          # Device index for local mic
input_sample_rate: 16000        # Gemini input rate
output_sample_rate: 24000       # Gemini output rate

# Video settings
fps_target: 1.0                 # Frames per second to Gemini
jpeg_quality: 85                # Image quality (0-100)

# Behavior
debug: false                    # Enable debug logging
```

### Environment Variables

```bash
export GEMINI_API_KEY="your-key"
export PHONE_URL="http://192.168.1.100:8080"
python main.py
```

## Phone App Options

### Recommended: IP Webcam (Free)
- **Pros**: Simple, reliable, supports both video and audio
- **URL**: `http://phone-ip:8080/video` for video
- **Audio**: Enable "Audio mode" in settings

### Alternatives

1. **DroidCam**
   - Acts as a webcam (V4L2 loopback required)
   - Requires additional setup on Linux

2. **Iriun Webcam**
   - USB and WiFi support
   - Higher quality, but requires driver install

3. **EpocCam**
   - Professional option with good quality

## Troubleshooting

### "Cannot connect to phone"
- Ensure phone and computer are on same WiFi network
- Check firewall rules (port 8080)
- Verify the IP address in IP Webcam app

### "No audio from phone"
- In IP Webcam, go to Settings → Audio → Enable audio
- Some phones require microphone permission
- Try `audio_input_source: "local"` to use computer mic instead

### "Gemini not responding"
- Check your API key is valid
- Verify internet connection
- Enable `--debug` for detailed logs

### "Tool execution failed"
- Ensure OpenClaw gateway is running (`openclaw gateway status`)
- Check gateway host/port configuration
- Verify tools are installed and available

### Audio device issues
List available audio devices:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Advanced Usage

### Custom Tool Mappings

Edit `openclaw_bridge.py` to add tool name mappings:

```python
mappings = {
    "my_custom_tool": "actual_openclaw_tool",
    # ...
}
```

### Using Local Microphone

```yaml
audio_input_source: "local"
local_mic_device: 0  # Use device index from sounddevice list
```

### Headless/Server Mode

For running without local audio playback:

```python
# In main.py, modify to disable audio output
# Or route audio to a file/http endpoint
```

## Development

### Project Structure

```
clawvision/
├── main.py                 # Entry point
├── gemini_live_client.py   # Gemini WebSocket client
├── audio_manager.py        # Audio I/O
├── video_capture.py        # Video capture
├── openclaw_bridge.py      # OpenClaw gateway client
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Adding New Features

1. **New Tool Handlers**: Extend `openclaw_bridge.py`
2. **Custom Audio Processing**: Modify `audio_manager.py`
3. **Video Effects**: Add processing in `video_capture.py`
4. **UI**: Consider adding a web interface using the snapshot feature

## License

MIT License - Feel free to modify and distribute.

## Contributing

Issues and PRs welcome! Focus areas:
- Better audio resampling
- Lower latency video streaming
- More robust reconnection handling
- Support for more phone apps

## Credits

- Google Gemini Live API
- OpenClaw project
- IP Webcam developers
