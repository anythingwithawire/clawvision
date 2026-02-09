# ClawVision (REST API Version)

A Linux-based Vision + Voice AI assistant using the Gemini REST API (for free tier users).

Captures voice from your microphone, takes pictures from your phone's camera, sends both to Gemini for understanding, and speaks the response.

## What's New in This Version

- **REST API Instead of WebSocket**: Works on Gemini free tier (no Live API needed)
- **Local Microphone**: Uses local mic instead of phone audio stream
- **TTS Integration**: Uses Google TTS (gTTS) for voice responses
- **Simpler Architecture**: No WebSocket complexity, just HTTP requests

## Requirements

- Python 3.8+
- Gemini API key (free tier works!)
- Android phone with IP Webcam app
- Microphone
- Speakers/headphones

## Setup

### 1. Install Dependencies

```bash
cd ~/.openclaw/workspace/clawvision
pip install -r requirements.txt
```

### 2. Get Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key
3. Copy the key

### 3. Set Up Phone Camera

1. Install "IP Webcam" app on your Android phone
2. Connect phone to same WiFi as your computer
3. Open the app and start the server
4. Note the URL (e.g., `http://192.168.1.100:8080`)

### 4. Configure ClawVision

Create a config file:

```bash
python main.py --create-config config.yaml
```

Edit `config.yaml` and add your API key and phone URL.

## Usage

### Run with config file:
```bash
python main.py --config config.yaml
```

### Run with command-line args:
```bash
python main.py --phone-url http://192.168.1.100:8080 --gemini-key YOUR_KEY
```

### Run with environment variable:
```bash
export GEMINI_API_KEY="your-key-here"
export PHONE_URL="http://192.168.1.100:8080"
python main.py
```

## How It Works

1. **Video Loop**: Continuously captures frames from phone camera
2. **Audio Loop**: Records from local mic until silence detected
3. **Processing**: Sends image + audio to Gemini REST API
4. **Response**: Gets text response, converts to speech with gTTS
5. **Repeat**: Back to listening for more voice input

## Testing the API

Test your Gemini API connection:

```bash
export GEMINI_API_KEY="your-key-here"
python test_api.py
```

## Troubleshooting

### "gTTS not available"
Install it: `pip install gtts`

### No audio recorded
Check your microphone in system settings. List devices with:
```bash
python -m sounddevice
```

### Can't connect to phone
Make sure phone and computer are on same WiFi. Try opening the phone URL in a browser.

### API errors
Check your API key is correct and has quota available at https://aistudio.google.com/app/apikey

## Files

- `main.py` - Main application loop
- `gemini_rest_client.py` - REST API client for Gemini
- `video_capture.py` - Phone camera capture
- `audio_manager.py` - Audio handling (kept for compatibility)
- `openclaw_bridge.py` - OpenClaw tool execution
- `test_api.py` - API connection test

## Architecture

```
┌─────────────┐     ┌─────────────┐
│  Phone Cam  │────▶│ VideoCapture│
└─────────────┘     └──────┬──────┘
                           │
┌─────────────┐     ┌──────▼──────┐     ┌─────────────┐
│ Local Mic   │────▶│AudioRecorder│────▶│ Gemini REST │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                              ┌────────────────┴────────┐
                              │                         ▼
                              ▼                  ┌─────────────┐
                        ┌──────────┐             │ OpenClaw    │
                        │  gTTS    │             │ (tools)     │
                        └────┬─────┘             └─────────────┘
                             │
                             ▼
                        ┌──────────┐
                        │ Speakers │
                        └──────────┘
```

## License

MIT
