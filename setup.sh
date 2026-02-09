#!/bin/bash
#
# Quick setup script for ClawVision
# Run this after cloning to get started quickly
#

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  ClawVision Quick Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}Error: Python 3.10+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Check for virtual environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check for system audio libraries
echo "Checking system audio libraries..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    if ! dpkg -l | grep -q libportaudio2; then
        echo -e "${YELLOW}Warning: libportaudio2 not found${NC}"
        echo "Install with: sudo apt-get install libportaudio2"
    else
        echo -e "${GREEN}✓ libportaudio2 found${NC}"
    fi
elif command -v dnf &> /dev/null; then
    # Fedora
    if ! rpm -q portaudio &> /dev/null; then
        echo -e "${YELLOW}Warning: portaudio not found${NC}"
        echo "Install with: sudo dnf install portaudio"
    else
        echo -e "${GREEN}✓ portaudio found${NC}"
    fi
elif command -v pacman &> /dev/null; then
    # Arch
    if ! pacman -Q portaudio &> /dev/null; then
        echo -e "${YELLOW}Warning: portaudio not found${NC}"
        echo "Install with: sudo pacman -S portaudio"
    else
        echo -e "${GREEN}✓ portaudio found${NC}"
    fi
fi
echo ""

# Create config file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "Creating sample configuration file..."
    cp config.yaml.example config.yaml
    echo -e "${YELLOW}⚠ Please edit config.yaml with your API keys${NC}"
else
    echo -e "${GREEN}✓ config.yaml already exists${NC}"
fi
echo ""

# Instructions
echo "═══════════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Get a Gemini API key:"
echo "   https://aistudio.google.com/app/apikey"
echo ""
echo "2. Edit config.yaml and add your API key:"
echo "   nano config.yaml"
echo ""
echo "3. Install IP Webcam on your Android phone:"
echo "   https://play.google.com/store/apps/details?id=com.pas.webcam"
echo ""
echo "4. Start the IP Webcam app and note the IP address"
echo ""
echo "5. Update config.yaml with your phone's IP address"
echo ""
echo "6. Run ClawVision:"
echo "   python main.py --config config.yaml"
echo ""
echo -e "${GREEN}Happy hacking!${NC}"
