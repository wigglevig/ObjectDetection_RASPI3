#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install system dependencies for OpenCV and pyttsx3
echo "Installing system dependencies..."
sudo apt-get install -y \
    libgl1-mesa-glx \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libatlas-base-dev \
    libv4l-dev \
    v4l-utils \
    espeak \
    python3-dev \
    python3-venv

# Set up virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv and install requirements
echo "Installing Python requirements..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Run with: source venv/bin/activate && python rpi_main.py"
