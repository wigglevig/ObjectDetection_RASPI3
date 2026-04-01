#!/bin/bash

echo "Starting Python 3.11 installation using Pyenv..."

# 1. Install prerequisites for building Python on Raspberry Pi OS (Debian)
echo "Installing prerequisites..."
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# 2. Install Pyenv
echo "Installing Pyenv..."
curl https://pyenv.run | bash

# 3. Add Pyenv to bashrc so it persists
# Check if it's already in .bashrc to avoid duplicates
if ! grep -q 'pyenv' ~/.bashrc; then
    echo "Adding pyenv to ~/.bashrc"
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
fi

# Temporarily export for this current script session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

# 4. Install Python 3.11
echo "Compiling Python 3.11... (This might take 10-15 minutes on a Raspberry Pi 4)"
pyenv install 3.11.9

# 5. Set it as the default python for your user
pyenv global 3.11.9

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Python 3.11 installed successfully!"
echo "⚠️ IMPORTANT: Please restart your Raspberry Pi or run: source ~/.bashrc"
echo "After that, running 'python -V' should say Python 3.11.9"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
