#!/bin/bash

set -e  # Exit on error

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS system"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    echo "Installing dependencies via Homebrew..."
    brew install gcc@14 flatbuffers cmake
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux system"
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Installing dependencies via apt..."
        sudo apt-get update
        sudo apt-get install -y gcc g++ cmake flatbuffers-compiler libflatbuffers-dev libomp-dev
    elif command -v dnf &> /dev/null; then
        # Fedora
        echo "Installing dependencies via dnf..."
        sudo dnf install -y gcc gcc-c++ cmake flatbuffers flatbuffers-devel libomp-devel
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo "Installing dependencies via pacman..."
        sudo pacman -Syu --noconfirm gcc cmake flatbuffers openmp
    else
        echo "Unsupported Linux distribution. Please install the following packages manually:"
        echo "- GCC/G++"
        echo "- CMake"
        echo "- FlatBuffers"
        echo "- OpenMP"
        exit 1
    fi
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "All dependencies installed successfully!" 