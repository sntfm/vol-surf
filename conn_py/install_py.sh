#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Install Python deps
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Compile FlatBuffers
flatc -o "$SCRIPT_DIR" --python "$PROJECT_ROOT/schemas/OptionData.fbs"

