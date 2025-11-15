#!/bin/bash

# Set project root
export PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcp-keys/kitchen-forecast-key.json"

# Activate venv
source "$PROJECT_ROOT/venv/Scripts/activate"

echo "✅ Environment ready!"
echo "📁 Project root: $PROJECT_ROOT"
