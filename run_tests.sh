#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/venv"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install -q -r requirements.txt

case "${1:-all}" in
    unit)
        echo "Running unit tests..."
        pytest tests/ -m "not integration" -v
        ;;
    integration)
        echo "Running integration tests (requires Docker + Ollama running)..."
        pytest tests/test_integration.py -v
        ;;
    all)
        echo "Running all tests..."
        pytest tests/ -v
        ;;
    *)
        echo "Usage: $0 [unit|integration|all]"
        exit 1
        ;;
esac
