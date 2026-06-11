#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse flags — --verbose can follow the suite name
SUITE="${1:-all}"
VERBOSE=0
for arg in "$@"; do
    if [[ "$arg" == "--verbose" ]]; then
        VERBOSE=1
    fi
done

# Verify the api container is running
if ! docker compose ps --status running api 2>/dev/null | grep -q "api"; then
    echo "Error: api container is not running. Start it with: docker compose up -d"
    exit 1
fi

# -s disables pytest output capture so print() calls show up in the terminal
PYTEST_EXTRA="-v"
EXEC_ENV=()
if [[ "$VERBOSE" == "1" ]]; then
    PYTEST_EXTRA="-v -s"
    EXEC_ENV=(-e SHOW_OCR=1)
    echo "Verbose mode: full pipeline trace enabled"
fi

case "$SUITE" in
    unit)
        echo "Running unit tests..."
        docker compose exec "${EXEC_ENV[@]}" api pytest tests/ -m "not integration" $PYTEST_EXTRA
        ;;
    integration)
        echo "Running integration tests..."
        docker compose exec "${EXEC_ENV[@]}" api pytest tests/test_integration.py $PYTEST_EXTRA
        ;;
    all)
        echo "Running all tests..."
        docker compose exec "${EXEC_ENV[@]}" api pytest tests/ $PYTEST_EXTRA
        ;;
    *)
        echo "Usage: $0 [unit|integration|all] [--verbose]"
        exit 1
        ;;
esac
