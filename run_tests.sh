#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verify the api container is running
if ! docker compose ps --status running api 2>/dev/null | grep -q "api"; then
    echo "Error: api container is not running. Start it with: docker compose up -d"
    exit 1
fi

case "${1:-all}" in
    unit)
        echo "Running unit tests..."
        docker compose exec api pytest tests/ -m "not integration" -v
        ;;
    integration)
        echo "Running integration tests..."
        docker compose exec api pytest tests/test_integration.py -v
        ;;
    all)
        echo "Running all tests..."
        docker compose exec api pytest tests/ -v
        ;;
    *)
        echo "Usage: $0 [unit|integration|all]"
        exit 1
        ;;
esac
