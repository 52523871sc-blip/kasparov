#!/bin/bash
# Start Chess API with Stockfish

echo "=========================================="
echo "Starting Chess API with Stockfish Engine"
echo "=========================================="

# Kill any existing Stockfish processes
pkill -f stockfish 2>/dev/null

# Start the API
python3 chess_api.py