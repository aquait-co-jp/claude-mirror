#!/bin/bash

# claude-proxy.sh - Start Claude Code with the OpenAI proxy
# 
# This script starts the proxy server in the background and then runs Claude.
# When Claude exits, it automatically terminates the proxy server.
#
# Usage:
#   ./claude-proxy.sh         # Normal mode
#   ./claude-proxy.sh --debug # Debug mode (verbose logs)

# Check for debug mode
DEBUG_MODE=0
if [[ "$1" == "--debug" ]]; then
    DEBUG_MODE=1
fi

# Change to the script's directory
cd "$(dirname "$0")"

# Check if Claude is installed
if ! command -v claude &> /dev/null; then
    echo "Error: Claude Code CLI not found. Install it with: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

echo "Starting proxy server..."

# Set log level based on debug mode
LOG_LEVEL="error"
if [[ $DEBUG_MODE -eq 1 ]]; then
    LOG_LEVEL="debug"
    # In debug mode, redirect to log file
    uv run uvicorn server:app --host 0.0.0.0 --port 8082 --log-level $LOG_LEVEL > proxy-server.log 2>&1 &
else
    # In normal mode, suppress all output
    uv run uvicorn server:app --host 0.0.0.0 --port 8082 --log-level $LOG_LEVEL > /dev/null 2>&1 &
fi
PROXY_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 2

# Make sure the server is running
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "Failed to start proxy server"
    exit 1
fi

echo "Starting Claude Code connected to proxy..."
echo "When you exit Claude, the proxy server will also be stopped."

# Set environment variable and run Claude
ANTHROPIC_BASE_URL=http://0.0.0.0:8082 claude

# When Claude exits, kill the proxy server
echo "Claude has exited. Stopping proxy server..."
kill $PROXY_PID 2>/dev/null || true
wait $PROXY_PID 2>/dev/null || true

echo "Done!"