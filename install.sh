#!/bin/bash

# Simple install script for claude-mirror

echo "Installing claude-mirror..."

# Remove previous build artifacts
rm -rf dist/ build/ *.egg-info/

# Check if uv is installed
if command -v uv &> /dev/null; then
    # Install in development mode using uv
    uv pip install -e .
    
    echo "✅ Package installed with uv."
    echo ""
    echo "To configure your setup, run:"
    echo "  claude-mirror --setup"
    echo ""
    echo "After setup, you can run:"
    echo "  claude-mirror"
else
    # Fallback to pip if uv is not available
    echo "Note: uv not found, using pip instead"
    pip install -e .
    
    echo "✅ Package installed with pip."
    echo ""
    echo "To configure your setup, run:"
    echo "  claude-mirror --setup"
    echo ""
    echo "After setup, you can run:"
    echo "  claude-mirror"
fi