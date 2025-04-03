#!/bin/bash

# Simple install script for claude-mirror

echo "Installing claude-mirror..."

# Remove previous build artifacts
rm -rf dist/ build/ *.egg-info/

# Find the Python executable to use
if [ -n "$PYTHON" ]; then
    # Use the Python specified by environment variable
    PYTHON_CMD="$PYTHON"
elif command -v python3 &> /dev/null; then
    # Prefer python3 if available
    PYTHON_CMD="python3"
else
    # Fall back to python
    PYTHON_CMD="python"
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Install package
echo "Installing package with $PYTHON_CMD..."
$PYTHON_CMD -m pip install -e .

# Create portable launcher script for pyenv users
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create bin directory if it doesn't exist
mkdir -p $SCRIPT_DIR/bin

# Create the launcher script
cat > $SCRIPT_DIR/bin/claude-mirror <<EOF
#!/bin/bash
# Portable launcher script for claude-mirror

# Find the right Python with the package installed
if command -v python3 &> /dev/null; then
    if python3 -c "import claude_mirror" 2>/dev/null; then
        # Found package in python3
        python3 -m claude_mirror.cli "\$@"
        exit \$?
    fi
fi

# Try with 'python'
if command -v python &> /dev/null; then
    if python -c "import claude_mirror" 2>/dev/null; then
        # Found package in python
        python -m claude_mirror.cli "\$@"
        exit \$?
    fi
fi

# If we get here, we couldn't find the package
echo "Error: claude_mirror package not found in Python path"
echo "Try running: python -m pip install -e /path/to/claude-mirror"
exit 1
EOF
chmod +x $SCRIPT_DIR/bin/claude-mirror

# Option to install to a system location
install_to_path() {
    if [[ "$EUID" -eq 0 ]]; then
        # If running as root/sudo, install to /usr/local/bin
        install_dir="/usr/local/bin"
    else
        # Otherwise use ~/.local/bin which is commonly in PATH
        install_dir="$HOME/.local/bin"
        mkdir -p "$install_dir"
    fi
    
    echo "Installing launcher script to $install_dir"
    cp "$SCRIPT_DIR/bin/claude-mirror" "$install_dir/"
    
    # Check if directory is in PATH
    if [[ ":$PATH:" != *":$install_dir:"* ]]; then
        echo "NOTE: $install_dir is not in your PATH."
        echo "Add this line to your shell profile (.bashrc, .zshrc, etc.):"
        echo "    export PATH=\"$install_dir:\$PATH\""
    else
        echo "✅ Launcher installed successfully to $install_dir"
    fi
}

echo "✅ Package installed successfully."
echo ""
echo "The package provides three ways to run it:"
echo ""
echo "OPTION 1: Install launcher globally (recommended for all environments)"
read -p "Would you like to install the launcher to your PATH? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_to_path
fi
echo ""
echo "OPTION 2: Use the local launcher script (works with any Python version)"
echo "    Add this to your PATH: $SCRIPT_DIR/bin"
echo "    You can add it to your path temporarily with:"
echo "    export PATH=\"$SCRIPT_DIR/bin:\$PATH\""
echo ""
echo "OPTION 3: Run with Python module syntax (always works)"
echo "    Run directly with:"
echo "    $PYTHON_CMD -m claude_mirror.cli"
echo ""
echo "After setup, start the server with:"
echo "    claude-mirror [--setup]    # if using Option 1 or 2"
echo "    $PYTHON_CMD -m claude_mirror.cli [--setup]    # if using Option 3"