#!/usr/bin/env python3
"""
Wrapper for Claude Code with Claude Mirror

This script starts the proxy server and then launches the Claude Code CLI.
When Claude exits, the proxy server is automatically terminated.

Usage:
  python claude-mirror.py         # Normal mode (minimal output)
  python claude-mirror.py --debug # Debug mode (verbose logs)
"""

import os
import sys
import subprocess
import time
import signal
import atexit
import logging
import socket

# Check for debug mode
DEBUG_MODE = "--debug" in sys.argv

# Set up logging
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

def check_claude_installed():
    """Check if Claude CLI is installed and available."""
    try:
        subprocess.run(["claude", "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=False)
        return True
    except FileNotFoundError:
        return False

def wait_for_server(port, timeout=10):
    """Wait until the server is accepting connections on the specified port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout):
            logger.info("Waiting for server to start...")
            time.sleep(1)
    return False

def start_proxy_server():
    """Start the proxy server as a subprocess."""
    if DEBUG_MODE:
        logger.info("Starting proxy server in debug mode...")
        # In debug mode, create a log file for detailed output
        log_file = open("proxy-server.log", "w")
        stdout_dest = log_file
        stderr_dest = log_file
    else:
        print("Starting proxy server...")
        # In normal mode, suppress output completely
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
    
    # Add --log-level flag based on debug mode
    log_level = "debug" if DEBUG_MODE else "error"
    
    # Start the proxy server with appropriate output redirection
    proxy_process = subprocess.Popen(
        ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082", "--log-level", log_level],
        stdout=stdout_dest,
        stderr=stderr_dest
    )
    
    # Register a function to kill the proxy server when this script exits
    def cleanup():
        if proxy_process.poll() is None:  # If process is still running
            if DEBUG_MODE:
                logger.info("\nShutting down proxy server...")
            else:
                print("\nShutting down proxy server...")
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_process.kill()
        
        # Close the log file if in debug mode
        if DEBUG_MODE:
            log_file.close()
    
    atexit.register(cleanup)
    
    # Wait for server to be ready
    if not wait_for_server(8082):
        print("Error: Proxy server failed to start within the timeout period.")
        sys.exit(1)
        
    if DEBUG_MODE:
        logger.info("Proxy server is running on http://0.0.0.0:8082")
    else:
        print("Proxy server is running on http://0.0.0.0:8082")
    return proxy_process

def run_claude():
    """Run the Claude CLI connected to our proxy."""
    if DEBUG_MODE:
        logger.info("Starting Claude Code connected to proxy...")
        logger.info("When you exit Claude, the proxy server will also be stopped.")
    else:
        print("Starting Claude Code connected to proxy...")
        print("When you exit Claude, the proxy server will also be stopped.")
    
    # Set up environment for Claude to use our proxy
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://0.0.0.0:8082"
    
    # Run Claude with the environment variable set and pass through stdio
    claude_process = subprocess.run(
        ["claude"],
        env=env,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    return claude_process.returncode

def main():
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if Claude is installed
    if not check_claude_installed():
        logger.error("Error: Claude Code CLI not found.")
        logger.error("Install it with: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)
    
    # Start the proxy server
    proxy_process = start_proxy_server()
    
    try:
        # Run Claude
        exit_code = run_claude()
        
        # Exit with the same code as Claude
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("\nInterrupted by user. Shutting down...")
    finally:
        # Ensure the proxy server is terminated
        if proxy_process.poll() is None:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_process.kill()

if __name__ == "__main__":
    main()