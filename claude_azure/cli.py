#!/usr/bin/env python3
"""
Command-line entry point for Claude Azure

This module provides the main entry point for the claude-azure command.
It starts the proxy server and then launches the Claude Code CLI.
When Claude exits, the proxy server is automatically terminated.

Usage:
  claude-azure         # Normal mode (minimal output)
  claude-azure --debug # Debug mode (verbose logs)
"""

import os
import sys
import subprocess
import time
import signal
import atexit
import logging
import socket
import argparse
import yaml
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Connect Claude Code to Azure OpenAI and other OpenAI-compatible providers"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--config", type=str, help="Path to config.yaml file", 
                      default=None)
    parser.add_argument("--setup", action="store_true", help="Run interactive setup to create config file")
    return parser.parse_args()

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

def display_config(config):
    """Display current configuration in a readable format."""
    print("\n=== Current Configuration ===\n")
    
    # Display providers
    print("Configured Providers:")
    if "providers" in config and config["providers"]:
        for provider, details in config["providers"].items():
            if provider == "openai":
                api_key = details.get("api_key", "")
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "..."
                print(f"  • OpenAI: API Key: {masked_key}")
            elif provider == "azure":
                api_key = details.get("api_key", "")
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "..."
                print(f"  • Azure OpenAI: Endpoint: {details.get('endpoint', '')}")
                print(f"                 API Key: {masked_key}")
                print(f"                 API Version: {details.get('api_version', '')}")
            elif provider == "databricks":
                token = details.get("token", "")
                masked_token = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "..."
                print(f"  • Databricks: Host: {details.get('host', '')}")
                print(f"               Token: {masked_token}")
    else:
        print("  No providers configured.")
    
    # Display model categories
    print("\nModel Mappings:")
    if "model_categories" in config and config["model_categories"]:
        big_config = config["model_categories"].get("big", {})
        if big_config:
            print(f"  • Big model: {big_config.get('provider', '')}/{big_config.get('deployment', '')}")
        else:
            print("  • Big model: Not configured")
            
        small_config = config["model_categories"].get("small", {})
        if small_config:
            print(f"  • Small model: {small_config.get('provider', '')}/{small_config.get('deployment', '')}")
        else:
            print("  • Small model: Not configured")
    else:
        print("  No model mappings configured.")

def configure_provider(config, provider_name):
    """Configure a specific provider."""
    if "providers" not in config:
        config["providers"] = {}
    
    if provider_name == "openai":
        print("\n=== OpenAI Configuration ===\n")
        api_key = input("Enter your OpenAI API key: ")
        while not api_key.strip():
            print("API key cannot be empty.")
            api_key = input("Enter your OpenAI API key: ")
        config["providers"]["openai"] = {"api_key": api_key}
        print("OpenAI configuration updated.")
        
    elif provider_name == "azure":
        print("\n=== Azure OpenAI Configuration ===\n")
        api_key = input("Enter your Azure OpenAI API key: ")
        while not api_key.strip():
            print("API key cannot be empty.")
            api_key = input("Enter your Azure OpenAI API key: ")
        
        endpoint = input("Enter your Azure OpenAI endpoint (example: your-instance.openai.azure.com): ")
        while not endpoint.strip():
            print("Endpoint cannot be empty.")
            endpoint = input("Enter your Azure OpenAI endpoint: ")
        
        print("Enter API version (example: 2023-05-15)")
        api_version = input("API version: ")
        while not api_version.strip():
            print("API version cannot be empty.")
            api_version = input("API version: ")
        
        config["providers"]["azure"] = {
            "api_key": api_key,
            "endpoint": endpoint,
            "api_version": api_version
        }
        print("Azure OpenAI configuration updated.")
        
    elif provider_name == "databricks":
        print("\n=== Databricks Configuration ===\n")
        token = input("Enter your Databricks token: ")
        while not token.strip():
            print("Token cannot be empty.")
            token = input("Enter your Databricks token: ")
        
        host = input("Enter your Databricks host (example: adb-12345678901234.12.azuredatabricks.net): ")
        while not host.strip():
            print("Host cannot be empty.")
            host = input("Enter your Databricks host: ")
        
        config["providers"]["databricks"] = {
            "token": token,
            "host": host
        }
        print("Databricks configuration updated.")
    
    return config

def configure_model_category(config, category):
    """Configure a model category (big or small)."""
    if "model_categories" not in config:
        config["model_categories"] = {}
    
    if "providers" not in config or not config["providers"]:
        print("Error: No providers configured. Please configure a provider first.")
        return config

    providers = list(config["providers"].keys())
    
    print(f"\n=== {category.title()} Model Configuration ===\n")
    if category == "big":
        print("Configure which model to use for computationally intensive tasks.")
        print("Claude Code uses this model for complex reasoning and generation.")
    else:
        print("Configure which model to use for lighter, faster tasks.")
        print("Claude Code uses this model for simpler queries and routine assistance.")
    
    # Show provider options
    print("Available providers:", ", ".join(providers))
    provider = input(f"Which provider do you want to use for '{category}' models? Available: " + ", ".join(providers) + ": ")
    while provider not in providers:
        print(f"Error: '{provider}' is not in the configured providers list.")
        provider = input("Which provider? Choose from: " + ", ".join(providers) + ": ")
    
    # Get deployment name with suggestions
    if provider == "openai":
        if category == "big":
            print("Enter model name for 'big' models (suggestion: gpt-4o)")
        else:
            print("Enter model name for 'small' models (suggestion: gpt-4o-mini)")
    elif provider == "azure":
        if category == "big":
            print("Enter deployment name for 'big' models (example: your-gpt4-deployment)")
        else:
            print("Enter deployment name for 'small' models (example: your-gpt35-deployment)")
    elif provider == "databricks":
        if category == "big":
            print("Enter model name for 'big' models (suggestion: databricks-claude-3-sonnet)")
        else:
            print("Enter model name for 'small' models (suggestion: databricks-claude-3-haiku)")
    
    model = input("Model/deployment name: ")
    while not model.strip():
        print("Model/deployment name cannot be empty.")
        model = input("Model/deployment name: ")
    
    config["model_categories"][category] = {
        "provider": provider,
        "deployment": model
    }
    
    print(f"{category.title()} model configuration updated.")
    return config

def save_config(config, config_path):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nConfiguration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False

def remove_provider(config, provider_name):
    """Remove a provider from the configuration."""
    if "providers" in config and provider_name in config["providers"]:
        # Check if this provider is being used by any model category
        in_use = False
        if "model_categories" in config:
            for category, cat_config in config["model_categories"].items():
                if cat_config.get("provider") == provider_name:
                    in_use = True
                    print(f"Warning: This provider is used by the '{category}' model category.")
        
        if in_use:
            confirm = input("Removing this provider will break model mappings. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Provider removal canceled.")
                return config
        
        del config["providers"][provider_name]
        print(f"{provider_name} provider removed.")
        
        # Update model categories that used this provider
        if "model_categories" in config:
            for category, cat_config in list(config["model_categories"].items()):
                if cat_config.get("provider") == provider_name:
                    del config["model_categories"][category]
                    print(f"Removed {category} model category that used {provider_name}.")
    else:
        print(f"Provider {provider_name} not found in configuration.")
    
    return config

def interactive_setup():
    """Run interactive setup to create or modify config file."""
    print("\n=== Claude Azure Setup ===\n")
    print("This utility will help you configure Claude Azure.")
    print("The configuration file will be created in ~/.claude-azure/config.yaml")
    
    print("\nAbout Model Categories:")
    print("Claude Code uses two categories of models:")
    print("  • Big model: Used for complex reasoning, code generation, and in-depth assistance")
    print("  • Small model: Used for quick responses, simple queries, and routine tasks")
    print("\nYou'll need to configure which provider and model to use for each category.")
    
    # Create config directory if it doesn't exist
    config_dir = os.path.expanduser("~/.claude-azure")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")
    
    # Load existing config if available
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                if not isinstance(config, dict):
                    config = {}
        except Exception as e:
            print(f"Error loading existing config: {str(e)}")
            config = {}
    
    # Ensure basic structure
    if "providers" not in config:
        config["providers"] = {}
    if "model_categories" not in config:
        config["model_categories"] = {}
    
    # Main setup loop
    while True:
        display_config(config)
        
        print("\n=== Setup Menu ===")
        print("1. Add/Update OpenAI Configuration")
        print("2. Add/Update Azure OpenAI Configuration")
        print("3. Add/Update Databricks Configuration")
        print("4. Remove a Provider")
        print("5. Configure 'Big' Model")
        print("6. Configure 'Small' Model")
        print("7. Save and Exit")
        print("8. Exit Without Saving")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            config = configure_provider(config, "openai")
        elif choice == '2':
            config = configure_provider(config, "azure")
        elif choice == '3':
            config = configure_provider(config, "databricks")
        elif choice == '4':
            if not config["providers"]:
                print("No providers configured to remove.")
                continue
                
            print("\nAvailable providers:", ", ".join(config["providers"].keys()))
            provider = input("Which provider do you want to remove? ")
            if provider in config["providers"]:
                config = remove_provider(config, provider)
            else:
                print(f"Provider '{provider}' not found.")
        elif choice == '5':
            config = configure_model_category(config, "big")
        elif choice == '6':
            config = configure_model_category(config, "small")
        elif choice == '7':
            # Validate config before saving
            if not config["providers"]:
                print("Error: You must configure at least one provider.")
                continue
                
            if not config["model_categories"]:
                print("Error: You must configure at least one model category.")
                continue
                
            # Check that each model category references a valid provider
            invalid_models = []
            for category, cat_config in config["model_categories"].items():
                provider = cat_config.get("provider")
                if provider not in config["providers"]:
                    invalid_models.append(category)
            
            if invalid_models:
                print(f"Error: These model categories reference invalid providers: {', '.join(invalid_models)}")
                continue
            
            if save_config(config, config_path):
                return config_path
        elif choice == '8':
            confirm = input("Are you sure you want to exit without saving? (y/n): ")
            if confirm.lower() == 'y':
                print("Setup exited without saving.")
                if os.path.exists(config_path):
                    return config_path
                return None
        else:
            print("Invalid choice. Please enter a number from 1-8.")

def wait_for_server(port, timeout=20):
    """Wait until the server is accepting connections on the specified port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout):
            print("Waiting for server to start...")
            time.sleep(1)
    return False

def start_proxy_server(debug_mode, config_path=None):
    """Start the proxy server as a subprocess."""
    server_script = Path(__file__).parent / "server.py"
    
    if debug_mode:
        print("Starting proxy server in debug mode...")
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
    log_level = "debug" if debug_mode else "error"
    
    # Base command
    cmd = ["uvicorn", str(server_script).replace(".py", ":app"), 
           "--host", "0.0.0.0", "--port", "8082", "--log-level", log_level]
    
    # Add config path if provided
    env = os.environ.copy()
    if config_path:
        if not os.path.isfile(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
        env["CLAUDE_AZURE_CONFIG"] = config_path
    
    # Start the proxy server with appropriate output redirection
    proxy_process = subprocess.Popen(
        cmd,
        stdout=stdout_dest,
        stderr=stderr_dest,
        env=env
    )
    
    # Register a function to kill the proxy server when this script exits
    def cleanup():
        if proxy_process.poll() is None:  # If process is still running
            print("\nShutting down proxy server...")
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_process.kill()
        
        # Close the log file if in debug mode
        if debug_mode:
            log_file.close()
    
    atexit.register(cleanup)
    
    # Wait for server to be ready
    if not wait_for_server(8082):
        print("Error: Proxy server failed to start within the timeout period.")
        sys.exit(1)
        
    print("Proxy server is running on http://0.0.0.0:8082")
    return proxy_process

def run_claude():
    """Run the Claude CLI connected to our proxy."""
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
    """Main entry point for the claude-azure command."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging based on debug mode
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run setup if requested
    if args.setup:
        config_path = interactive_setup()
        if not config_path:
            print("Setup failed or was canceled. Exiting.")
            sys.exit(1)
        print("\nSetup complete. You can now run 'claude-azure' to start using it.")
        sys.exit(0)
    
    # Check if Claude is installed
    if not check_claude_installed():
        print("Error: Claude Code CLI not found.")
        print("Install it with: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)
    
    # Use specified config path or try default locations
    config_path = args.config
    if not config_path:
        # Check if config exists in user directory
        user_config = os.path.expanduser("~/.claude-azure/config.yaml")
        if os.path.isfile(user_config):
            config_path = user_config
    
    # If still no config file, suggest running setup
    if not config_path or not os.path.isfile(config_path):
        print("Error: No configuration file found.")
        print("Please run 'claude-azure --setup' to create a configuration file.")
        sys.exit(1)
    
    # Set config path in environment variable
    os.environ["CLAUDE_AZURE_CONFIG"] = config_path
    
    # Start the proxy server
    proxy_process = start_proxy_server(args.debug, config_path)
    
    try:
        # Run Claude
        exit_code = run_claude()
        
        # Exit with the same code as Claude
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nInterrupted by user. Shutting down...")
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