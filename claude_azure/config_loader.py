import os
import yaml
import re
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List, Union, Literal

def load_config() -> Dict:
    """Load configuration from config.yaml or the path specified in CLAUDE_AZURE_CONFIG env var"""
    # Check for config path in environment variable
    config_path = os.environ.get("CLAUDE_AZURE_CONFIG")
    
    # If no env var, look in default locations
    if not config_path:
        # First try current directory
        if os.path.isfile("config.yaml"):
            config_path = "config.yaml"
        # Finally try user's home directory
        else:
            home_config = os.path.join(os.path.expanduser("~"), ".claude-azure", "config.yaml")
            if os.path.isfile(home_config):
                config_path = home_config
    
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Config file not found. Please run 'claude-azure --setup' to create a configuration file, "
            "or specify a config file with 'claude-azure --config /path/to/config.yaml'."
        )
        
    try:
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        # Parse YAML and return the dict directly
        return yaml.safe_load(config_str)
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        raise

# Global configuration instance (will be loaded on first import)
config = load_config()

def get_model_config(category: str):
    """Get model configuration for a given category (large, small)"""
    if category not in config["model_categories"]:
        raise ValueError(f"Unknown model category: {category}. Available categories: {list(config['model_categories'].keys())}")
    
    # Get the provider and deployment from the category mapping
    cat_config = config["model_categories"][category]
    
    provider = cat_config.get("provider")
    deployment = cat_config.get("deployment")
    
    if not provider or not deployment:
        raise ValueError(f"Invalid model configuration for category '{category}'. 'provider' and 'deployment' are required.")
    
    if provider not in config["providers"]:
        raise ValueError(f"Provider '{provider}' not configured")
    
    provider_config = config["providers"][provider]
    
    return {
        "provider": provider,
        "deployment": deployment,
        "config": provider_config
    }

def map_to_litellm_model(category: str) -> str:
    """Map a model category to a LiteLLM formatted model string"""
    if category not in config["model_categories"]:
        raise ValueError(f"Unknown model category: {category}. Available categories: {list(config['model_categories'].keys())}")
    
    # Get the category config
    cat_config = config["model_categories"][category]
    
    # Format as provider/deployment string for LiteLLM
    provider = cat_config.get("provider")
    deployment = cat_config.get("deployment")
    
    if not provider or not deployment:
        raise ValueError(f"Invalid model configuration for category '{category}'. 'provider' and 'deployment' are required.")
    
    # Return as provider/deployment format
    return f"{provider}/{deployment}"

def get_provider_params(model: str) -> Dict[str, Any]:
    """Get provider-specific parameters for a model"""
    # Handle category references
    if model in config["model_categories"]:
        return get_provider_params(map_to_litellm_model(model))
    
    # Extract provider from model string
    provider = None
    if model.startswith("openai/"):
        provider = "openai"
    elif model.startswith("anthropic/"):
        provider = "anthropic"
    elif model.startswith("azure/"):
        provider = "azure"
    elif model.startswith("databricks/"):
        provider = "databricks"
    else:
        raise ValueError(f"Unknown model format: {model}. Use provider/model format or a category name.")
    
    if provider not in config["providers"]:
        raise ValueError(f"Provider '{provider}' not configured")
    
    provider_config = config["providers"][provider]
    params = {}
    
    # Add provider-specific parameters
    if provider == "openai":
        params["api_key"] = provider_config["api_key"]
        if "base_url" in provider_config:
            params["api_base"] = provider_config["base_url"]
    elif provider == "anthropic":
        params["api_key"] = provider_config["api_key"]
    elif provider == "azure":
        params["api_key"] = provider_config["api_key"]
        params["api_base"] = provider_config["endpoint"]
        params["api_version"] = provider_config["api_version"]
    elif provider == "databricks":
        # For databricks, extract the deployment from the model string
        deployment = model.split('/')[-1]
        params["api_key"] = provider_config["token"]
        params["api_base"] = f"{provider_config['host']}/serving-endpoints/{deployment}"
        params["model"] = "databricks"
    
    return params