import os
import yaml
import re
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List, Union, Literal

# Skip Pydantic models for now to simplify configuration

# Removed environment variable handling completely - no backward compatibility

def load_config() -> Dict:
    """Load configuration directly from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        # Parse YAML and return the dict directly
        # No longer processing environment variables
        return yaml.safe_load(config_str)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        raise

# Global configuration instance
config = load_config()

def get_model_config(category: str):
    """Get model configuration for a given category (large, small)"""
    if category not in config["model_categories"]:
        raise ValueError(f"Unknown model category: {category}. Available categories: {list(config['model_categories'].keys())}")
    
    # Get the provider/model string from the category mapping
    model_string = config["model_categories"][category]
    
    # Split into provider and deployment
    if "/" not in model_string:
        raise ValueError(f"Invalid model format for category '{category}': {model_string}. Expected format: 'provider/model'")
    
    provider, deployment = model_string.split("/", 1)
    
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
    # For our simplified direct mapping, just return the configured value
    if category not in config["model_categories"]:
        raise ValueError(f"Unknown model category: {category}. Available categories: {list(config['model_categories'].keys())}")
    
    return config["model_categories"][category]

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