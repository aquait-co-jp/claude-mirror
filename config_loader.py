import os
import yaml
import re
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List, Union, Literal

# Skip Pydantic models for now to simplify configuration

def load_env_variables(config_str: str) -> str:
    """
    Replace ${ENV_VAR} or $ENV_VAR patterns with their values from environment variables.
    If the environment variable is not set, load it from .env first
    """
    # Load .env explicitly to make sure we get these variables
    from dotenv import load_dotenv
    load_dotenv()
    
    pattern = r'\${([^}]+)}|\$([a-zA-Z0-9_]+)'
    
    def replace_var(match):
        # Extract variable name from either ${VAR} or $VAR format
        var_name = match.group(1) if match.group(1) else match.group(2)
        if var_name not in os.environ:
            raise ValueError(f"Required environment variable '{var_name}' is not set. Please set it in your .env file.")
        return os.environ[var_name]
    
    return re.sub(pattern, replace_var, config_str)

def load_config() -> Dict:
    """Load configuration from config.yaml and environment variables"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        # Replace environment variables in the config file
        config_str = load_env_variables(config_str)
        
        # Parse YAML and return the dict directly
        return yaml.safe_load(config_str)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        raise

# Global configuration instance
config = load_config()

def get_model_config(category: str):
    """Get model configuration for a given category (big, small)"""
    if category not in config["model_categories"]:
        raise ValueError(f"Unknown model category: {category}. Available categories: {list(config['model_categories'].keys())}")
    
    model_config = config["model_categories"][category]
    provider = model_config["provider"]
    
    if provider not in config["providers"]:
        raise ValueError(f"Provider '{provider}' not configured")
    
    provider_config = config["providers"][provider]
    
    return {
        "provider": provider,
        "deployment": model_config["deployment"],
        "config": provider_config
    }

def map_to_litellm_model(category: str) -> str:
    """Map a model category to a LiteLLM formatted model string"""
    model_config = get_model_config(category)
    provider = model_config["provider"]
    deployment = model_config["deployment"]
    
    if provider == "openai":
        return f"openai/{deployment}"
    elif provider == "anthropic":
        return f"anthropic/{deployment}"
    elif provider == "azure":
        return f"azure/{deployment}"
    elif provider == "databricks":
        return f"databricks/{deployment}"
    else:
        raise ValueError(f"Unknown provider: {provider}")

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