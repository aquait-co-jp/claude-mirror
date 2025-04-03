from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal, Type, TypeVar
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
# Removed dotenv import - no longer using environment variables
import re
from datetime import datetime
import sys

# Removed .env loading completely - using only config.yaml

# Configure logging with more verbose output
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO level for more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import config loader
from .config_loader import config, map_to_litellm_model, get_provider_params
USE_CONFIG = True
logger.info("Using config.yaml for configuration")

# Import uvicorn
import uvicorn

# Configure uvicorn with more verbose logging
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

print("\n======= LOGGING INITIALIZED =======")
sys.stdout.flush()

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add explicit startup message
print("\n======= STARTING SERVER: Environment Check =======")
sys.stdout.flush()

if USE_CONFIG:
    # Use strict configuration from config_loader (no fallbacks)
    print("Loading model configuration from config.yaml with STRICT validation (no fallbacks)...")
    
    # We now use direct mapping without swapping flags - mapping is defined in the config
    
    # Check available providers based on config
    has_openai = "openai" in config["providers"] and "api_key" in config["providers"]["openai"] and config["providers"]["openai"]["api_key"]
    has_anthropic = "anthropic" in config["providers"] and "api_key" in config["providers"]["anthropic"] and config["providers"]["anthropic"]["api_key"]
    has_azure = ("azure" in config["providers"] and 
                "api_key" in config["providers"]["azure"] and config["providers"]["azure"]["api_key"] and
                "endpoint" in config["providers"]["azure"] and config["providers"]["azure"]["endpoint"])
    has_databricks = ("databricks" in config["providers"] and 
                     "token" in config["providers"]["databricks"] and config["providers"]["databricks"]["token"] and
                     "host" in config["providers"]["databricks"] and config["providers"]["databricks"]["host"])
    
    # Store values from configured providers only (no default fallbacks)
    # For any provider not in the config, we'll fail if it's requested
    OPENAI_API_KEY = config["providers"]["openai"]["api_key"] if has_openai else None
    ANTHROPIC_API_KEY = config["providers"]["anthropic"]["api_key"] if has_anthropic else None
    AZURE_OPENAI_API_KEY = config["providers"]["azure"]["api_key"] if has_azure else None
    AZURE_OPENAI_ENDPOINT = config["providers"]["azure"]["endpoint"] if has_azure else None
    AZURE_OPENAI_API_VERSION = config["providers"]["azure"]["api_version"] if has_azure else None
    DATABRICKS_TOKEN = config["providers"]["databricks"]["token"] if has_databricks else None
    DATABRICKS_HOST = config["providers"]["databricks"]["host"] if has_databricks else None
    
    # Require both categories to be configured explicitly - no defaults
    if "big" not in config["model_categories"]:
        raise ValueError("Error: 'big' category must be explicitly configured in config.yaml")
        
    if "small" not in config["model_categories"]:
        raise ValueError("Error: 'small' category must be explicitly configured in config.yaml")
        
    # No fallbacks - we need these values for legacy code paths but they come directly from config
    BIG_MODEL = config["model_categories"]["big"]["deployment"]
    SMALL_MODEL = config["model_categories"]["small"]["deployment"]
    
    # Log which configurations are present
    print("Available provider configurations:")
    print(f"  - OpenAI API: {'CONFIGURED' if has_openai else 'NOT CONFIGURED'}")
    print(f"  - Azure OpenAI: {'CONFIGURED' if has_azure else 'NOT CONFIGURED'}")
    print(f"  - Databricks: {'CONFIGURED' if has_databricks else 'NOT CONFIGURED'}")
    print(f"  - Anthropic API: {'CONFIGURED' if has_anthropic else 'NOT CONFIGURED'}")
    
    # Log model categories
    print("\nConfigured model categories:")
    for category, model_config in config["model_categories"].items():
        print(f"  - {category}: {model_config['provider']}/{model_config['deployment']}")
    
    # No fallbacks with Azure deployments either - if model mappings reference azure, they must be defined
    AZURE_DEPLOYMENTS = {}
    if has_azure:
        for category, model_config in config["model_categories"].items():
            if model_config["provider"] == "azure":
                AZURE_DEPLOYMENTS[model_config["deployment"]] = model_config["deployment"]

# Configuration is strictly required via config_loader

class ModelHelper:
    """Helper class for model handling operations to reduce code duplication."""
    
    @staticmethod
    def get_azure_deployment(model_name):
        """Get Azure deployment name for a given model"""
        if model_name not in AZURE_DEPLOYMENTS:
            raise ValueError(f"No Azure deployment configured for model: {model_name}")
        return AZURE_DEPLOYMENTS[model_name]
    
    @staticmethod
    def get_clean_model_name(model):
        """Extract the base model name without provider prefix."""
        if model.startswith("anthropic/"):
            return model[len("anthropic/"):]
        elif model.startswith("openai/"):
            return model[len("openai/"):]
        elif model.startswith("azure/"):
            return model[len("azure/"):]
        elif model.startswith("databricks/"):
            return model[len("databricks/"):]
        return model
    
    @staticmethod
    def get_display_model_name(model):
        """Get a clean model name for display purposes."""
        if "/" in model:
            return model.split("/")[-1]
        return model
    
    @staticmethod
    def determine_provider(model):
        """Determine the provider based on model prefix."""
        if model.startswith("openai/"):
            return "openai"
        elif model.startswith("anthropic/"):
            return "anthropic"
        elif model.startswith("azure/"):
            return "azure"
        elif model.startswith("databricks/"):
            return "databricks"
        # Default model provider detection
        if "haiku" in model.lower() or "sonnet" in model.lower() or "claude" in model.lower():
            return "anthropic"
        return "unknown"
    
    @staticmethod
    def map_model_for_openai(original_model):
        """Map models based on configuration"""
        # If the model is a category name, map it according to config
        if original_model in config["model_categories"].keys():
            mapped_model = map_to_litellm_model(original_model)
            logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {mapped_model} (category mapping)")
            return mapped_model
            
        # Keep explicit provider prefixes as is
        if any(original_model.startswith(prefix) for prefix in ['openai/', 'azure/', 'databricks/']):
            return original_model
            
        # Handle Anthropic models - map to configured alternatives if needed
        if original_model.startswith('anthropic/') or 'claude' in original_model.lower():
            clean_model = ModelHelper.get_clean_model_name(original_model)
            
            # Map to appropriate category based on model type
            if 'haiku' in clean_model.lower():
                if 'small' not in config["model_categories"]:
                    raise HTTPException(status_code=400, detail="Cannot map Claude Haiku model: 'small' category not configured")
                mapped_model = map_to_litellm_model('small')
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {mapped_model}")
                return mapped_model
            
            elif 'sonnet' in clean_model.lower() or 'opus' in clean_model.lower():
                if 'big' not in config["model_categories"]:
                    raise HTTPException(status_code=400, detail="Cannot map Claude Sonnet/Opus model: 'big' category not configured")
                mapped_model = map_to_litellm_model('big')
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {mapped_model}")
                return mapped_model
        
        # No defaults for unrecognized models - require explicit provider prefix
        if not any(original_model.startswith(prefix) for prefix in ['openai/', 'azure/', 'databricks/', 'anthropic/']):
            raise HTTPException(status_code=400, detail=f"Unrecognized model: {original_model}. Must use explicit provider prefix or a configured model category.")
        
        return original_model
    
    @staticmethod
    def configure_provider_parameters(model, litellm_request):
        """Configure provider-specific parameters based on the model."""
        # Handle category references
        if model in config["model_categories"].keys():
            model = map_to_litellm_model(model)
        
        # Determine provider from model string - no defaults
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
            # No default provider - require explicit provider prefix
            raise HTTPException(status_code=400, 
                detail="Unrecognized model prefix. Please specify model with explicit provider prefix or use a category name.")
        
        # Check if provider is configured
        if provider not in config["providers"]:
            raise HTTPException(status_code=400, detail=f"{provider} configuration required but not configured")
        
        # Get provider-specific parameters
        params = get_provider_params(model)
        
        # Update the request with provider parameters
        for key, value in params.items():
            litellm_request[key] = value
        
        if provider == "azure":
            # Extract the deployment name from azure/deployment_name
            deployment_name = model[6:]
            litellm_request["model"] = f"azure/{deployment_name}"
            logger.debug(f"Using Azure OpenAI configuration for deployment: {deployment_name}")
        elif provider == "databricks":
            # Extract the Databricks model name
            databricks_model = model.split("/")[-1]
            litellm_request["model"] = "databricks"
            logger.debug(f"Using Databricks configuration for model: {databricks_model}")
        
        return litellm_request

app = FastAPI()

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool

# Create a TypeVar for generic model type
T = TypeVar('T', bound=BaseModel)

# Base model class for common validation logic
class ModelRequestBase(BaseModel):
    model: str
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model(cls, v, info):
        """Shared model validation logic for all request types"""
        # Store the original model name
        original_model = v
        
        # Map the model using our helper method - no conditional swapping
        v = ModelHelper.map_model_for_openai(v)
            
        # Store the original model in the values dictionary
        # This will be accessible as request.original_model
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
            
        return v

class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None

class MessagesRequest(ModelRequestBase):
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Dict[str, Any], ToolChoice]] = None
    thinking: Optional[ThinkingConfig] = None

class TokenCountRequest(ModelRequestBase):
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Union[Dict[str, Any], ToolChoice]] = None

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

class ContentParser:
    """Helper class for content parsing and handling."""
    
    @staticmethod
    def parse_tool_result_content(content):
        """Helper function to properly parse and normalize tool result content."""
        if content is None:
            return "No content provided"
            
        if isinstance(content, str):
            return content
            
        if isinstance(content, list):
            result = ""
            for item in content:
                if isinstance(item, dict):
                    # Handle dictionary items
                    if "type" in item and item["type"] == "text":
                        result += item.get("text", "") + "\n"
                    elif "text" in item:
                        result += item.get("text", "") + "\n"
                    else:
                        try:
                            result += json.dumps(item) + "\n"
                        except:
                            result += str(item) + "\n"
                elif hasattr(item, "type") and item.type == "text":
                    # Handle objects with type attribute
                    result += item.text + "\n"
                elif hasattr(item, "text"):
                    # Handle objects with text attribute
                    result += item.text + "\n"
                elif isinstance(item, str):
                    # Handle string items
                    result += item + "\n"
                else:
                    # Handle other types
                    try:
                        result += str(item) + "\n"
                    except:
                        result += "Unparseable content\n"
            return result.strip()
            
        if isinstance(content, dict):
            if "type" in content and content["type"] == "text":
                return content.get("text", "")
            try:
                return json.dumps(content)
            except:
                return str(content)
        
        # Handle objects with type property
        if hasattr(content, "type") and content.type == "text":
            return content.text
                
        # Fallback for any other type
        try:
            return str(content)
        except:
            return "Unparseable content"
    
    # This function has been removed as its functionality is now handled inline
    # in the places where it was used, to avoid issues with object/dict detection
    
    @staticmethod
    def process_content_for_openai(messages):
        """Process message content for OpenAI compatibility."""
        processed_messages = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                # Skip non-dict messages
                logger.warning(f"Skipping non-dict message: {type(msg)}")
                continue
                
            processed_msg = msg.copy()
            
            # Handle content field
            if "content" in msg:
                content = msg["content"]
                
                # Check if content is a list (content blocks)
                if isinstance(content, list):
                    # Convert complex content blocks to simple string
                    text_content = ""
                    for block in content:
                        # Skip None blocks
                        if block is None:
                            continue
                            
                        # Handle objects with type attribute
                        if hasattr(block, "type"):
                            if block.type == "text":
                                text_content += block.text + "\n"
                            # Handle tool_result content blocks
                            elif block.type == "tool_result":
                                tool_id = block.tool_use_id
                                text_content += f"[Tool Result ID: {tool_id}]\n"
                                result_content = block.content
                                
                                # Convert result content to a clean format for OpenAI
                                if isinstance(result_content, list):
                                    for item in result_content:
                                        if item is None:
                                            continue
                                        if isinstance(item, dict) and "type" in item and item["type"] == "text":
                                            text_content += item["text"] + "\n"
                                        elif hasattr(item, "type") and item.type == "text":
                                            text_content += item.text + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(item) + "\n"
                                            except (TypeError, ValueError):
                                                text_content += str(item) + "\n"
                                elif isinstance(result_content, str):
                                    text_content += result_content + "\n"
                                else:
                                    try:
                                        text_content += json.dumps(result_content) + "\n"
                                    except (TypeError, ValueError):
                                        text_content += str(result_content) + "\n"
                                        
                            # Handle tool_use content blocks
                            elif block.type == "tool_use":
                                tool_name = block.name
                                tool_id = block.id
                                try:
                                    tool_input_json = json.dumps(block.input)
                                except (TypeError, ValueError):
                                    tool_input_json = str(block.input)
                                text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input_json}\n\n"
                            
                            # Handle image content blocks
                            elif block.type == "image":
                                text_content += "[Image content - not displayed in text format]\n"
                        # Handle dictionary style content blocks (legacy format)
                        elif isinstance(block, dict) and "type" in block:
                            if block["type"] == "text":
                                text_content += block.get("text", "") + "\n"
                            elif block["type"] == "tool_result":
                                tool_id = block.get("tool_use_id", "unknown")
                                text_content += f"[Tool Result ID: {tool_id}]\n"
                                result_content = block.get("content", {})
                                
                                # Process result content
                                parsed_content = ContentParser.parse_tool_result_content(result_content)
                                text_content += parsed_content + "\n"
                            elif block["type"] == "tool_use":
                                tool_name = block.get("name", "unknown")
                                tool_id = block.get("id", "unknown")
                                tool_input = block.get("input", {})
                                try:
                                    tool_input_json = json.dumps(tool_input)
                                except (TypeError, ValueError):
                                    tool_input_json = str(tool_input)
                                text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input_json}\n\n"
                            elif block["type"] == "image":
                                text_content += "[Image content - not displayed in text format]\n"
                        # Handle unknown block types
                        else:
                            text_content += f"[Unknown content block: {type(block)}]\n"
                    
                    # Make sure content is never empty for OpenAI models
                    if not text_content.strip():
                        text_content = "..."
                    
                    processed_msg["content"] = text_content.strip()
                # Also check for None or empty string content
                elif content is None:
                    processed_msg["content"] = "..." # Empty content not allowed
                elif not isinstance(content, str):
                    # If content is not a string or list, convert to string
                    try:
                        processed_msg["content"] = json.dumps(content)
                    except (TypeError, ValueError):
                        processed_msg["content"] = str(content)
            
            # Remove any fields OpenAI doesn't support in messages
            for key in list(processed_msg.keys()):
                if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                    logger.warning(f"Removing unsupported field from message: {key}")
                    del processed_msg[key]
            
            processed_messages.append(processed_msg)
            
        return processed_messages

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                # System content blocks must be SystemContent objects
                if isinstance(block, SystemContent):
                    system_text += block.text + "\n\n"
                else:
                    logger.warning(f"Ignoring non-SystemContent block in system message: {type(block)}")
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if isinstance(content, list) and msg.role == "user" and any(isinstance(block, ContentBlockToolResult) for block in content):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if isinstance(block, ContentBlockText):
                        text_content += block.text + "\n"
                    elif isinstance(block, ContentBlockToolResult):
                        # Handle tool result - extract key information
                        tool_id = block.tool_use_id
                        result_content = block.content
                        
                        # Parse and format the content consistently
                        if isinstance(result_content, str):
                            formatted_content = result_content
                        elif hasattr(result_content, "__dict__"):
                            # Convert Pydantic or other object to dict
                            formatted_content = json.dumps(result_content.__dict__, indent=2)
                        elif isinstance(result_content, list) and any(hasattr(item, "__dict__") for item in result_content):
                            # List of objects - convert to list of dicts
                            formatted_list = []
                            for item in result_content:
                                if hasattr(item, "__dict__"):
                                    formatted_list.append(item.__dict__)
                                else:
                                    formatted_list.append(item)
                            formatted_content = json.dumps(formatted_list, indent=2)
                        else:
                            # Try regular JSON dump
                            try:
                                formatted_content = json.dumps(result_content, indent=2)
                            except TypeError:
                                # If that fails, use string representation
                                formatted_content = str(result_content)
                            
                        text_content += f"Tool result for {tool_id}:\n{formatted_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                if isinstance(content, list):  # Make sure content is a list
                    processed_content = []
                    for block in content:
                        # Use specific type checks for each content block type
                        if isinstance(block, ContentBlockText):
                            processed_content.append({"type": "text", "text": block.text})
                        elif isinstance(block, ContentBlockImage):
                            processed_content.append({"type": "image", "source": block.source})
                        elif isinstance(block, ContentBlockToolUse):
                            # Handle tool use blocks
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif isinstance(block, ContentBlockToolResult):
                            # Handle tool result blocks
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id
                            }
                            
                            # Process the content field according to its type
                            if isinstance(block.content, str):
                                # If it's a simple string, create a text block for it
                                processed_content_block["content"] = [{"type": "text", "text": block.content}]
                            elif isinstance(block.content, list):
                                # If it's already a list of blocks, keep it
                                # Make sure each item is a dict, not an object
                                normalized_content = []
                                for item in block.content:
                                    if hasattr(item, "type") and hasattr(item, "__dict__"):
                                        # Convert object to dict
                                        normalized_content.append(item.__dict__)
                                    else:
                                        normalized_content.append(item)
                                processed_content_block["content"] = normalized_content
                            else:
                                # Try to convert to dict if it's an object with __dict__
                                if hasattr(block.content, "__dict__"):
                                    processed_content_block["content"] = [{"type": "text", "text": json.dumps(block.content.__dict__)}]
                                else:
                                    # Convert other types to string representation
                                    processed_content_block["content"] = [{"type": "text", "text": json.dumps(block.content)}]
                                
                            processed_content.append(processed_content_block)
                    
                    messages.append({"role": msg.role, "content": processed_content})
                else:
                    # If content isn't a string or list, convert to string
                    messages.append({"role": msg.role, "content": str(content)})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        for tool in anthropic_request.tools:
            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                }
            }
            openai_tools.append(openai_tool)
            
        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        # Log details about the tool_choice object to assist in debugging
        logger.info(f"Tool choice type: {type(anthropic_request.tool_choice)}")
        logger.info(f"Tool choice dir: {dir(anthropic_request.tool_choice)}")
        logger.info(f"Tool choice repr: {repr(anthropic_request.tool_choice)}")
        
        # We now have defined ToolChoice as a proper Pydantic model
        # Handle it based on its structure
        tool_choice = anthropic_request.tool_choice
        
        # Get the type safely regardless of whether it's a dict or object
        choice_type = None
        if isinstance(tool_choice, dict) and "type" in tool_choice:
            choice_type = tool_choice["type"]
        elif isinstance(tool_choice, ToolChoice):
            choice_type = tool_choice.type
        elif hasattr(tool_choice, "type"):
            choice_type = tool_choice.type
        
        # Process based on the choice type
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool":
            # Get the name safely
            tool_name = None
            if isinstance(tool_choice, dict) and "name" in tool_choice:
                tool_name = tool_choice["name"]
            elif isinstance(tool_choice, ToolChoice) and tool_choice.name:
                tool_name = tool_choice.name
            elif hasattr(tool_choice, "name"):
                tool_name = tool_choice.name
            
            if tool_name:
                litellm_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name}
                }
            else:
                # No tool name found, default to auto
                litellm_request["tool_choice"] = "auto"
        else:
            # Default to auto if the format is unrecognized
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = ModelHelper.get_clean_model_name(original_request.model)
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Extract data from ModelResponse object
        choices = litellm_response.choices
        message = choices[0].message if choices and len(choices) > 0 else None
        content_text = message.content if message and hasattr(message, 'content') else ""
        tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
        finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
        usage_info = litellm_response.usage
        response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - only for Claude models
        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data with better object handling
                function = None
                tool_id = None
                name = None
                arguments = None
                
                # Handle different types of tool_call objects
                if isinstance(tool_call, dict):
                    # Handle dict-style tool calls
                    if 'function' in tool_call:
                        function = tool_call['function']
                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                else:
                    # Handle object-style tool calls
                    if hasattr(tool_call, 'function'):
                        function = tool_call.function
                    if hasattr(tool_call, 'id'):
                        tool_id = tool_call.id
                    else:
                        tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                
                # Extract name and arguments from function
                if isinstance(function, dict):
                    name = function.get('name', '')
                    arguments = function.get('arguments', {})
                elif hasattr(function, 'name') and hasattr(function, 'arguments'):
                    name = function.name
                    arguments = function.arguments
                else:
                    # Default values if we can't extract properly
                    name = "unknown_function"
                    arguments = {}
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                # Handle object arguments if they have __dict__
                elif hasattr(arguments, '__dict__'):
                    arguments = arguments.__dict__
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_claude_model:
            # For non-Claude models, convert tool calls to text format
            logger.debug(f"Converting tool calls to text for non-Claude model: {clean_model}")
            
            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # Extract function data with better object handling
                function = None
                tool_id = "unknown"
                name = "unknown"
                arguments = {}
                
                # Handle different types of tool_call objects
                if isinstance(tool_call, dict):
                    # Handle dict-style tool calls
                    if 'function' in tool_call:
                        function = tool_call['function']
                    if 'id' in tool_call:
                        tool_id = tool_call['id']
                else:
                    # Handle object-style tool calls
                    if hasattr(tool_call, 'function'):
                        function = tool_call.function
                    if hasattr(tool_call, 'id'):
                        tool_id = tool_call.id
                
                # Extract name and arguments from function
                if isinstance(function, dict):
                    name = function.get('name', 'unknown')
                    arguments = function.get('arguments', {})
                elif hasattr(function, 'name'):
                    name = function.name
                    if hasattr(function, 'arguments'):
                        arguments = function.arguments
                
                # Format the arguments for display
                if isinstance(arguments, str):
                    try:
                        # Try to parse and reformat
                        arguments_str = json.dumps(json.loads(arguments), indent=2)
                    except json.JSONDecodeError:
                        # Keep as is if not valid JSON
                        arguments_str = arguments
                elif hasattr(arguments, '__dict__'):
                    # Convert object to formatted JSON
                    arguments_str = json.dumps(arguments.__dict__, indent=2)
                else:
                    # Convert dict or other object to formatted JSON
                    try:
                        arguments_str = json.dumps(arguments, indent=2)
                    except TypeError:
                        arguments_str = str(arguments)
                
                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
            
            # Add or append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # Get usage information from response
        prompt_tokens = usage_info.prompt_tokens
        completion_tokens = usage_info.completion_tokens
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                # First handle dict-style tool calls (typical from LiteLLM)
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    # Make sure function is treated as dict
                                    if hasattr(function, '__dict__') and not isinstance(function, dict):
                                        function = function.__dict__
                                    
                                    name = ""
                                    if isinstance(function, dict):
                                        name = function.get('name', '')
                                    elif hasattr(function, 'name'):
                                        name = function.name
                                    
                                    tool_id = ""
                                    if 'id' in tool_call:
                                        tool_id = tool_call['id']
                                    else:
                                        tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                # Then handle object-style tool calls
                                else:
                                    function = None
                                    if hasattr(tool_call, 'function'):
                                        function = tool_call.function
                                    
                                    name = ""
                                    if function is not None and hasattr(function, 'name'):
                                        name = function.name
                                    
                                    tool_id = ""
                                    if hasattr(tool_call, 'id'):
                                        tool_id = tool_call.id
                                    else:
                                        tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments with better object handling
                            arguments = None
                            # First handle dict-style tool calls
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call['function']
                                # Handle function as dict
                                if isinstance(function, dict) and 'arguments' in function:
                                    arguments = function['arguments']
                                # Handle function as object
                                elif hasattr(function, 'arguments'):
                                    arguments = function.arguments
                            # Then handle object-style tool calls
                            elif hasattr(tool_call, 'function'):
                                function = tool_call.function
                                if hasattr(function, 'arguments'):
                                    arguments = function.arguments
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, convert to JSON string
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    # If it's an object with __dict__, convert to JSON
                                    elif hasattr(arguments, '__dict__'):
                                        args_json = json.dumps(arguments.__dict__)
                                    # Otherwise, try to parse as JSON string
                                    else:
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment or can't be parsed, treat as string
                                    args_json = str(arguments)
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # NOTE: The request is already parsed by FastAPI into a MessagesRequest object
        # We don't need to manually parse the raw request body, except for logging
        
        # For logging, get the original model name
        original_model = request.original_model or request.model
        
        # Get the display name for logging using our helper
        display_model = ModelHelper.get_display_model_name(original_model)
        
        # Get clean model name for capability check
        clean_model = ModelHelper.get_clean_model_name(request.model)
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Configure provider-specific parameters using our helper
        litellm_request = ModelHelper.configure_provider_parameters(request.model, litellm_request)
        
        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # Use our helper method to process the messages for OpenAI compatibility
            litellm_request["messages"] = ContentParser.process_content_for_openai(litellm_request["messages"])
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_traceback = traceback.format_exc()
        
        # Get line number and file info
        tb = traceback.extract_tb(exc_traceback)
        filename, line, func, text = tb[-1] if tb else ("unknown", 0, "unknown", "unknown")
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback,
            "location": f"{filename}:{line} in {func}",
            "code": text
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Log all error details
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        
        # Format error for response with more details
        error_message = f"Error: {str(e)}\nLocation: {error_details['location']}\nCode: {error_details['code']}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # The request is already parsed by FastAPI into a TokenCountRequest object
        # Get the original model name
        original_model = request.original_model or request.model
        
        # Get the display name for logging using our helper
        display_model = ModelHelper.get_display_model_name(original_model)
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Configure provider-specific parameters using our helper
        converted_request = ModelHelper.configure_provider_parameters(request.model, converted_request)
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Count tokens
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_traceback = traceback.format_exc()
        
        # Get line number and file info
        tb = traceback.extract_tb(exc_traceback)
        filename, line, func, text = tb[-1] if tb else ("unknown", 0, "unknown", "unknown")
        
        # Log detailed error info
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "location": f"{filename}:{line} in {func}",
            "code": text,
            "traceback": error_traceback
        }
        logger.error(f"Error counting tokens: {json.dumps(error_details, indent=2)}")
        
        # Return error with more details
        error_message = f"Error counting tokens: {str(e)}\nLocation: {error_details['location']}\nCode: {error_details['code']}"
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    print("\n======= STARTING SERVER: Direct execution mode =======")
    print("Server is starting at http://0.0.0.0:8082")
    print("Press CTRL+C to stop the server")
    sys.stdout.flush()
    
    # Configure uvicorn to run with more visible logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")