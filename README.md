# Claude Mirror 🪞✨

**Use Claude Code with OpenAI, Azure, Databricks and More** 🤝

A proxy server that mirrors Claude Code's interface to OpenAI models, Azure OpenAI deployments, and other compatible providers. 🌉

## Installation

Clone the repository and install locally:

```bash
# Clone the repository
git clone https://github.com/ericmichael/claude-mirror.git
cd claude-mirror

# Run the install script
./install.sh

# Or manually install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

Once installed, run the interactive setup:

```bash
claude-mirror --setup
```

This will guide you through creating a configuration file with your API keys and model mappings. After setup, run:

```bash
claude-mirror
```

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Optional: Azure OpenAI or Databricks endpoints

### Setup 🛠️

1. **Install the package**:
   ```bash
   pip install claude-mirror
   ```

2. **Run the interactive setup**:
   ```bash
   claude-mirror --setup
   ```

   This will guide you through creating a configuration file at `~/.claude-mirror/config.yaml`.
   
   Alternatively, you can manually create a configuration file:
   
   ```yaml
   # Provider configuration
   providers:
     # OpenAI configuration
     openai:
       api_key: your_openai_api_key_here
     
     # Optional: Azure OpenAI configuration
     # azure:
     #   api_key: your_azure_api_key_here
     #   endpoint: your-instance.openai.azure.com
     #   api_version: 2023-05-15
     
     # Optional: Databricks configuration
     # databricks:
     #   token: your_databricks_token_here
     #   host: adb-12345678901234.12.azuredatabricks.net

   # Direct mapping from model categories to provider-specific models
   model_categories:
     # Claude models will map to these categories
     large: openai/gpt-4o          # Claude-3-Sonnet maps to this
     small: openai/gpt-4o-mini     # Claude-3-Haiku maps to this
   ```

3. **Run Claude Mirror**:
   ```bash
   claude-mirror
   ```

### Command-line Options

```bash
# Normal usage (no arguments needed)
claude-mirror

# Interactive setup to create/update config
claude-mirror --setup

# Debug mode with detailed logs
claude-mirror --debug  

# Use a specific config file
claude-mirror --config /path/to/my/config.yaml
```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **The claude-mirror command handles everything**:
   - Starts the proxy server
   - Configures the environment
   - Launches Claude Code connected to the proxy
   - Automatically shuts down the proxy when you exit Claude

4. **That's it!** Your Claude Code client will now use your configured models through the proxy. 🎯

## Model Mapping Rules 🗺️

The proxy follows strict rules with no fallbacks:

1. **Category-based Mapping**: 
   - `large` and `small` categories in `config.yaml` MUST be defined
   - Claude Sonnet models → map to the "large" category
   - Claude Haiku models → map to the "small" category

2. **Direct Model References**:
   - Each category maps directly to a provider/model in the format: `provider/model-name`
   - For example: `large: openai/gpt-4o` means Claude-3-Sonnet requests will use OpenAI's GPT-4o
   - No default fallbacks or silent provider selection

3. **Error Conditions**:
   - Missing required categories results in clear errors
   - Invalid model format results in clear errors
   - Missing configuration values in config.yaml result in clear errors

## Advanced Configuration Options

The configuration is based on two simple concepts:
1. **Providers**: Where to get models from (OpenAI, Azure, etc.)
2. **Model Categories**: How Claude models map to your chosen provider models

### OpenAI Configuration (Default)

The simplest setup uses OpenAI models:

```yaml
providers:
  openai:
    api_key: your_openai_api_key_here
    # Optional: base_url for alternative OpenAI-compatible APIs
    # base_url: http://localhost:8000/v1

model_categories:
  large: openai/gpt-4o
  small: openai/gpt-4o-mini
```

### Alternative OpenAI-Compatible APIs

You can use other OpenAI-compatible APIs by configuring a custom base URL:

```yaml
providers:
  openai:
    api_key: your_api_key_here
    base_url: http://localhost:8000/v1  # Example for LocalAI

model_categories:
  large: openai/local-model-name
  small: openai/smaller-local-model
```

This works with services like LocalAI, LM Studio, or any other API that implements the OpenAI interface.

### Azure OpenAI Configuration

To use Azure OpenAI Service:

```yaml
providers:
  azure:
    api_key: your_azure_api_key_here
    endpoint: your-instance.openai.azure.com
    api_version: 2023-05-15

model_categories:
  large: azure/my-gpt4-deployment-name
  small: azure/my-gpt35-deployment-name
```

### Databricks Configuration

To use Databricks:

```yaml
providers:
  databricks:
    token: your_databricks_token_here
    host: adb-12345678901234.12.azuredatabricks.net

model_categories:
  large: databricks/databricks-claude-3-sonnet
  small: databricks/databricks-claude-3-haiku
```

You can also mix providers if needed, for example using OpenAI for one category and Databricks for another.

## How It Works 🧩

This proxy works by:

1. **Loading configuration** directly from `config.yaml` with strict validation (no defaults)
2. **Mapping model names** according to explicit rules in the configuration
3. **Routing requests** to the appropriate provider based on model prefix
4. **Converting responses** back to Anthropic format

The proxy maintains compatibility with Claude Code while ensuring strict configuration control using a single configuration file with no environment variables.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁