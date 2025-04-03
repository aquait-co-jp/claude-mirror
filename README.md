# Claude Code but with OpenAI Models 🧙‍♂️🔄 ¯\\_(ツ)_/¯

**Use Claude Code with OpenAI Models** 🤝

A proxy server that lets you use Claude Code with OpenAI models like GPT-4o / gpt-4.5 and o3-mini. 🌉


![Claude Code but with OpenAI Models](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Optional: Azure OpenAI or Databricks endpoints

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install UV**:
   ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Configure `config.yaml` and `.env`**:

   Strict configuration is required with no default fallbacks. The system uses YAML configuration with clear model mapping.

   **Required: Create `config.yaml`**:
   
   ```yaml
   # Model Provider Configuration - Define all providers you want to use
   providers:
     openai:
       api_key: ${OPENAI_API_KEY}
     # Optional providers:
     # anthropic:
     #   api_key: ${ANTHROPIC_API_KEY}
     # azure:
     #   api_key: ${AZURE_OPENAI_API_KEY}
     #   endpoint: ${AZURE_OPENAI_ENDPOINT}
     #   api_version: ${AZURE_OPENAI_API_VERSION}
     # databricks:
     #   token: ${DATABRICKS_TOKEN}
     #   host: ${DATABRICKS_HOST}

   # Model Category Mapping - BOTH categories MUST be defined
   model_categories:
     big:  # Used for claude-3-sonnet models
       provider: openai  # Which provider to use
       deployment: gpt-4o  # Specific model/deployment name
     
     small:  # Used for claude-3-haiku models
       provider: openai  # Which provider to use
       deployment: gpt-4o-mini  # Specific model/deployment name
   ```

   **Required: Create `.env`** with any API keys referenced in `config.yaml`:
   
   ```
   OPENAI_API_KEY=sk-your-openai-key
   ```

4. **Start the proxy server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082
   ```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use your configured models through the proxy. 🎯

## Model Mapping Rules 🗺️

The proxy follows strict rules with no fallbacks:

1. **Category-based Mapping**: 
   - `big` and `small` categories in `config.yaml` MUST be defined
   - Claude Sonnet models → map to the "big" category
   - Claude Haiku models → map to the "small" category

2. **Direct Model References**:
   - Use categories directly: `model="big"` or `model="small"`
   - Use explicit provider prefixes: `model="openai/gpt-4o"` or `model="azure/deployment-name"`
   - No default fallbacks or silent provider selection

3. **Error Conditions**:
   - Missing required categories results in clear errors
   - Models without provider prefixes result in clear errors
   - Environment variables missing from .env result in clear errors

## Advanced Configuration Options

The configuration is based on two simple concepts:
1. **Providers**: Where to get models from (OpenAI, Azure, etc.)
2. **Model Categories**: How Claude models map to your chosen provider models

### OpenAI Configuration (Default)

The simplest setup uses OpenAI models:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}

model_categories:
  big:
    provider: openai
    deployment: gpt-4o
  small:
    provider: openai
    deployment: gpt-4o-mini
```

### Azure OpenAI Configuration

To use Azure OpenAI Service:

```yaml
providers:
  azure:
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_version: ${AZURE_OPENAI_API_VERSION}

model_categories:
  big:
    provider: azure
    deployment: my-gpt4-deployment-name
  small:
    provider: azure
    deployment: my-gpt35-deployment-name
```

### Databricks Configuration

To use Databricks:

```yaml
providers:
  databricks:
    token: ${DATABRICKS_TOKEN}
    host: ${DATABRICKS_HOST}

model_categories:
  big:
    provider: databricks
    deployment: databricks-claude-3-sonnet
  small:
    provider: databricks
    deployment: databricks-claude-3-haiku
```

You can also mix providers if needed, for example using OpenAI for one category and Databricks for another.

## How It Works 🧩

This proxy works by:

1. **Loading configuration** with strict validation (no defaults)
2. **Mapping model names** according to explicit rules in `config.yaml`
3. **Routing requests** to the appropriate provider based on model prefix
4. **Converting responses** back to Anthropic format

The proxy maintains compatibility with Claude Code while ensuring strict configuration control and no silent fallbacks.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁