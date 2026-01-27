# 🧬 TxGemma-MCP

<div align="center">

[![CI Tests](https://github.com/t-neumann/TxGemma-MCP/actions/workflows/tests.yml/badge.svg)](https://github.com/t-neumann/TxGemma-MCP/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/tobneu/txgemma-mcp)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

**TxGemma-MCP** is a Model Context Protocol (MCP) server exposing Google DeepMind's TxGemma therapeutic AI models for agentic drug discovery workflows.

[Features](#-features) •
[Installation](#-installation) •
[Configuration](#️-configuration) •
[Docker](#-docker-deployment) •
[Contributing](#-Development) •
[Architecture](#-Architecture)

</div>

---

## 🚀 Features

* **Dual Models**: Prediction model for fast TDC tasks + Chat model for explanations
* **Configuration-Driven**: Control models, tools, and behavior via `config.yaml`
* **Dynamic Tool Generation**: Tools auto-generated from TDC prompts
* **Smart Tool Filtering**: Load only Drug SMILES tools by default (fast, focused)
* **Lazy Model Loading**: Models load on first use (fast startup)
* **GPU Optimized**: Efficient memory usage with FP16
* **Dual Transport**: FastMCP powers both stdio (MCP) and streamable-http (web API) modes
* **Environment Overrides**: Override config with environment variables

---

## 📁 Architecture

```
txgemma-mcp/
├── config.yaml               # Main configuration file
├── server.py                 # FastMCP entrypoint
├── txgemma/
│   ├── __init__.py
│   ├── config.py             # Configuration loader 
│   ├── model.py              # Predict + Chat model singletons
│   ├── chat_factory.py       # Chat tool registration 
│   ├── tool_factory.py       # Auto-generate TDC tools from prompts
│   ├── executor.py           # Execute tool calls with models
│   └── prompts.py            # Load TDC prompts from HuggingFace
├── tests/
│   ├── test_model.py
│   ├── test_executor.py
│   ├── test_chat_factory.py
│   ├── test_tool_factory.py
│   ├── test_config.py 
│   └── test_server.py
└── pyproject.toml
```

### Key Design Principles

1. **Configuration-First**: All runtime settings in `config.yaml`
2. **Dual Models**: Fast predictions + conversational explanations
3. **Smart Defaults**: Drug SMILES tools only (fast, focused)
4. **Lazy Loading**: Models load only when needed
5. **Singleton Pattern**: One instance per model type
6. **Environment Overrides**: Config can be overridden via env vars
7. **Dual Transport**: FastMCP provides stdio (MCP) and streamable-http (web API)

---

## 🧩 Installation

### Prerequisites

* **Python ≥ 3.11**
* **GPU recommended** (CUDA or MPS) - Models are 2B-27B parameters
* **uv** (package manager)
* **HuggingFace account** (for model access)

### Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/t-neumann/TxGemma-MCP.git
cd TxGemma-MCP

# 3. Install dependencies
uv sync

# 4. Login to HuggingFace
uv run huggingface-cli login
```

**Important**: Accept TxGemma terms on HuggingFace:  
https://huggingface.co/google/txgemma-2b-predict

---

## ⚙️ Configuration

TxGemma-MCP is configured via `config.yaml`. The default configuration is optimized for development (fast, low VRAM).

### Default Configuration

```yaml
# Prediction Model (for TDC tasks)
predict:
  model: "google/txgemma-2b-predict"
  max_new_tokens: 64

# Chat Model (for explanations)
chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 100

# Tool Loading
tools:
  prompts:
    filename: "tdc_prompts.json"
    # Prompts are auto-downloaded from predict model repo
  
  # Only load Drug SMILES tools (recommended)
  filter_placeholder: "Drug SMILES"
  
  # Enable conversational chat tool
  enable_chat: true
```

### Configuration Presets

#### Development (Default - 22GB VRAM)
```yaml
predict:
  model: "google/txgemma-2b-predict"
chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 100
tools:
  filter_placeholder: "Drug SMILES"
```

#### Production (36GB VRAM)
```yaml
predict:
  model: "google/txgemma-9b-predict"
chat:
  model: "google/txgemma-9b-chat"
  max_new_tokens: 200
tools:
  filter_placeholder: "Drug SMILES"
```

#### Research (54GB+ VRAM)
```yaml
predict:
  model: "google/txgemma-27b-predict"
chat:
  model: "google/txgemma-27b-chat"
  max_new_tokens: 500
tools:
  filter_placeholder: null  # Load all tools
```

### Environment Variable Overrides

Override config without editing files:

```bash
# Override models
export TXGEMMA_PREDICT_MODEL=google/txgemma-9b-predict
export TXGEMMA_CHAT_MODEL=google/txgemma-27b-chat

# Override chat response length
export TXGEMMA_CHAT_MAX_TOKENS=500

# Load all tools instead of filtering
export TXGEMMA_FILTER_PLACEHOLDER=null

# Run server
uv run fastmcp run server.py
```

### Available Models

| Model | Size | VRAM | Speed | Accuracy | Use Case |
|-------|------|------|-------|----------|----------|
| `google/txgemma-2b-predict` | ~4GB | 8GB | ⚡⚡⚡ | ⭐⭐ | Development |
| `google/txgemma-9b-predict` | ~18GB | 24GB | ⚡⚡ | ⭐⭐⭐ | Production |
| `google/txgemma-27b-predict` | ~54GB | 64GB | ⚡ | ⭐⭐⭐⭐ | Research |
| `google/txgemma-9b-chat` | ~18GB | 24GB | ⚡⚡ | ⭐⭐⭐ | Explanations |
| `google/txgemma-27b-chat` | ~54GB | 64GB | ⚡ | ⭐⭐⭐⭐ | Detailed explanations |

### Tool Filtering Options

```yaml
tools:
  # Option 1: Filter by placeholder (recommended)
  filter_placeholder: "Drug SMILES"  # Only drug-development tools, fast
  
  # Option 2: Load all tools (slow)
  filter_placeholder: null  # All available tools
  
  # Option 3: Limit complexity
  filter_placeholder: "Drug SMILES"
  max_placeholders: 2  # Only simple tools
  
  # Option 4: Use local prompts for testing
  prompts:
    local_override: "/path/to/custom_prompts.json"
```

**Why filter?** Loading all tools can take 10-30 seconds and may overwhelm LLM agents with too many choices. Filtering to Drug SMILES covers the majority of molecular property prediction use cases.

**For detailed filtering options and examples**, see [FILTERING](docs/FILTERING.md)

---

## 🧬 Available Tools

### Prediction Tools

TDC prediction tools for molecular properties. The exact number and types of tools depend on what's available in the TxGemma model repository. Below are **examples** of available tool categories:

| Category | Example Tools |
|----------|--------------|
| **Toxicity** | `tdc_ClinTox_predict`, `tdc_hERG_predict` |
| **ADME** | `tdc_BBB_Martins_predict`, `tdc_Clearance_Hepatocyte_AZ_predict` |
| **Binding** | `tdc_BindingDB_Kd_predict`, `tdc_DAVIS_predict` |
| **Solubility** | `tdc_ESOL_predict`, `tdc_AqSolDB_predict` |
| **Clinical** | Various phase-specific predictions |

**Note**: With `filter_placeholder: "Drug SMILES"` (default), only tools requiring drug SMILES are loaded. This covers the majority of molecular property prediction tasks and provides faster startup. Set to `null` in config.yaml to load all available tools.

### Chat Tool (Configurable)

**`txgemma_chat`** - Conversational Q&A about drug discovery

**Note:** This tool is enabled by default but can be disabled via `tools.enable_chat: false` in config.yaml.

Example queries:
```json
{"question": "Why might aspirin cause stomach bleeding?"}
{"question": "What makes a good blood-brain barrier penetrant drug?"}
{"question": "Explain the mechanism of action for CC(=O)OC1=CC=CC=C1C(=O)O"}
```

---

## 🐳 Docker Deployment

### Build


```bash
docker buildx build --platform linux/amd64 -t tobneu/txgemma-mcp:latest --push .
```

### Deployment

```bash
# Create cache directory
mkdir -p ~/.cache/huggingface

docker run -d --gpus all \
  --restart unless-stopped \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  tobneu/txgemma-mcp:latest

# Check logs
docker logs -f <container-id>

# Verify config
docker logs <container-id> 2>&1 | grep "configured"
```

### Override Config in Docker

```bash
# Override models at runtime
docker run -d --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e TXGEMMA_PREDICT_MODEL=google/txgemma-27b-predict \
  -e TXGEMMA_CHAT_MODEL=google/txgemma-27b-chat \
  -e TXGEMMA_CHAT_MAX_TOKENS=500 \
  -p 8000:8000 \
  tobneu/txgemma-mcp:latest
```

---

## 🧪 Example Usage

### Via MCP Protocol

Use with Claude Desktop, Cline, or any MCP client:

```json
{
  "mcpServers": {
    "txgemma": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "-e",
        "HF_TOKEN",
        "tobneu/txgemma-mcp:latest"
      ]
    }
  }
}
```

### Via HTTP API

```bash
# Start server with streamable-http transport
uv run fastmcp run server.py --transport streamable-http

# Use MCP Inspector
npx @modelcontextprotocol/inspector ---transport http --server-url http://localhost:8000/mcp

```

### Programmatically

```python
from txgemma import execute_tool, execute_chat

# Predict toxicity
result = execute_tool(
    "tdc_ClinTox_predict",
    {"Drug SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}
)
print(f"Toxicity: {result}")

# Get explanation
explanation = execute_chat(
    "Why might aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) cause stomach bleeding?"
)
print(f"Explanation: {explanation}")
```

---

## 🧰 Development

### Run Tests

```bash
# All tests (fast, no GPU)
uv run pytest -v

# With GPU tests
uv run pytest --run-gpu -v

# Specific test file
uv run pytest tests/test_config.py -v

# With coverage
uv run pytest --cov=txgemma --cov-report=html
```

### Lint and Format

```bash
# Check linting
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff
```

### CI/CD

GitHub Actions runs:
- Linting (ruff)
- Type checking (mypy)
- Tests (pytest)
- GPU tests (on self-hosted runner)

---

## 🏗️ Architecture Details

### Model Loading Strategy

- **Lazy Loading**: Models load on first `generate()` call
- **Singleton**: One instance per model type (predict/chat)
- **Configuration**: Models determined by `config.yaml` or env vars
- **Device Auto-Detection**: CUDA > MPS > CPU

### Configuration Priority

1. **Explicit arguments** (testing/overrides)
2. **Environment variables** (`TXGEMMA_*`)
3. **Config file** (`config.yaml`)
4. **Hardcoded defaults** (fallback)

### Prompt Flow

```
Client Request
    ↓
server.py (FastMCP)
    ↓
executor.py (execute_tool or execute_chat)
    ↓
prompts.py (load template) + model.py (generate)
    ↓
Result → Client
```

### Memory Management

**Development (2b + 9b):**
- Predict model: ~4GB
- Chat model: ~18GB
- Total: ~22GB VRAM

**Production (9b + 9b):**
- Predict model: ~18GB
- Chat model: ~18GB
- Total: ~36GB VRAM

**First Generation:**
- Model download: ~10-60 seconds (one-time)
- Model load: ~10-30 seconds
- Generation: ~1-5 seconds

**Subsequent Generations:**
- ~1-2 seconds (predict)
- ~2-5 seconds (chat)

---

## 📝 Adding Custom Tools

### Option 1: Wait for Official Updates (Recommended)

TxGemma prompts are maintained by Google. New tasks auto-appear when they're added to the HuggingFace repo.

### Option 2: Local Override

Create `custom_prompts.json`:

```json
{
  "your_tool_name": {
    "template": "Instruction: Your instruction.\nContext: Background.\nQuestion: {Your Placeholder}?\nAnswer:",
    "metadata": {
      "description": "Tool description",
      "category": "custom"
    }
  }
}
```

Update `config.yaml`:

```yaml
tools:
  prompts:
    local_override: "/path/to/custom_prompts.json"
```

The tool auto-generates with:
- Name from JSON key
- Input schema from `{placeholders}`
- Description from metadata

---

## 🚀 Production Best Practices

### Security

```bash
# Use secrets manager for HF_TOKEN
docker run -d --gpus all \
  --restart unless-stopped \
  -e HF_TOKEN=$(aws secretsmanager get-secret-value ...) \
  -e HF_HOME=/root/.cache/huggingface \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  tobneu/txgemma-mcp:latest

```

### Scaling

- Use GPU instances (AWS g5, g4dn, p3)
- Implement request queuing for high load
- Consider model serving frameworks (vLLM, TGI)
- Cache frequently used predictions

---

## 📚 Resources

* [TxGemma Documentation](https://developers.google.com/health-ai-developer-foundations/txgemma)
* [TxGemma Paper (arXiv)](https://arxiv.org/abs/2504.06196)
* [Model Context Protocol](https://modelcontextprotocol.io)
* [Therapeutic Data Commons](https://tdcommons.ai)
* [FastMCP Documentation](https://github.com/jlowin/fastmcp)

---

## ⚠️ Limitations

* **GPU Required**: Models need 8-64GB VRAM depending on size
* **First Load**: Initial download and load takes time
* **Context Length**: Limited by model's context window
* **Rate Limits**: HuggingFace Hub has download limits

---

## 🐛 Troubleshooting

### Config Not Loading

```bash
# Check config exists
ls -la config.yaml

# Verify environment variables
printenv | grep TXGEMMA

# Check Docker logs
docker logs <container-id> 2>&1 | grep -i config
```

### Models Not Changing

```bash
# Environment variable names need TXGEMMA_ prefix
export TXGEMMA_PREDICT_MODEL=google/txgemma-9b-predict  # ✅ Correct
export PREDICT_MODEL=google/txgemma-9b-predict          # ❌ Wrong

# Verify config loaded
docker logs <container-id> 2>&1 | grep "configured"
```

### Out of Memory

```bash
# Use smaller models
export TXGEMMA_PREDICT_MODEL=google/txgemma-2b-predict
export TXGEMMA_CHAT_MODEL=google/txgemma-9b-chat

# Or reduce chat length
export TXGEMMA_CHAT_MAX_TOKENS=100
```

### Tools Not Loading

```bash
# Check filter setting
docker logs <container-id> 2>&1 | grep "filter"

# Load all tools (slower)
export TXGEMMA_FILTER_PLACEHOLDER=null
```

---

## 🙏 Acknowledgments

* **Google DeepMind** for TxGemma models
* **Therapeutic Data Commons** for training data and benchmarks
* **Anthropic** for Model Context Protocol specification
* **FastMCP** project for MCP server framework

---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Author:</strong> Tobias Neumann<br>
  <strong>Version:</strong> 0.1.0
</p>