# 🧬 TxGemma-MCP

<div align="center">

<!-- Badges -->
<p align="center">
  <a href="https://github.com/t-neumann/TxGemma-MCP/actions/workflows/tests.yml">
    <img src="https://github.com/t-neumann/TxGemma-MCP/actions/workflows/tests.yml/badge.svg" alt="CI Tests">
  </a>
  <a href="https://github.com/t-neumann/TxGemma-MCP/releases">
    <img src="https://img.shields.io/github/v/release/t-neumann/TxGemma-MCP?include_prereleases&label=version" alt="Version">
  </a>
  <a href="https://github.com/t-neumann/TxGemma-MCP/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/t-neumann/TxGemma-MCP" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg" alt="Python">
  </a>
  <a href="docs/TESTING.md">
    <img src="https://img.shields.io/badge/coverage-96%25-brightgreen.svg" alt="Coverage">
  </a>
  <a href="https://github.com/jlowin/fastmcp">
    <img src="https://img.shields.io/badge/MCP-FastMCP-orange.svg" alt="FastMCP">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://hub.docker.com/r/tobneu/txgemma-mcp">
    <img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker">
  </a>
</p>

**TxGemma-MCP** is a Model Context Protocol (MCP) server exposing Google DeepMind's TxGemma therapeutic AI models for agentic drug discovery workflows.

[Features](#-features) •
[Installation](#-installation) •
[Configuration](#️-configuration) •
[Docker](#-docker-deployment) •
[Testing](#-testing) •
[Architecture](#-architecture-details)

</div>

---

## 🚀 Features

* **Dual Models**: Prediction model for fast TDC tasks + Chat model for explanations
* **Configuration-Driven**: Control models, tools, and behavior via `config.yaml`
* **Dynamic Tool Generation**: 700+ tools auto-generated from TDC prompts
* **Advanced Tool Filtering**: Filter by placeholder, complexity, and regex patterns
* **Lazy Model Loading**: Models load on first use (fast startup)
* **Security Hardened**: Input validation, SQL injection protection, XSS prevention
* **GPU Optimized**: Efficient memory usage with FP16
* **Dual Transport**: FastMCP powers both stdio (MCP) and streamable-http (web API) modes
* **Environment Overrides**: Override config with environment variables
* **Production-Ready**: 447+ tests, 96% coverage, comprehensive CI/CD

---

## 📁 Architecture

```
txgemma-mcp/
├── config.yaml               # Main configuration file
├── server.py                 # FastMCP entrypoint
├── scripts/
│   └── analyze_tools.py      # CLI tool for exploring available tools
├── txgemma/
│   ├── __init__.py           # Package exports with lazy loading
│   ├── config.py             # Configuration loader with env overrides
│   ├── model.py              # Predict + Chat model singletons
│   ├── chat_factory.py       # Chat tool registration
│   ├── tool_factory.py       # Auto-generate TDC tools from prompts
│   ├── executor.py           # Execute tool calls with models
│   ├── prompts.py            # Load TDC prompts from HuggingFace
│   ├── validation.py         # Input validation & security (NEW)
│   └── cache_utils.py        # Global state management (NEW)
├── tests/
│   ├── unit/                 # Unit tests (fast, mocked)
│   │   ├── test_validation.py      # 66 tests - Security
│   │   ├── test_executor.py        # 40 tests
│   │   ├── test_cache_utils.py     # 26 tests
│   │   ├── test_config.py          # 50+ tests
│   │   └── test_chat_factory.py    # 25+ tests
│   ├── integration/          # Integration tests (real components)
│   │   ├── test_tool_factory.py    # 52 tests
│   │   ├── test_prompts.py         # 60+ tests
│   │   ├── test_server.py          # 50+ tests (with security!)
│   │   └── test_analyze_tools.py   # 50+ tests - CLI
│   └── gpu/                  # GPU tests (optional)
│       └── test_model.py           # 28 tests
├── docs/
│   ├── TESTING.md            # Comprehensive testing guide
│   └── FILTERING.md          # Tool filtering guide
└── pyproject.toml
```

### Key Design Principles

1. **Configuration-First**: All runtime settings in `config.yaml`
2. **Dual Models**: Fast predictions + conversational explanations
3. **Smart Filtering**: Load only what you need (Drug SMILES by default)
4. **Lazy Loading**: Models load only when needed
5. **Security by Design**: Input validation, injection protection, secure exec()
6. **Singleton Pattern**: One instance per model type
7. **Environment Overrides**: Config can be overridden via env vars
8. **Dual Transport**: FastMCP provides stdio (MCP) and streamable-http (web API)
9. **Test-Driven**: 96% coverage, comprehensive test suite

---

## 🧩 Installation

### Prerequisites

* **Python 3.11 or 3.12** (both tested in CI)
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
uv sync --all-extras

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
  
  # Filter by placeholder 
  filter_placeholder: "Drug SMILES"  # Only load drug tools
  
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
  exclude_name_pattern: "^(ToxCast|Tox21)"  # Exclude ToxCast overload of tools for Agents
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

# Exclude tool patterns
export TXGEMMA_EXCLUDE_NAME_PATTERN="^ToxCast"

# Run server
uv run fastmcp run server.py
```

**Priority**: Environment variables > `config.yaml` > defaults

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
  # Option 1: Filter by single placeholder (most common)
  filter_placeholder: "Drug SMILES"  # Only drug-development tools
  
  # Option 2: Filter by multiple placeholders
  filter_placeholders: ["Drug SMILES", "Target sequence"]
  match_all: true  # Require ALL placeholders (AND logic)
  
  # Option 3: Fuzzy matching
  filter_placeholder: "sequence"
  exact_match: false  # Matches "Target sequence", "Protein sequence", etc.
  
  # Option 4: Limit complexity
  filter_placeholder: "Drug SMILES"
  max_placeholders: 2  # Only simple tools
  
  # Option 5: Exclude by regex pattern (NEW!)
  filter_placeholder: "Drug SMILES"
  exclude_name_pattern: "^ToxCast"  # Exclude ToxCast tools
  
  # Option 6: Complex combinations
  filter_placeholder: "Drug SMILES"
  max_placeholders: 2
  exclude_name_pattern: "^(ToxCast|Tox21)"
  
  # Option 7: Load all tools (slow, not recommended)
  filter_placeholder: null  # All 700+ tools
```

**Why filter?** Loading all tools can take 10-30 seconds and may overwhelm LLM agents with too many choices. Filtering to Drug SMILES covers the majority of molecular property prediction use cases.

**For detailed filtering options, examples, and CLI usage**, see [FILTERING.md](docs/FILTERING.md)

---

## 🔍 Exploring Available Tools

**NEW in v0.1.1**: `analyze_tools.py` CLI for exploring the tool catalog!

```bash
# List all placeholders with usage counts
python scripts/analyze_tools.py --list-placeholders

# Show all Drug SMILES tools
python scripts/analyze_tools.py --placeholder "Drug SMILES"

# Fuzzy search for sequence-related tools
python scripts/analyze_tools.py --placeholder "sequence" --fuzzy

# Show simple tools only (≤2 parameters)
python scripts/analyze_tools.py --simple

# Exclude ToxCast tools
python scripts/analyze_tools.py --exclude "^ToxCast"

# Combine filters: Drug SMILES + simple + no ToxCast
python scripts/analyze_tools.py --placeholder "Drug SMILES" --simple --exclude "^ToxCast"

# Export to JSON
python scripts/analyze_tools.py --json > tools.json

# Show template details
python scripts/analyze_tools.py --template "tdc_ClinTox_predict"

# Get help
python scripts/analyze_tools.py --help
```

See [docs/FILTERING.md](docs/FILTERING.md) for complete CLI documentation and examples.

---

## 🧬 Available Tools

### Prediction Tools

**700+ TDC prediction tools** for molecular properties. The exact number and types depend on your filtering configuration. 

**Default**: With `filter_placeholder: "Drug SMILES"`, ~400-500 tools are loaded (excludes protein/sequence-based tools).

**Note**: Use `exclude_name_pattern: "^ToxCast"` to remove the bulk of regulatory assays to not overload agents.

### Chat Tool (Configurable)

**`txgemma_chat`** - Conversational Q&A about drug discovery

**Note:** Enabled by default, disable via `tools.enable_chat: false` in config.yaml.

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
# Override models and filtering at runtime
docker run -d --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e TXGEMMA_PREDICT_MODEL=google/txgemma-27b-predict \
  -e TXGEMMA_CHAT_MODEL=google/txgemma-27b-chat \
  -e TXGEMMA_CHAT_MAX_TOKENS=500 \
  -e TXGEMMA_EXCLUDE_NAME_PATTERN="^ToxCast" \
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
npx @modelcontextprotocol/inspector --transport http --server-url http://localhost:8000/mcp
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

## Testing

### Quick Start

```bash
# Run all fast tests (no GPU required)
uv run pytest -m "not gpu"

# With coverage report
uv run pytest -m "not gpu" --cov=txgemma --cov-report=html
open htmlcov/index.html
```

### Test Suite Overview

| Module | Tests | Coverage | Purpose |
|--------|-------|----------|---------|
| **validation.py** | 66 | 100% | Input validation, SQL injection, XSS protection |
| **tool_factory.py** | 52 | 97% | Tool generation, parameter normalization |
| **executor.py** | 40 | 96% | Tool execution, parameter mapping |
| **cache_utils.py** | 26 | 100% | Global state management |
| **prompts.py** | 60+ | 97%+ | TDC prompt loading (local/HuggingFace) |
| **config.py** | 50+ | 96%+ | Configuration with env overrides |
| **chat_factory.py** | 25+ | 96%+ | Chat tool registration |
| **server.py** | 50+ | 95%+ | Server init, **exec() security** 🛡️ |
| **analyze_tools.py** | 50+ | 91%+ | CLI tool analysis |
| **model.py** | 28 | 95%+ | Model loading (GPU tests) |
| **TOTAL** | **447+** | **~96%** | **Production-ready!** |

**Runtime**: ~3-4 seconds (without GPU)

### Run Specific Tests

```bash
# By category
pytest tests/unit/ -v              # Unit tests only
pytest tests/integration/ -v       # Integration tests only

# By module
pytest tests/unit/test_validation.py -v
pytest tests/integration/test_server.py -v

# Security tests
pytest -m security -v

# GPU tests (requires GPU)
pytest -m gpu -v
```

### CI/CD

GitHub Actions runs:
- **Linting** (ruff) - Python 3.11 & 3.12
- **Unit tests** - Fast, mocked dependencies
- **Integration tests** - Real components, no GPU
- **GPU tests** (optional) - On self-hosted EC2 runner
- **Coverage reporting** - Uploaded to Codecov

**For complete testing documentation**, see [TESTING.md](docs/TESTING.md)

---

## 🧰 Development

### Lint and Format

```bash
# Check linting
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Code Quality

- **Ruff** for linting and formatting
- **Type hints** throughout codebase
- **Security checks** (exec() usage validated)
- **Import sorting** and organization
- **Docstrings** for all public APIs

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
validation.py (input validation) ← NEW!
    ↓
executor.py (execute_tool or execute_chat)
    ↓
prompts.py (load template) + model.py (generate)
    ↓
Result → Client
```

### Security Architecture

1. **Input Validation** (`validation.py`):
   - SQL injection prevention
   - Path traversal protection
   - Command injection prevention
   - XSS protection
   - SMILES string validation

2. **Server Security** (`server.py`):
   - Safe `exec()` usage with validation
   - Code injection prevention
   - Malicious input rejection

3. **Parameter Security** (`executor.py`):
   - Parameter name normalization
   - Whitespace stripping
   - Type validation

**All security-critical code has 100% test coverage** 🛡️

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
  -e TXGEMMA_EXCLUDE_NAME_PATTERN="^(ToxCast|Tox21)" \
  -p 8000:8000 \
  tobneu/txgemma-mcp:latest
```

### Scaling

- Use GPU instances (AWS g5, g4dn, p3)
- Implement request queuing for high load
- Consider model serving frameworks (vLLM, TGI)
- Cache frequently used predictions
- Use tool filtering to reduce initialization time

### Monitoring

```bash
# Check loaded tools
docker logs <container-id> 2>&1 | grep "Loaded.*tools"

# Check excluded tools
docker logs <container-id> 2>&1 | grep "Excluded.*matching pattern"

# Verify configuration
docker logs <container-id> 2>&1 | grep "configured"
```

---

## 📚 Resources

* [TxGemma Documentation](https://developers.google.com/health-ai-developer-foundations/txgemma)
* [TxGemma Paper (arXiv)](https://arxiv.org/abs/2504.06196)
* [Model Context Protocol](https://modelcontextprotocol.io)
* [Therapeutic Data Commons](https://tdcommons.ai)
* [FastMCP Documentation](https://github.com/jlowin/fastmcp)

**Project Documentation:**
* [Testing Guide](docs/TESTING.md) - Comprehensive test suite documentation
* [Filtering Guide](docs/FILTERING.md) - Tool filtering options and examples

---

## ⚠️ Limitations

* **GPU Required**: Models need 8-64GB VRAM depending on size
* **First Load**: Initial download and load takes time
* **Context Length**: Limited by model's context window (~8K tokens)
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

# Check exclusion pattern
docker logs <container-id> 2>&1 | grep "Excluded"

# Load all tools (slower)
export TXGEMMA_FILTER_PLACEHOLDER=null
```

### Tool Filtering Not Working

```bash
# Verify pattern is correct
python scripts/analyze_tools.py --exclude "^ToxCast" --json | jq '.[].name'

# Check logs for exclusion
docker logs <container-id> 2>&1 | grep "Excluded.*tools matching pattern"
```

---

## 📋 Changelog

### v0.1.1 (2026-02-12)

**🎉 Major Release: Security, Testing, and Filtering Improvements**

**New Features:**
- ✨ Advanced tool filtering with regex patterns (`exclude_name_pattern`)
- ✨ Multiple placeholder filtering with AND/OR logic
- ✨ `analyze_tools.py` CLI for exploring tool catalog
- ✨ Comprehensive input validation and security hardening

**Security:**
- 🛡️ SQL injection protection
- 🛡️ Path traversal protection
- 🛡️ Command injection prevention
- 🛡️ XSS protection
- 🛡️ Safe `exec()` usage with validation
- 🛡️ SMILES string validation

**Testing:**
- ✅ 447+ comprehensive tests (was ~170)
- ✅ 96% average coverage (was ~85%)
- ✅ Security tests for all critical paths
- ✅ Python 3.11 & 3.12 CI matrix testing
- ✅ GPU test suite with self-hosted runner support

**Improvements:**
- ⚡ Faster test suite (~3-4s for all fast tests)
- 📝 Comprehensive documentation (TESTING.md, FILTERING.md)
- 🔧 Better error messages and logging
- 🎯 Improved tool filtering performance
- 🐛 Fixed parameter mapping edge cases

**Infrastructure:**
- 🔄 Improved CI/CD with caching
- 📊 Coverage reporting to Codecov
- 🎨 Ruff linting and formatting
- 🐳 Updated Docker configuration

### v0.1.0 (Initial Release)

- Initial release with dual model support
- Basic tool filtering
- Configuration system
- Docker deployment
- FastMCP integration

---

## 🙏 Acknowledgments

* **Google DeepMind** for TxGemma models
* **Therapeutic Data Commons** for training data and benchmarks
* **Anthropic** for Model Context Protocol specification and Claude
* **FastMCP** project for MCP server framework
* **Astral** for uv and ruff tools

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Author:</strong> Tobias Neumann<br>
  <strong>Version:</strong> 0.1.1<br>
</p>