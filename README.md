# 🧬 TxGemma-MCP

[![CI Tests](https://github.com/t-neumann/TxGemma-MCP/actions/workflows/tests.yml/badge.svg)](https://github.com/t-neumann/TxGemma-MCP/actions/workflows/tests.yml)

**TxGemma-MCP** is a Model Context Protocol (MCP) server exposing Google DeepMind's TxGemma therapeutic AI models for agentic drug discovery workflows.

TxGemma is a family of open language models fine-tuned from Gemma 2, specifically designed for therapeutic property prediction across molecules, proteins, and clinical outcomes.

---

## 🚀 Features

* **Dynamic Tool Generation**: Tools are auto-generated from TDC prompt definitions downloaded from HuggingFace
* **Lazy Model Loading**: Model only loads when first tool is called (fast startup)
* **GPU Optimized**: Designed for efficient GPU memory usage
* **Dual Transport**: FastMCP powers both stdio (MCP) and SSE (web API) modes
* **Auto-Downloading Prompts**: TDC prompts automatically downloaded from HuggingFace on first run
* **Flexible Filtering**: Load all tools or filter by placeholder, complexity, or use case
* **10+ Prediction Tools**: Toxicity, BBB permeability, clinical trials, retrosynthesis, and more

---

## 📁 Architecture

```
txgemma-mcp/
├── server.py                 # FastMCP entrypoint (stdio + SSE)
├── txgemma/
│   ├── __init__.py
│   ├── model.py              # TxGemma model wrapper (lazy-loaded singleton)
│   ├── tool_factory.py       # Auto-generate MCP tools from JSON
│   ├── executor.py           # Execute tool calls with model
│   └── prompts.py            # Load & validate TDC prompt templates from HF
├── tests/
│   └── test_tools.py
└── pyproject.toml
```

### Key Design Principles

1. **Separation of Concerns**: MCP layer, model management, and prompt handling are separate
2. **HuggingFace Integration**: Prompts auto-downloaded from `google/txgemma-2b-predict`
3. **Lazy Loading**: Model (~5GB) only loads when first prediction is requested
4. **Singleton Pattern**: One model instance shared across all tool calls
5. **Schema Inference**: Input schemas auto-generated from prompt placeholders
6. **Dual Transport**: FastMCP provides both stdio (MCP) and SSE (web API) modes

---

## 🧩 Installation

### Prerequisites

* **Python ≥ 3.11**
* **GPU recommended** (CUDA or MPS) - TxGemma models are large (2B-27B parameters)
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


## 🧬 Available Tools

Tools are defined in `data/tdc_prompts.json` and auto-generated on server startup:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `predict_toxicity` | `Drug SMILES` | Predict toxicity of a drug molecule |
| `predict_bbb_permeability` | `Drug SMILES` | Check if drug crosses blood-brain barrier |
| `predict_drug_target_interaction` | `Drug SMILES`, `Target sequence` | Predict binding to protein target |
| `predict_solubility` | `Drug SMILES` | Predict aqueous solubility |
| `predict_clearance` | `Drug SMILES` | Predict drug clearance rate |
| `predict_clinical_trial_outcome` | `Drug SMILES`, `Trial phase`, `Indication` | Predict trial success |
| `retrosynthesis` | `Product SMILES` | Generate reactants for synthesis |
| `predict_bioavailability` | `Drug SMILES` | Predict oral bioavailability |
| `predict_lipophilicity` | `Molecule SMILES` | Predict logP value |
| `predict_adverse_events` | `Drug SMILES` | Predict potential adverse reactions |

**Note**: Tools are automatically generated from TDC prompts in the TxGemma HuggingFace repository. The exact list depends on what's available in `google/txgemma-2b-predict/tdc_prompts.json`.

---

## 📝 Adding New Tools

### Option 1: Wait for Official Updates (Recommended)

TxGemma prompts are maintained by Google in the HuggingFace model repository. When they add new tasks, they'll automatically be available next time you run the server (prompts are cached, so delete cache to force update).

### Option 2: Local Override

Create a local `data/tdc_prompts.json` file to add custom tools:

```json
{
  "your_new_tool": {
    "template": "Instruction: Your instruction.\nContext: Background info.\nQuestion: Question with {placeholder}?\nAnswer:",
    "metadata": {
      "description": "Tool description for MCP",
      "category": "admet"
    }
  }
}
```

Then modify `txgemma/prompts.py` to use local override:

```python
loader = PromptLoader(
    local_override=Path("data/tdc_prompts.json")
)
```

The tool will be auto-generated with:
- Name: `your_new_tool`
- Input schema: auto-detected from `{placeholder}` variables
- Description: from metadata or Context line

---

## 🧪 Example Usage

### Programmatically

```python
from txgemma import execute_tool

result = execute_tool(
    "predict_toxicity",
    {"drug_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
)
print(result)
```

---

## 🧰 Development

### Run Tests

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest -v

# With coverage
uv run pytest --cov=txgemma --cov-report=html
```

### Lint and Format

```bash
uv run ruff check txgemma tests server.py
uv run ruff format txgemma tests server.py
```

---

## 🏗️ Architecture Details

### Model Loading Strategy

- **Lazy Loading**: Model loads on first `generate()` call, not at import
- **Singleton**: One model instance shared across all requests
- **Device Auto-Detection**: Automatically uses CUDA > MPS > CPU

### Prompt Flow

```
1. Client calls tool → server.py
2. server.py → executor.py
3. executor.py → prompts.py (load template)
4. executor.py → model.py (generate prediction)
5. Result → client
```

### Memory Management

- Model: ~5GB GPU memory (2B model)
- First generation: ~30 seconds (model download + load)
- Subsequent generations: ~1-2 seconds

---

## ⚙️ Configuration

### Change Model Size

Edit `txgemma/model.py`:

```python
def __init__(
    self,
    model_name: str = "google/txgemma-9b-predict",  # or 27b
    max_new_tokens: int = 64,
):
```

Available models:
- `google/txgemma-2b-predict` (fastest, 5GB)
- `google/txgemma-9b-predict` (balanced, 18GB)
- `google/txgemma-27b-predict` (most accurate, 54GB)

### Adjust Generation Parameters

Edit `txgemma/model.py` `generate()`:

```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=128,  # Generate more tokens
    do_sample=True,      # Enable sampling
    temperature=0.7,     # Adjust randomness
)
```

---

## 🚀 Deployment

### Docker (Coming Soon)

```bash
docker build -t txgemma-mcp .
docker run --gpus all -p 8000:8000 txgemma-mcp
```

### Production Considerations

- Use GPU instances (AWS g4dn, g5, p3)
- Consider model caching to persistent volume
- Implement request queuing for high load
- Add telemetry/monitoring

---

## 📚 Resources

* [TxGemma Documentation](https://developers.google.com/health-ai-developer-foundations/txgemma)
* [TxGemma Paper (arXiv)](https://arxiv.org/abs/2504.06196)
* [Model Context Protocol](https://modelcontextprotocol.io)
* [Therapeutic Data Commons](https://tdcommons.ai)

---

## ⚠️ Limitations

* **GPU Required**: Models need significant GPU memory (5-54GB depending on size)
* **First Load**: Initial model download and load takes time
* **Deterministic**: Using `do_sample=False` for reproducible predictions
* **Context Length**: Limited by model's context window

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

* Google DeepMind for TxGemma models
* Therapeutic Data Commons for training data
* Anthropic for MCP specification

---

**Author**: Tobias Neumann  
**Version**: 0.1.0