# Testing Guide

## Test Organization

Tests are organized by GPU requirements:

```
tests/
├── conftest.py           # Pytest configuration
├── test_prompts.py       # ✅ Fast, no GPU (60+ tests)
├── test_tool_factory.py  # ✅ Fast, no GPU
├── test_executor.py      # ✅ Fast, mocked model
├── test_model.py         # ⚠️  Requires GPU (~5GB download)
└── test_integration.py   # ⚠️  Requires GPU (e2e tests)
```

## Running Tests

### Quick Tests (No GPU Required)

Run fast tests without model loading:

```bash
# All fast tests
uv run pytest

# With coverage
uv run pytest --cov=txgemma --cov-report=html

# Specific test file
uv run pytest tests/test_prompts.py -v

# Specific test
uv run pytest tests/test_prompts.py::TestPromptLoader::test_load_from_local_simple_format -v
```

**Expected time**: ~5-10 seconds

### GPU Tests (Requires GPU + Model Download)

Run tests that require model loading:

```bash
# All GPU tests
uv run pytest --run-gpu

# GPU tests with coverage
uv run pytest --run-gpu --cov=txgemma

# Slow tests (includes model loading)
uv run pytest --run-slow

# Both GPU and slow tests
uv run pytest --run-gpu --run-slow
```

**Expected time**: 
- First run: ~5-10 minutes (model download)
- Subsequent runs: ~30-60 seconds (cached model)

### Selective Test Running

```bash
# Run only unit tests (fast)
uv run pytest -m "not gpu and not slow"

# Run only GPU tests
uv run pytest -m gpu --run-gpu

# Run only slow tests
uv run pytest -m slow --run-slow

# Skip specific markers
uv run pytest -m "not gpu"
```

### CI/CD Tests

For continuous integration (no GPU):

```bash
# Fast tests only - perfect for CI
uv run pytest -m "not gpu and not slow" --cov=txgemma

# Alternative: explicitly skip
uv run pytest --ignore=tests/test_model.py --ignore=tests/test_integration.py
```

## Test Markers

Tests use pytest markers to indicate requirements:

- `@pytest.mark.gpu` - Requires GPU hardware
- `@pytest.mark.slow` - Takes >5 seconds to run
- `@pytest.mark.integration` - End-to-end integration test
- `@pytest.mark.unit` - Fast unit test

## Where to Run GPU Tests

### Option 1: Local Development (Recommended if you have GPU)

```bash
# One-time setup
uv sync --all-extras
uv run huggingface-cli login

# Run GPU tests
uv run pytest --run-gpu --run-slow
```

**Requirements**:
- CUDA GPU with 8GB+ VRAM (for 2B model)
- Or Apple Silicon Mac with 16GB+ RAM (MPS)

### Option 2: GitHub Actions with Self-Hosted Runner

Set up a self-hosted runner with GPU:

```yaml
# .github/workflows/test-gpu.yml
name: GPU Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run GPU tests
        run: uv run pytest --run-gpu --run-slow
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

### Option 3: Cloud GPU (AWS, GCP, etc.)

Launch a GPU instance and run tests:

```bash
# AWS EC2 g4dn.xlarge or similar
ssh gpu-instance

# Setup
git clone https://github.com/your-username/txgemma-mcp.git
cd txgemma-mcp
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install and test
uv sync --all-extras
uv run huggingface-cli login
uv run pytest --run-gpu --run-slow

# Terminate instance when done
```

### Option 4: Skip GPU Tests in CI

Most practical for open-source projects:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run fast tests only
        run: uv run pytest -m "not gpu and not slow" --cov=txgemma --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

## Test Strategy

### Fast Tests (Always Run)

✅ **test_prompts.py** - 60+ tests
- Prompt loading from local files
- Template formatting
- Placeholder discovery
- Filtering logic
- No model required

✅ **test_tool_factory.py** - Tool building
- Schema generation
- Filtering logic
- Tool metadata
- No model required

✅ **test_executor.py** - Execution logic
- Mock the model
- Test error handling
- Test prompt formatting
- No model required

### GPU Tests (Run Manually or on GPU CI)

⚠️ **test_model.py** - Model loading and generation
- Model initialization
- Loading/unloading
- Text generation
- Edge cases
- Requires GPU + 5GB download

⚠️ **test_integration.py** - End-to-end tests
- Full pipeline: prompt → model → result
- Real TDC prompts
- Actual predictions
- Requires GPU

## Writing New Tests

### For Non-GPU Code

```python
# tests/test_my_feature.py
import pytest

def test_my_feature():
    """Fast test without GPU."""
    # Your test here
    assert True
```

### For GPU-Required Code

```python
# tests/test_my_gpu_feature.py
import pytest

@pytest.mark.gpu
@pytest.mark.slow
class TestMyGPUFeature:
    """Tests requiring GPU."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load model once for all tests."""
        from txgemma.model import TxGemmaModel
        model = TxGemmaModel()
        model.load()
        yield model
        model.unload()
    
    def test_with_model(self, loaded_model):
        """Test using real model."""
        result = loaded_model.generate("Test prompt")
        assert isinstance(result, str)
```

### With Mocked Model

```python
# tests/test_my_feature.py
import pytest
from unittest.mock import Mock, patch

@patch('my_module.get_model')
def test_with_mocked_model(mock_get_model):
    """Test without loading real model."""
    mock_model = Mock()
    mock_model.generate.return_value = "Mocked result"
    mock_get_model.return_value = mock_model
    
    # Your test here
    result = my_function()
    assert result == "Mocked result"
```

## Best Practices

### 1. Prefer Mocking for Unit Tests

```python
# ✅ Good - fast, no GPU
@patch('txgemma.executor.get_model')
def test_executor(mock_model):
    mock_model.return_value.generate.return_value = "result"
    # Test logic

# ❌ Avoid - slow, needs GPU
def test_executor():
    model = TxGemmaModel()
    model.load()  # Takes minutes!
```

### 2. Use Class-Scoped Fixtures for GPU Tests

```python
# ✅ Good - load model once
@pytest.fixture(scope="class")
def loaded_model():
    model = TxGemmaModel()
    model.load()  # Once per class
    yield model
    model.unload()

# ❌ Avoid - loads model per test
@pytest.fixture
def loaded_model():
    model = TxGemmaModel()
    model.load()  # Every test!
    yield model
    model.unload()
```

### 3. Mark GPU Tests Explicitly

```python
# ✅ Good - clear markers
@pytest.mark.gpu
@pytest.mark.slow
def test_generation():
    pass

# ❌ Avoid - unmarked GPU test
def test_generation():
    model = TxGemmaModel()
    model.load()  # Will fail in CI!
```

### 4. Skip Gracefully When No GPU

```python
@pytest.fixture
def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

def test_gpu_feature(skip_if_no_gpu):
    # This test needs GPU
    pass
```

## Coverage

### Generate Coverage Report

```bash
# Fast tests only (for CI)
uv run pytest -m "not gpu and not slow" --cov=txgemma --cov-report=html

# All tests (if GPU available)
uv run pytest --run-gpu --run-slow --cov=txgemma --cov-report=html

# Open report
open htmlcov/index.html
```

### Coverage Goals

- **Fast tests**: Should cover 80%+ of non-model code
- **GPU tests**: Should cover model and integration paths
- **Combined**: Aim for 90%+ total coverage

## Debugging Tests

### Run with Verbose Output

```bash
uv run pytest -vv tests/test_prompts.py
```

### Run with Logging

```bash
uv run pytest -v --log-cli-level=INFO
```

### Run Single Test with PDB

```bash
uv run pytest tests/test_prompts.py::test_name --pdb
```

### Show Print Statements

```bash
uv run pytest -v -s
```

## Continuous Integration

### Recommended CI Strategy

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  # Fast tests on every commit
  fast-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run fast tests
        run: uv run pytest -m "not gpu and not slow" --cov=txgemma
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
  
  # GPU tests on self-hosted runner (optional)
  gpu-tests:
    runs-on: [self-hosted, gpu]
    if: github.ref == 'refs/heads/main'  # Only on main branch
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run GPU tests
        run: uv run pytest --run-gpu --run-slow
```

## Summary

### Quick Reference

| Command | Use Case | Time | GPU |
|---------|----------|------|-----|
| `pytest` | Fast tests | ~5s | ❌ |
| `pytest --run-gpu` | GPU tests | ~30s+ | ✅ |
| `pytest --run-slow` | Slow tests | ~30s+ | Maybe |
| `pytest -m "not gpu"` | Skip GPU | ~5s | ❌ |
| `pytest --cov=txgemma` | With coverage | ~10s | ❌ |

### Test Distribution

- **~60 tests**: No GPU required (prompts, tool_factory, executor)
- **~15 tests**: GPU required (model, integration)
- **Total**: ~75 tests

### Recommended Workflow

1. **Development**: Run fast tests frequently (`pytest`)
2. **Before commit**: Run all fast tests with coverage
3. **Before merge**: Run GPU tests if available
4. **CI**: Only fast tests (practical for most projects)
5. **Release**: Run all tests on GPU machine

---

**Need help?** Check `tests/conftest.py` for fixture details or run `pytest --markers` to see all available markers.