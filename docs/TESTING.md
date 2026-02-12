<!--
TxGemma-MCP Testing Guide
Author: Tobias Neumann
License: MIT
Version: 0.1.1
Date: 2026-02-12
-->

# Testing Guide

> **Comprehensive test suite**: 447+ tests | **Coverage**: ~96%

---

## Table of Contents

- [Quick Start](#quick-start)
- [Test Organization](#test-organization)
- [Test Suite Overview](#test-suite-overview)
- [Running Tests](#running-tests)
- [Test Markers](#test-markers)
- [CI/CD](#cicd)
- [Where to Run GPU Tests](#where-to-run-gpu-tests)
- [Test Fixtures & Patterns](#test-fixtures--patterns)
- [Mocking Best Practices](#mocking-best-practices)
- [Performance Benchmarks](#performance-benchmarks)
- [Summary](#summary)

---

## Quick Start

### Fast Tests (No GPU Required)

```bash
# Run all fast tests (unit + integration, skip GPU)
uv run pytest -m "not gpu"

# With coverage report
uv run pytest -m "not gpu" --cov=txgemma --cov-report=html
open htmlcov/index.html
```

**Expected**: ~445 tests pass in ~3-4 seconds ⚡

### GPU Tests (Optional)

```bash
# Run GPU tests (requires GPU + models)
uv run pytest -m gpu -v

# Expected: ~2 tests pass in 30-90 seconds (after initial download)
```

---

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                     # Pytest configuration (--run-gpu flag)
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_validation.py          # 66 tests - Input validation & security
│   ├── test_executor.py            # 40 tests - Execution logic
│   ├── test_cache_utils.py         # 26 tests - Cache management
│   ├── test_config.py              # 50+ tests - Configuration system
│   └── test_chat_factory.py        # 25+ tests - Chat tool registration
├── integration/                    # Integration tests (real components)
│   ├── test_tool_factory.py        # 52 tests - Tool generation
│   ├── test_prompts.py             # 60+ tests - Prompt loading
│   ├── test_server.py              # 50+ tests - Server initialization 🛡️
│   └── test_analyze_tools.py       # 50+ tests - CLI tool
└── gpu/                            # GPU tests (requires GPU hardware)
    └── test_model.py               # 28 tests - Model loading & generation
```

---

## Test Suite Overview

| Module | Location | Tests | Coverage | Purpose | Speed |
|--------|----------|-------|----------|---------|-------|
| **validation.py** | `tests/unit/` | 66 | 100% | Input validation, SQL injection, path traversal, XSS, command injection protection | ⚡ <1s |
| **tool_factory.py** | `tests/integration/` | 52 | 97% | MCP tool generation, placeholder extraction, parameter normalization, tool filtering | ⚡ <1s |
| **executor.py** | `tests/unit/` | 40 | 96% | Tool execution, parameter mapping, whitespace handling, error handling | ⚡ <1s |
| **cache_utils.py** | `tests/unit/` | 26 | 100% | Global state management, cache reset, context manager overrides | ⚡ <1s |
| **prompts.py** | `tests/integration/` | 60+ | 97%+ | TDC prompt loading (local/HuggingFace), template formatting, filtering | ⚡ 1s |
| **config.py** | `tests/unit/` | 50+ | 96%+ | YAML config loading, environment variable overrides, singleton behavior | ⚡ <1s |
| **chat_factory.py** | `tests/unit/` | 25+ | 96%+ | Chat tool registration with FastMCP, error handling | ⚡ <1s |
| **server.py** | `tests/integration/` | 50+ | 95%+ | Server init, tool registration, **exec() security** 🛡️, resource endpoints | ⚡ 1s |
| **analyze_tools.py** | `tests/integration/` | 50+ | 91%+ | CLI tool for analyzing/filtering tools, argument parsing, JSON output | ⚡ 1s |
| **model.py** | `tests/gpu/` | 28 | 95%+ | Predict & chat model loading, singleton pattern, GPU generation | 🐌 30-90s |
| **TOTAL** | | **447+** | **~96%** | **Complete coverage of all core functionality** | **3-4s (no GPU)** |

### Key Features Tested

**Security** 🛡️:
- SQL injection, path traversal, command injection, XSS protection (validation.py)
- Code injection via exec() - malicious tool names & parameters (server.py)
- Input validation for all user-facing parameters

**Core Functionality**:
- Tool generation from TDC prompts with proper schema
- Dual model system (predict + chat)
- Configuration with environment variable priority
- Parameter mapping (normalized ↔ original names)
- Caching and global state management

**Integration**:
- FastMCP server initialization
- Resource endpoints (server info, tool lists)
- CLI tool for tool analysis
- HuggingFace prompt loading

---

## Running Tests

### By Category

```bash
# Unit tests only (fast, isolated)
pytest tests/unit/ -v

# Integration tests only (real components, no GPU)
pytest tests/integration/ -v

# GPU tests only (requires GPU)
pytest tests/gpu/ -v
pytest -m gpu -v  # Alternative

# All tests except GPU
pytest -m "not gpu" -v
```

### By Module

```bash
# Specific test file
pytest tests/unit/test_validation.py -v

# Specific test class
pytest tests/unit/test_validation.py::TestInputValidator -v

# Specific test
pytest tests/unit/test_validation.py::TestInputValidator::test_sql_injection_protection -v
```

### With Coverage

```bash
# Coverage for all fast tests
pytest -m "not gpu" --cov=txgemma --cov-report=html --cov-report=term
open htmlcov/index.html

# Coverage for specific module
pytest tests/unit/test_validation.py --cov=txgemma.validation --cov-report=term-missing
```

### Selective Test Running

```bash
# Run only fast tests (explicitly skip GPU)
uv run pytest -m "not gpu"

# Run only GPU tests (with flag)
uv run pytest -m gpu --run-gpu

# Run specific test class
uv run pytest tests/gpu/test_model.py::TestTxGemmaPredictModel --run-gpu

# Show test output (disable capture)
uv run pytest -v -s
```

---

## Test Markers

Tests use pytest markers for organization:

```python
@pytest.mark.unit          # Unit tests (fast, isolated)
@pytest.mark.integration   # Integration tests (real components)
@pytest.mark.gpu           # GPU tests (requires GPU hardware)
@pytest.mark.security      # Security-critical tests
```

### Using Markers

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only security tests
pytest -m security -v

# Run everything except GPU
pytest -m "not gpu" -v

# Run unit OR integration (exclude GPU)
pytest -m "unit or integration" -v
```

### Marker Configuration

Markers are defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (fast, mocked dependencies)",
    "integration: Integration tests (real dependencies, no GPU)",
    "gpu: GPU tests (requires CUDA)",
    "security: Security-critical tests",
]
```

---

## CI/CD

### GitHub Actions Workflow

**File**: `.github/workflows/tests.yml`

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests
- Manual workflow dispatch

**Jobs**:

#### 1. Unit Tests (No GPU) - Matrix: Python 3.11 & 3.12
```bash
# What runs
pytest -m "not gpu" --cov=txgemma --cov-report=xml --cov-report=term

# Duration: ~5 seconds
# Coverage: Uploaded to Codecov (if token set)
# Artifacts: Test results + coverage reports
```

**Runs on**: `ubuntu-latest`

#### 2. Code Quality (Linting)
```bash
# What runs
ruff check .
ruff format --check .

# Duration: ~3 seconds
```

**Runs on**: `ubuntu-latest`

#### 3. GPU Tests (EC2) - Optional
```bash
# What runs (only when enabled)
pytest --run-gpu --cov=txgemma

# Duration: ~5-15 min (first run) / ~60-90s (cached)
# Requires: ENABLE_GPU_TESTS=true variable
# Runs only on: main branch or manual dispatch
```

**Runs on**: `[self-hosted, gpu, linux]` (EC2 g5.xlarge with NVIDIA A10G)

**When it runs**:
- ✅ Pushed to `main` branch AND `ENABLE_GPU_TESTS=true`
- ✅ Manual workflow dispatch AND `ENABLE_GPU_TESTS=true`
- ❌ Pull requests (fast tests only)
- ❌ If `ENABLE_GPU_TESTS` not set or false

### Setting Up CI/CD

#### Minimum Setup (No GPU) - Works Out of the Box

Just commit and push - unit tests run automatically:
```bash
git add .
git commit -m "Your changes"
git push origin your-branch
```

✅ Unit tests run on Python 3.11 & 3.12  
✅ Linting checks run  
✅ No secrets required  
✅ Works for all contributors  

#### Full Setup (With GPU Tests)

**Prerequisites**:
1. AWS account with EC2 access
2. HuggingFace account with API token
3. (Optional) Codecov account for coverage tracking

**Required Secrets** (Settings → Secrets and variables → Actions):
- `HF_TOKEN` - HuggingFace API token (for model downloads)
- `CODECOV_TOKEN` - (Optional) Codecov upload token

**Required Variables** (Settings → Secrets and variables → Actions → Variables):
- `ENABLE_GPU_TESTS` - Set to `true` to enable GPU tests

**Self-Hosted Runner Setup**:

1. **Launch EC2 Instance**:
   - Type: `g5.xlarge` (NVIDIA A10G, 24GB VRAM)
   - AMI: Deep Learning AMI (Ubuntu 22.04)
   - Storage: 100GB EBS (for models)
   - Security: SSH access

2. **Configure Runner**:
   ```bash
   # On EC2 instance
   mkdir ~/actions-runner && cd ~/actions-runner
   
   # Download runner (check latest version at github.com/actions/runner/releases)
   curl -o actions-runner-linux-x64-2.313.0.tar.gz -L \
     https://github.com/actions/runner/releases/download/v2.313.0/actions-runner-linux-x64-2.313.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.313.0.tar.gz
   
   # Configure (get token from: Settings → Actions → Runners → New runner)
   ./config.sh \
     --url https://github.com/YOUR_USERNAME/txgemma-mcp \
     --token YOUR_RUNNER_TOKEN_FROM_GITHUB \
     --name ec2-gpu-runner \
     --labels self-hosted,gpu,linux \
     --work _work
   
   # Install as service
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

3. **Verify in GitHub**:
   - Go to **Settings** → **Actions** → **Runners**
   - You should see `ec2-gpu-runner` with status **Idle** (green)

4. **Enable GPU Tests**:
   - Go to **Settings** → **Secrets and variables** → **Actions** → **Variables**
   - Click **New repository variable**
   - Name: `ENABLE_GPU_TESTS`
   - Value: `true`

### Workflow Behavior

| Event | Fast Tests | Linting | GPU Tests |
|-------|------------|---------|-----------|
| Push to `develop` | ✅ Both Pythons | ✅ | ❌ |
| Push to `main` | ✅ Both Pythons | ✅ | ✅ (if enabled) |
| Pull request | ✅ Both Pythons | ✅ | ❌ |
| Manual dispatch | ✅ Both Pythons | ✅ | ✅ (if enabled) |

### Coverage Tracking with Codecov

**What you get** (if `CODECOV_TOKEN` is set):
- 📊 Coverage dashboard at `https://codecov.io/gh/YOUR_USERNAME/txgemma-mcp`
- 📈 Coverage trends over time
- 💬 PR comments showing coverage changes
- 🎯 Line-by-line coverage visualization

**Without Codecov**:
- Coverage still runs in CI
- Results shown in test logs
- No historical tracking
- No PR comments

**To view coverage locally**:
```bash
uv run pytest -m "not gpu" --cov=txgemma --cov-report=html
open htmlcov/index.html
```

### For Contributors (No GPU Access)

**Recommended workflow**:

```bash
# 1. Before committing - run fast tests
uv run pytest -m "not gpu"

# 2. Check linting
uv run ruff check --fix .
uv run ruff format .

# 3. Commit and push
git add .
git commit -m "Your changes"
git push origin your-branch

# 4. CI runs automatically
# - Unit tests on Python 3.11 & 3.12
# - Linting checks
# - Results visible in PR
```

✅ Fast feedback (~5 seconds)  
✅ No GPU needed  
✅ No secrets needed  

### For Maintainers (With GPU Access)

**Before merging to main**:

```bash
# 1. Run full test suite locally (if GPU available)
uv run pytest --run-gpu

# 2. Check everything passes
uv run pytest -m "not gpu" --cov=txgemma
uv run ruff check .

# 3. Merge to main
git checkout main
git merge your-branch
git push origin main

# 4. Monitor CI
# - GitHub Actions tab shows all jobs
# - GPU tests run on EC2 (if enabled)
# - Coverage uploaded to Codecov
```

### Managing the Self-Hosted Runner

**Check runner status**:
```bash
# On EC2
sudo ./svc.sh status

# View logs
sudo journalctl -u actions.runner.* -f
```

**Stop/Start runner**:
```bash
# Stop
sudo ./svc.sh stop

# Start
sudo ./svc.sh start

# Restart
sudo ./svc.sh restart
```

**Remove runner**:
```bash
# On EC2
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token YOUR_REMOVAL_TOKEN_FROM_GITHUB
```

### CI/CD Summary

**Current configuration**:
- ✅ Fast tests: Always run (~5 seconds)
- ✅ Linting: Always run (~3 seconds)
- ✅ GPU tests: Optional (~60-90 seconds when enabled, after initial download)
- ✅ Python 3.11 & 3.12 matrix testing
- ✅ Coverage reporting (Codecov optional)

**Typical CI run times**:
- Pull request (no GPU): ~8-10 seconds
- Push to main (with GPU enabled): ~60-90 seconds (after initial download)
- First GPU test run: ~5-15 minutes (model download)

---

## Where to Run GPU Tests

### Option 1: Local Development with Apple Silicon (Recommended)

If you have a Mac with M1/M2/M3 chip:

```bash
# One-time setup
uv sync --all-extras

# Run GPU tests (uses Metal Performance Shaders)
uv run pytest --run-gpu
```

**Requirements**:
- Apple Silicon Mac (M1/M2/M3)
- 32GB+ RAM recommended (for dual model tests)
- Automatic GPU via Metal (no setup needed!)

### Option 2: Local Development with NVIDIA GPU

```bash
# One-time setup
uv sync --all-extras
uv run huggingface-cli login

# Run GPU tests
uv run pytest --run-gpu
```

**Requirements**:
- CUDA GPU with 24GB+ VRAM (for default dev config: 2b + 9b)
- CUDA 12.1+ and cuDNN installed

### Option 3: GitHub Actions with Self-Hosted EC2 Runner

Our CI/CD uses a self-hosted EC2 g5.xlarge instance:

```yaml
# Runs automatically on main branch pushes (when enabled)
gpu-tests:
  runs-on: [self-hosted, gpu, linux]
  steps:
    - run: uv run pytest --run-gpu
```

**EC2 Instance Specs**:
- Type: g5.xlarge
- GPU: NVIDIA A10G (24GB VRAM)
- Storage: 100GB EBS
- Cost: ~$1/hour

**Setup Instructions**:
1. Launch EC2 g5.xlarge instance with Deep Learning AMI
2. Install GitHub Actions runner
3. Set `ENABLE_GPU_TESTS=true` in repository variables
4. Push to main branch

### Option 4: Skip GPU Tests in Development

Most practical workflow:

```bash
# Development cycle (fast)
uv run pytest  # GPU tests auto-skipped

# Before PR (fast tests only)
uv run pytest -m "not gpu" --cov=txgemma

# Let CI handle GPU tests on EC2
# (runs automatically when merged to main)
```

---

## Test Fixtures & Patterns

### autouse Fixtures

Automatic test isolation:

```python
@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset config before/after each test."""
    reset_config()
    yield
    reset_config()
```

**Used in**: `test_config.py`, `test_cache_utils.py`

### Context Managers

Temporary state overrides:

```python
with ParameterMappingOverride(test_mapping):
    # Test with custom mapping
    result = get_cached_parameter_mapping()
# Original mapping restored automatically
```

**Used in**: `test_cache_utils.py`

### Dependency Injection

Clean mocking without patches:

```python
# Function accepts optional dependencies
def execute_tool(tool_name, args, _loader=None, _model=None):
    loader = _loader or get_loader()  # Use injected or default
    # ...

# Test injects mock
def test_execution(mock_loader):
    result = execute_tool("tool", {}, _loader=mock_loader)
```

**Used in**: `test_executor.py`

---

## Mocking Best Practices

### 1. Patch at the Source

```python
# ❌ Wrong: Patch where imported
@patch("txgemma.executor.InputValidator")

# ✅ Right: Patch where defined
@patch("txgemma.validation.InputValidator")
```

### 2. Use Fixtures for Reusable Mocks

```python
@pytest.fixture
def mock_loader():
    loader = Mock()
    loader.get.return_value = mock_template
    return loader

def test_with_mock(mock_loader):
    # Use the fixture
    result = function_using_loader(mock_loader)
```

### 3. Verify Behavior, Not Implementation

```python
# ✅ Good: Test behavior
assert result == expected_output

# ❌ Bad: Test implementation details
mock.some_internal_method.assert_called()
```

---

## Performance Benchmarks

### Test Execution Times

| Test Suite | Count | Time (No GPU) | Time (With GPU) |
|------------|-------|---------------|-----------------|
| test_validation.py | 66 | ~1s | ~1s |
| test_tool_factory.py | 52 | ~1s | ~1s |
| test_executor.py | 40 | ~1s | ~1s |
| test_cache_utils.py | 26 | <1s | <1s |
| test_prompts.py | 60+ | ~1s | ~1s |
| test_config.py | 50+ | ~1s | ~1s |
| test_chat_factory.py | 25+ | <1s | <1s |
| test_server.py | 50+ | ~1s | ~1s |
| test_analyze_tools.py | 50+ | ~1s | ~1s |
| **Fast Total** | **~419** | **~3-4s** | **~3-4s** |
| test_model.py | 28 | skipped | ~5-15min (first) / ~30-90s (cached) |
| **All Tests** | **~447** | **~3-4s** | **~5-15min (first) / ~33-94s (cached)** |

**Note**: GPU test time depends on config:
- Development (2b + 9b): ~5min first / ~30-60s cached
- Production (9b + 9b): ~10min first / ~60-75s cached
- Research (27b + 27b): ~15min first / ~75-90s cached

### Hardware Performance

| Platform | GPU | Config | First Run | Cached Run |
|----------|-----|--------|-----------|------------|
| Apple M1 Pro | MPS (Metal) | Dev (2b+9b) | ~3min | ~30s |
| Apple M2 Max | MPS (Metal) | Dev (2b+9b) | ~2.5min | ~25s |
| Apple M3 Max | MPS (Metal) | Prod (9b+9b) | ~5min | ~40s |
| AWS g5.xlarge | A10G (24GB) | Dev (2b+9b) | ~5min | ~60s |
| Local NVIDIA RTX 3090 | CUDA (24GB) | Dev (2b+9b) | ~4min | ~45s |
| AWS g5.2xlarge | A10G (32GB) | Prod (9b+9b) | ~8min | ~75s |

---

## Summary

### Quick Reference

| Command | Use Case | Time | GPU Required |
|---------|----------|------|--------------|
| `uv run pytest` | Fast tests (default) | ~3-4s | ❌ |
| `uv run pytest --run-gpu` | All tests with GPU | ~5-15min / ~30-90s | ✅ |
| `uv run pytest -m "not gpu"` | Explicitly skip GPU | ~3-4s | ❌ |
| `uv run pytest -m gpu --run-gpu` | Only GPU tests | ~5-15min / ~30-90s | ✅ |
| `uv run pytest tests/unit/` | Unit tests only | ~1s | ❌ |
| `uv run pytest tests/integration/` | Integration tests only | ~2s | ❌ |

### Test Distribution

- **~419 tests**: No GPU required (unit + integration)
- **~28 tests**: GPU required (model loading & generation)
- **Total**: ~447 tests
- **Coverage**: ~96% average across all modules
- **Security**: 100% coverage on security-critical code

### Development Cycle

1. **Write code** → Run `uv run pytest` (fast tests)
2. **Test specific module** → Run `uv run pytest tests/unit/test_validation.py`
3. **Before commit** → Run `uv run pytest -m "not gpu" --cov=txgemma`
4. **Push to PR** → CI runs fast tests automatically
5. **Merge to main** → CI runs GPU tests on EC2 (if enabled)