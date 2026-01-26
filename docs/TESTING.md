# Testing Guide

## Test Organization

Tests are organized by GPU requirements:

```
tests/
├── conftest.py           # Pytest configuration (--run-gpu flag)
├── test_prompts.py       # ✅ Fast, no GPU (60+ tests)
├── test_tool_factory.py  # ✅ Fast, no GPU (31 tests)
├── test_config.py        # ✅ Fast, no GPU (32 tests)
├── test_chat_factory.py  # ✅ Fast, mocked model (10+ tests)
├── test_executor.py      # ✅ Fast, mocked model (15+ tests)
├── test_server.py        # ✅ Fast, no GPU (26 tests)
└── test_model.py         # ⚠️  Requires GPU (~8-36GB download)
```

## Running Tests

### Quick Tests (No GPU Required)

Run fast tests without model loading:

```bash
# All fast tests (default - GPU tests skipped)
uv run pytest

# With coverage
uv run pytest --cov=txgemma --cov-report=html

# Specific test file
uv run pytest tests/test_config.py -v

# Specific test
uv run pytest tests/test_config.py::TestConfigModels::test_predict_config_defaults -v
```

**Expected time**: ~10-15 seconds  
**Coverage**: ~85%+ of non-model code

### GPU Tests (Requires GPU + Model Download)

Run tests that require model loading:

```bash
# All GPU tests
uv run pytest --run-gpu

# GPU tests with coverage
uv run pytest --run-gpu --cov=txgemma

# Specific GPU test
uv run pytest tests/test_model.py --run-gpu -v
```

**Expected time**: 
- First run: ~5-15 minutes (model download from HuggingFace - depends on config)
- Subsequent runs: ~30-90 seconds (cached models)
- Apple Silicon (M1/M2/M3): Fast! Uses Metal Performance Shaders

**Note**: GPU tests will load models based on your `config.yaml` or default configuration. Development config (2b + 9b) downloads ~22GB.

### Selective Test Running

```bash
# Run only fast tests (explicitly skip GPU)
uv run pytest -m "not gpu"

# Run only GPU tests (with flag)
uv run pytest -m gpu --run-gpu

# Run specific test class
uv run pytest tests/test_model.py::TestTxGemmaPredictModel --run-gpu

# Run configuration tests
uv run pytest tests/test_config.py -v

# Show test output (disable capture)
uv run pytest -v -s --run-gpu
```

### CI/CD Tests

For continuous integration (no GPU):

```bash
# Fast tests only - perfect for CI
uv run pytest -m "not gpu" --cov=txgemma --cov-report=xml

# Alternative: run without --run-gpu flag (GPU tests auto-skipped)
uv run pytest --cov=txgemma
```

## Test Markers

Tests use pytest markers to indicate requirements:

- `@pytest.mark.gpu` - Requires GPU hardware (CUDA or MPS)

**Note**: Use `@pytest.mark.gpu` for any test requiring model loading. All other tests run by default and are automatically skipped when --run-gpu is not provided.

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

## Test Strategy

### Fast Tests (Always Run) - ~170+ tests

✅ **test_config.py** (32 tests) - Configuration system
- Config loading from YAML
- Environment variable overrides
- Pydantic validation
- Singleton behavior
- No model required

✅ **test_prompts.py** (60+ tests) - Prompt template system
- Loading from local files and HuggingFace
- Template formatting and validation
- Placeholder discovery and filtering
- Config integration
- No model required

✅ **test_tool_factory.py** (31 tests) - MCP tool generation
- Schema generation from TDC prompts
- Tool filtering and introspection
- Metadata handling
- Config integration
- No model required

✅ **test_executor.py** (15+ tests) - Execution logic
- Mock the model for fast tests
- Test error handling for both predict and chat
- Test prompt formatting
- No model required

✅ **test_chat_factory.py** (10+ tests) - Chat tool registration
- Chat tool registration with FastMCP
- Error handling
- Config integration (enable_chat)
- No model required

✅ **test_server.py** (26 tests) - Server initialization
- FastMCP setup
- Tool registration
- Config integration
- Resource endpoints
- Mocked components

### GPU Tests (Run on GPU Hardware) - ~20+ tests

⚠️ **test_model.py** - Model loading and generation
- Both predict and chat models
- Model initialization from config
- Singleton pattern
- Loading/unloading on GPU
- Text generation with real models
- Edge cases and error handling
- Requires GPU + model downloads (size depends on config)

## New Test Coverage (v0.1.0)

### Configuration System (test_config.py)

```python
# Test config models
def test_predict_config_defaults()
def test_chat_config_defaults()
def test_prompts_config_defaults()
def test_tools_config_defaults()

# Test loading
def test_load_config_from_yaml()
def test_load_config_with_nested_prompts()
def test_load_config_file_not_exists()

# Test environment overrides
def test_env_override_predict_model()
def test_env_override_chat_model()
def test_env_override_chat_max_tokens()
def test_env_override_filter_placeholder()
def test_multiple_env_overrides()

# Test use cases
def test_development_config()
def test_production_config()
def test_research_config()
```

### Chat System (test_chat_factory.py)

```python
# Test chat tool registration
def test_register_chat_tool()
def test_chat_tool_schema()
def test_chat_execution()
def test_chat_error_handling()
def test_enable_chat_config()
```

### Dual Model System (test_model.py - Updated)

```python
# Predict model tests
def test_predict_model_singleton()
def test_predict_model_config_integration()
def test_predict_generate()

# Chat model tests (NEW)
def test_chat_model_singleton()
def test_chat_model_config_integration()
def test_chat_generate()
def test_chat_template_handling()
```

## Test Configuration

### Custom Command-Line Flags

Defined in `conftest.py`:

- `--run-gpu` - Enable GPU tests (default: skip)

### pytest.ini Configuration

```ini
[pytest]
markers =
    gpu: marks tests as requiring GPU

asyncio_mode = auto
```

### Environment Variables for Testing

```bash
# Override config during tests
export TXGEMMA_PREDICT_MODEL=google/txgemma-2b-predict
export TXGEMMA_CHAT_MODEL=google/txgemma-9b-chat
export TXGEMMA_CHAT_MAX_TOKENS=100
export TXGEMMA_FILTER_PLACEHOLDER="Drug SMILES"

# Run tests with custom config
uv run pytest --run-gpu
```

## Coverage

### Generate Coverage Report

```bash
# Fast tests only (for CI)
uv run pytest -m "not gpu" --cov=txgemma --cov-report=html

# All tests (if GPU available)
uv run pytest --run-gpu --cov=txgemma --cov-report=html

# Open report
open htmlcov/index.html
```

### Coverage Goals

- **Fast tests**: Should cover 85%+ of non-model code
- **GPU tests**: Should cover model and generation paths
- **Combined**: Aim for 90%+ total coverage

### Coverage by Module

| Module | Fast Tests | GPU Tests | Total |
|--------|-----------|-----------|-------|
| config.py | 100% | - | 100% |
| prompts.py | 95% | - | 95% |
| tool_factory.py | 95% | - | 95% |
| chat_factory.py | 90% | - | 90% |
| executor.py | 85% | 5% | 90% |
| server.py | 90% | - | 90% |
| model.py | 30% | 65% | 95% |

## Debugging Tests

### Verbose Output

```bash
# Show test names and results
uv run pytest -v

# Extra verbose with test docstrings
uv run pytest -vv

# Show print statements (disable capture)
uv run pytest -s

# Combine flags
uv run pytest -vv -s --run-gpu
```

### Debug with Logging

```bash
# Show log output at INFO level
uv run pytest --log-cli-level=INFO

# Show debug logs
uv run pytest --log-cli-level=DEBUG

# See config loading messages
uv run pytest tests/test_config.py --log-cli-level=INFO -v
```

### Debug with PDB

```bash
# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger on specific test
uv run pytest tests/test_config.py::TestConfigModels::test_predict_config_defaults --pdb
```

### Run Single Test

```bash
# Run one test function
uv run pytest tests/test_config.py::TestEnvironmentVariableOverrides::test_env_override_predict_model -v

# Run one test class
uv run pytest tests/test_model.py::TestTxGemmaPredictModel --run-gpu -v
```

## Continuous Integration (GitHub Actions)

### Workflow Overview

Our CI/CD pipeline (`.github/workflows/tests.yml`):

```yaml
jobs:
  # 1. Unit Tests - Every commit (Ubuntu, no GPU, Python 3.11 & 3.12)
  test:
    runs-on: ubuntu-latest
    steps:
      - run: uv run pytest -m "not gpu" --cov=txgemma

  # 2. Linting - Every commit
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  # 3. GPU tests - Only on main branch (EC2 self-hosted)
  gpu-tests:
    runs-on: [self-hosted, gpu, linux]
    if: vars.ENABLE_GPU_TESTS == 'true'
    steps:
      - run: uv run pytest --run-gpu
```

### What Runs When

| Event | Unit Tests | Lint | GPU Tests |
|-------|------------|------|-----------|
| PR to any branch | ✅ Python 3.11 & 3.12 | ✅ | ❌ |
| Push to develop | ✅ Python 3.11 & 3.12 | ✅ | ❌ |
| Push to main | ✅ Python 3.11 & 3.12 | ✅ | ✅ (if enabled) |
| Manual trigger | ✅ Python 3.11 & 3.12 | ✅ | ✅ (if enabled) |

### Required Setup

#### 1. GitHub Secrets (Required)

Add these secrets to your repository:

**Settings → Secrets and variables → Actions → Secrets**

##### `HF_TOKEN` (Required for GPU tests)
Your HuggingFace token for downloading TxGemma models.

**How to get it:**
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `txgemma-mcp-ci`
4. Type: **Read** (sufficient for downloading)
5. Click **"Generate token"**
6. Copy the token (starts with `hf_...`)

**How to add to GitHub:**
1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Name: `HF_TOKEN`
4. Value: Paste your `hf_...` token
5. Click **"Add secret"**

**Or via GitHub CLI:**
```bash
gh secret set HF_TOKEN
# Paste your token when prompted
```

##### `CODECOV_TOKEN` (Optional)
Token for uploading coverage reports to Codecov.io

**How to get it:**
1. Sign up at https://codecov.io with your GitHub account
2. Add your repository
3. Copy the token from repository settings

**How to add to GitHub:**
```bash
gh secret set CODECOV_TOKEN
# Paste your Codecov token when prompted
```

**Note:** If `CODECOV_TOKEN` is not set, coverage upload step will be skipped gracefully (won't fail CI).

#### 2. GitHub Variables (Optional - for GPU tests)

**Settings → Secrets and variables → Actions → Variables tab**

##### `ENABLE_GPU_TESTS` (Optional)
Controls whether GPU tests run on main branch.

**How to enable:**
1. Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
2. Click **"New repository variable"**
3. Name: `ENABLE_GPU_TESTS`
4. Value: `true`
5. Click **"Add variable"**

**Default behavior:**
- **Not set**: GPU tests are skipped (CI passes without GPU runner)
- **Set to `true`**: GPU tests run on main branch (requires self-hosted runner)
- **Set to `false`**: GPU tests are skipped

**This allows you to:**
- ✅ Set up CI immediately without GPU runner
- ✅ Enable GPU tests later when runner is ready
- ✅ Temporarily disable GPU tests without removing the workflow

### Setting Up Self-Hosted GPU Runner (Optional)

GPU tests require a self-hosted runner with GPU. This is optional - you can run the project without it.

#### Prerequisites

- AWS Account (or other cloud provider)
- EC2 g5.xlarge instance (or similar GPU instance)
  - GPU: NVIDIA A10G (24GB VRAM)
  - Storage: 100GB EBS
  - Cost: ~$1/hour
- Deep Learning AMI (Ubuntu) with CUDA pre-installed

#### Step 1: Launch EC2 Instance

```bash
# Launch EC2 g5.xlarge with Deep Learning AMI
# Security Group: Allow SSH (port 22) from your IP
# Key pair: Create or use existing
# Storage: 100GB gp3
```

#### Step 2: Install Docker and NVIDIA Container Toolkit

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Verify GPU
nvidia-smi

# Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

#### Step 3: Install GitHub Actions Runner

**Get runner token from GitHub:**
1. Go to your repo → **Settings** → **Actions** → **Runners**
2. Click **"New self-hosted runner"**
3. Choose **Linux** and **x64**
4. Copy the token shown (needed for next step)

**On your EC2 instance:**

```bash
# Create runner directory
mkdir ~/actions-runner && cd ~/actions-runner

# Download runner
curl -o actions-runner-linux-x64-2.313.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.313.0/actions-runner-linux-x64-2.313.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.313.0.tar.gz

# Configure (use the token from GitHub)
./config.sh \
  --url https://github.com/YOUR_USERNAME/txgemma-mcp \
  --token YOUR_RUNNER_TOKEN_FROM_GITHUB \
  --name ec2-gpu-runner \
  --labels self-hosted,gpu,linux \
  --work _work

# Install as a service (runs on boot)
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

**Verify in GitHub:**
- Go to **Settings** → **Actions** → **Runners**
- You should see `ec2-gpu-runner` with status **Idle** (green)

#### Step 4: Create HuggingFace Cache Directory

```bash
# On EC2
mkdir -p ~/hf_cache

# This will be used to persist model downloads between test runs
```

#### Step 5: Enable GPU Tests

Now that your runner is set up:

1. Go to GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
2. Click **"New repository variable"**
3. Name: `ENABLE_GPU_TESTS`
4. Value: `true`
5. Click **"Add variable"**

#### Step 6: Test It

```bash
# Trigger manually
# Go to Actions tab → CI Tests → Run workflow → Select main branch → Run

# Or push to main
git push origin main
```

Watch the Actions tab - you should see the GPU tests run on your EC2 instance!

### Managing the Self-Hosted Runner

#### Check Runner Status

```bash
# On EC2
sudo ./svc.sh status

# View logs
sudo journalctl -u actions.runner.* -f
```

#### Stop/Start Runner

```bash
# Stop
sudo ./svc.sh stop

# Start
sudo ./svc.sh start

# Restart
sudo ./svc.sh restart
```

#### Remove Runner

```bash
# On EC2
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token YOUR_REMOVAL_TOKEN_FROM_GITHUB
```

### Troubleshooting CI Issues

#### Unit Tests Fail

```bash
# Run locally to debug
uv run pytest -m "not gpu" -vv

# Check specific failing test
uv run pytest tests/test_config.py::TestConfigModels::test_predict_config_defaults -vv
```

#### Linting Fails

```bash
# See all issues
uv run ruff check .

# Auto-fix
uv run ruff check --fix .
uv run ruff format .

# Commit fixes
git add .
git commit -m "Fix linting issues"
git push
```

#### GPU Tests Skipped

This is normal if `ENABLE_GPU_TESTS` is not set to `true`.

**To enable:**
- Set `ENABLE_GPU_TESTS=true` variable (see instructions above)
- Make sure self-hosted runner is running

#### Runner Not Picking Up Jobs

```bash
# On EC2, check runner status
sudo ./svc.sh status

# Check logs
sudo journalctl -u actions.runner.* -f

# Restart runner
sudo ./svc.sh restart
```

#### HuggingFace Download Fails in CI

```bash
# Check HF_TOKEN is set
# GitHub repo → Settings → Secrets → Actions → Check HF_TOKEN exists

# Test token validity
uv run huggingface-cli whoami --token YOUR_TOKEN
```

#### Config Not Loading in Tests

```bash
# Check config.yaml exists
ls -la config.yaml

# Verify environment variables
echo $TXGEMMA_PREDICT_MODEL

# Run with debug logging
uv run pytest tests/test_config.py --log-cli-level=DEBUG -v
```

### CI/CD Best Practices

#### For Contributors (No GPU Access)

```bash
# 1. Before committing
uv run pytest -m "not gpu"
uv run ruff check --fix .
uv run ruff format .

# 2. Commit and push
git add .
git commit -m "Your changes"
git push origin your-branch

# 3. CI will run unit tests automatically
# GPU tests only run on main branch (handled by maintainers)
```

#### For Maintainers (With GPU Access)

```bash
# 1. Before merging to main
uv run pytest --run-gpu  # Run locally

# 2. Merge to main
git checkout main
git merge your-branch
git push origin main

# 3. CI runs full test suite on EC2 (if enabled)

# 4. Monitor Actions tab for results
```

### Coverage Tracking with Codecov

If you set up `CODECOV_TOKEN`, you get:

- 📊 Coverage dashboard at https://codecov.io/gh/YOUR_USERNAME/txgemma-mcp
- 📈 Coverage trends over time
- 💬 PR comments showing coverage changes
- 🎯 Line-by-line coverage visualization

**Without Codecov:**
- Coverage still runs in CI
- Results shown in test logs
- No historical tracking

**To view coverage locally:**
```bash
uv run pytest --cov=txgemma --cov-report=html
open htmlcov/index.html
```

## Performance Benchmarks

### Test Execution Times

| Test Suite | Count | Time (No GPU) | Time (With GPU) |
|------------|-------|---------------|-----------------|
| test_config.py | 32 | ~2s | ~2s |
| test_prompts.py | 60+ | ~3s | ~3s |
| test_tool_factory.py | 31 | ~2s | ~2s |
| test_chat_factory.py | 10+ | ~1s | ~1s |
| test_executor.py | 15+ | ~2s | ~2s |
| test_server.py | 26 | ~2s | ~2s |
| **Fast Total** | **~175** | **~12s** | **~12s** |
| test_model.py | 20+ | skipped | ~5-15min (first) / ~60-90s (cached) |
| **All Tests** | **~195** | **~12s** | **~5-15min (first) / ~72-102s (cached)** |

**Note**: GPU test time depends on config:
- Development (2b + 9b): ~5min first / ~60s cached
- Production (9b + 9b): ~10min first / ~75s cached
- Research (27b + 27b): ~15min first / ~90s cached

### Hardware Performance

| Platform | GPU | Config | First Run | Cached Run |
|----------|-----|--------|-----------|------------|
| Apple M1 Pro | MPS (Metal) | Dev (2b+9b) | ~3min | ~30s |
| Apple M2 Max | MPS (Metal) | Dev (2b+9b) | ~2.5min | ~25s |
| Apple M3 Max | MPS (Metal) | Prod (9b+9b) | ~5min | ~40s |
| AWS g5.xlarge | A10G (24GB) | Dev (2b+9b) | ~5min | ~60s |
| Local NVIDIA RTX 3090 | CUDA (24GB) | Dev (2b+9b) | ~4min | ~45s |
| AWS g5.2xlarge | A10G (32GB) | Prod (9b+9b) | ~8min | ~75s |

## Summary

### Quick Reference

| Command | Use Case | Time | GPU Required |
|---------|----------|------|--------------|
| `uv run pytest` | Fast tests (default) | ~12s | ❌ |
| `uv run pytest --run-gpu` | All tests with GPU | ~5-15min / ~60-90s | ✅ |
| `uv run pytest -m "not gpu"` | Explicitly skip GPU | ~12s | ❌ |
| `uv run pytest -m gpu --run-gpu` | Only GPU tests | ~5-15min / ~60-90s | ✅ |
| `uv run pytest tests/test_config.py` | Config tests only | ~2s | ❌ |

### Test Distribution

- **~175 tests**: No GPU required (config, prompts, tools, executor, chat, server)
- **~20 tests**: GPU required (dual model loading and generation)
- **Total**: ~195 tests
- **Coverage**: 90%+ with all tests

### New in This Version

✅ **Configuration Testing** (32 tests)
- YAML loading and validation
- Environment variable overrides
- Preset configurations
- Singleton behavior

✅ **Dual Model Support** (20+ tests)
- Predict model (TDC tasks)
- Chat model (explanations)
- Config integration
- Both models tested on GPU

✅ **Chat Tool Testing** (10+ tests)
- Registration and execution
- Error handling
- Config integration

### CI/CD Summary

**Minimum setup (no GPU):**
- ✅ Just commit code
- ✅ Unit tests run automatically
- ✅ No secrets needed

**Full setup (with GPU):**
1. Add `HF_TOKEN` secret
2. Set up EC2 self-hosted runner
3. Set `ENABLE_GPU_TESTS=true` variable
4. (Optional) Add `CODECOV_TOKEN` for coverage tracking

**Current configuration:**
- Fast tests: Always run (12-15 seconds)
- GPU tests: Optional (60-90 seconds when enabled, after initial download)
- Linting: Always run (2-3 seconds)

### Development Cycle

1. **Write code** → Run `uv run pytest` (fast tests)
2. **Test config changes** → Run `uv run pytest tests/test_config.py`
3. **Before commit** → Run `uv run pytest -m "not gpu" --cov=txgemma`
4. **Push to PR** → CI runs fast tests automatically
5. **Merge to main** → CI runs GPU tests on EC2 (if enabled)

---

**Need help?** 
- Check `conftest.py` for test configuration
- Run `pytest --markers` to see all available markers
- See `.github/workflows/tests.yml` for CI/CD details
- Check `config.yaml` for model configuration used in tests