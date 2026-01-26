FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH=${PATH}:/root/.local/bin

EXPOSE 8000

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY server.py ./
COPY txgemma ./txgemma
COPY config.yaml ./

# Install uv and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv sync --no-dev

# Use fastmcp CLI to configure transport, host, and port
CMD ["uv", "run", "fastmcp", "run", "server.py", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]