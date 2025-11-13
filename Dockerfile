FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install ffmpeg (required for Whisper to process audio files)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* .

# Install dependencies
RUN uv sync --frozen

# Download Whisper model during build
RUN uv run python -c "import whisper; whisper.load_model('base')"

# Copy application code
COPY app/ .

# Explicitly ensure LanceDB data directory is copied and verify it exists
RUN if [ ! -d "utils/lancedb_data" ]; then \
        echo "ERROR: utils/lancedb_data directory not found!" && \
        echo "Contents of utils/:" && ls -la utils/ && \
        exit 1; \
    else \
        echo "LanceDB data directory found. Contents:" && \
        ls -la utils/lancedb_data/ && \
        echo "LanceDB tables:" && \
        find utils/lancedb_data -name "*.lance" -type d | head -10 && \
        echo "Setting proper permissions for LanceDB data..." && \
        chmod -R u+rX utils/lancedb_data && \
        echo "Permissions set. Verifying access..." && \
        ls -la utils/lancedb_data/ && \
        echo "LanceDB data directory is ready."; \
    fi

# Use uv to run uvicorn with the correct environment
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]