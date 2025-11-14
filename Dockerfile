FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies
# - ffmpeg: required for Whisper to process audio files
# - libpq-dev: required to build psycopg2 (PostgreSQL adapter)
# - gcc: required to compile psycopg2
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* .

# Install dependencies
RUN uv sync --frozen

# Download Whisper model during build
RUN uv run python -c "import whisper; whisper.load_model('base')"

# Copy application code
COPY app/ .

# Use uv to run uvicorn with the correct environment
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]