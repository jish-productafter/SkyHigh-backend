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
# Ensure all files including hidden directories are copied
COPY app/ .

# Verify LanceDB data was copied correctly with all subdirectories
RUN if [ -d "utils/lancedb_data" ]; then \
        echo "=== LanceDB data directory verification ===" && \
        echo "Directory structure:" && \
        find utils/lancedb_data -type d | sort && \
        echo "" && \
        echo "Checking for _versions directories:" && \
        find utils/lancedb_data -name "_versions" -type d && \
        echo "" && \
        echo "Checking for manifest files:" && \
        find utils/lancedb_data -name "*.manifest" && \
        echo "" && \
        echo "Detailed check for A1_MINIMAL_vocabulary:" && \
        if [ -d "utils/lancedb_data/A1_MINIMAL_vocabulary.lance" ]; then \
            echo "  Directory exists" && \
            ls -la utils/lancedb_data/A1_MINIMAL_vocabulary.lance/ && \
            if [ -d "utils/lancedb_data/A1_MINIMAL_vocabulary.lance/_versions" ]; then \
                echo "  ✓ _versions directory exists" && \
                ls -la utils/lancedb_data/A1_MINIMAL_vocabulary.lance/_versions/; \
            else \
                echo "  ✗ _versions directory MISSING!"; \
            fi; \
        else \
            echo "  ✗ A1_MINIMAL_vocabulary.lance directory not found"; \
        fi; \
    else \
        echo "ERROR: utils/lancedb_data directory not found!"; \
    fi

# Explicitly ensure LanceDB data directory is copied and verify it exists with complete structure
RUN if [ ! -d "utils/lancedb_data" ]; then \
        echo "ERROR: utils/lancedb_data directory not found!" && \
        echo "Contents of utils/:" && ls -la utils/ && \
        exit 1; \
    else \
        echo "LanceDB data directory found. Contents:" && \
        ls -la utils/lancedb_data/ && \
        echo "LanceDB tables:" && \
        find utils/lancedb_data -name "*.lance" -type d | head -10 && \
        echo "Verifying table structure..." && \
        for table_dir in utils/lancedb_data/*.lance; do \
            if [ -d "$table_dir" ]; then \
                echo "Checking table: $table_dir" && \
                ls -la "$table_dir/" && \
                if [ ! -d "$table_dir/_versions" ]; then \
                    echo "WARNING: Missing _versions directory in $table_dir" && \
                    ls -la "$table_dir/" || true; \
                else \
                    echo "✓ _versions directory exists in $table_dir" && \
                    ls -la "$table_dir/_versions/" || true; \
                fi && \
                if [ ! -d "$table_dir/data" ]; then \
                    echo "WARNING: Missing data directory in $table_dir"; \
                else \
                    echo "✓ data directory exists in $table_dir"; \
                fi; \
            fi; \
        done && \
        echo "Setting proper permissions for LanceDB data..." && \
        chmod -R u+rX utils/lancedb_data && \
        echo "Permissions set. Verifying access..." && \
        ls -la utils/lancedb_data/ && \
        echo "LanceDB data directory is ready."; \
    fi

# Use uv to run uvicorn with the correct environment
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]