# Multi-stage Dockerfile for ML API
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY .env ./
COPY MLproject ./

# Create necessary directories
RUN mkdir -p model-cache data logs

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "src/api/app.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio black flake8 isort

# Keep container running for development
CMD ["tail", "-f", "/dev/null"]