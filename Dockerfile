# Multi-stage build for minimal production image
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /home/mluser/.local
COPY src/ ./src/
COPY MLproject .

# Set ownership and permissions
RUN chown -R mluser:mluser /app
USER mluser

# Add local user path
ENV PATH=/home/mluser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]