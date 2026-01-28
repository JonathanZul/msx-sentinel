FROM python:3.11-slim

LABEL maintainer="MSX-Sentinel"
LABEL description="Tier 1/2 container for WSI tiling and manifest operations"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set PYTHONPATH so imports resolve correctly
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-c", "from src.core.config import Config; print(f'Environment: {Config.get().environment.value}')"]
