# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Backend image: FastAPI + uvicorn serving the FSAE simulation API.
#
# Paired with webapp/Dockerfile (nginx) in docker-compose.yaml. This image
# exposes the API on :8001; the webapp image reverse-proxies /api -> :8001
# over the docker compose network.
# -----------------------------------------------------------------------------

FROM python:3.12-slim

WORKDIR /app

# Build-time dependencies for numpy/scipy/pandas wheels that may compile
# from source on uncommon platforms, and curl for container healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer caches when source changes.
COPY pyproject.toml ./
# Create minimal src/ so editable install has a package to point at. The real
# source is bind-mounted in compose or copied below for production.
RUN mkdir -p src/fsae_sim && touch src/fsae_sim/__init__.py
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code.
COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY Real-Car-Data-And-Stats/ ./Real-Car-Data-And-Stats/

# Re-install editable after copying so fsae_sim package metadata is complete.
RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8001

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
