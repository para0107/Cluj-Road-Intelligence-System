# ── backend/Dockerfile ────────────────────────────────────────────────────────
# Builds a lean FastAPI + Uvicorn image.
# Heavy ML / pipeline dependencies (torch, ultralytics, etc.) are intentionally
# excluded — the API only needs the database / web stack.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps needed by psycopg2-binary, shapely, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the backend requirements so Docker can cache this layer
COPY backend/requirements.backend.txt ./requirements.backend.txt
RUN pip install --no-cache-dir -r requirements.backend.txt

# Copy the rest of the project (we need backend/ and any shared modules)
COPY . .

# Expose the API port
EXPOSE 8000

# Start Uvicorn — no --reload in production
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
