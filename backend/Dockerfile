# ── backend/Dockerfile ────────────────────────────────────────────────────────
# Builds a lean FastAPI + Uvicorn image.
# Includes system tools (exiftool, ffmpeg) for portable telemetry processing.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install system dependencies for database, GIS, ExifTool, FFmpeg, and OpenCV
# Note: libgl1 is used as libgl1-mesa-glx is obsolete in newer Debian versions (Trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        libgeos-dev \
        exiftool \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
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