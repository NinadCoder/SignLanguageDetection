FROM python:3.10-slim

# System deps for OpenCV/MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # cut down TF threads/memory
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # Gunicorn: single worker (free tier), higher timeout for cold start
    WEB_CONCURRENCY=1 \
    GUNICORN_CMD_ARGS="--timeout 180 --graceful-timeout 30 --keep-alive 5"

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code & model
COPY . .

# Expose app port
EXPOSE 8000

# Important: avoid preloading app (keeps imports lazy)
CMD ["bash", "-lc", "gunicorn -w ${WEB_CONCURRENCY} -b 0.0.0.0:8000 app:app --preload=false"]
