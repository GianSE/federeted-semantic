# Build context is the project root (set in docker-compose.yml)
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY fl-server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy shared model/image utilities from ml-service
COPY ml-service/app/core/model_utils.py /app/core/model_utils.py
COPY ml-service/app/core/image_utils.py /app/core/image_utils.py
RUN touch /app/core/__init__.py

# Copy server application
COPY fl-server/app/ /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
