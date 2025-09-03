FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies for faiss wheel and basic runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# App code
COPY main.py /app/main.py

# Data dir
RUN mkdir -p /data

EXPOSE 9150

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9150"]

