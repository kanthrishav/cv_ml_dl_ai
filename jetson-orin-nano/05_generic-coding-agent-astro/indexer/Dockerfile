FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini tar unzip xz-utils p7zip-full gcc build-essential \
    libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# libs: API, clients, parsing, embeddings
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic httpx \
    meilisearch qdrant-client \
    beautifulsoup4 lxml \
    markdown-it-py mdit-py-plugins \
    pygments chardet pyyaml \
    fastembed==0.3.5

COPY app /app/app

EXPOSE 8383
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8383","--log-level","info","--access-log"]

