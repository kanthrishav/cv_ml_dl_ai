FROM python:3.10-slim

# base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client jq curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

# create a real user with UID 1000 so SSH / libc can resolve it
RUN useradd -m -u 1000 -s /bin/bash astro

WORKDIR /app

# install python deps as root into the image
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx ollama

# your app
COPY app /app/app

# runtime environment
ENV PYTHONPATH=/app
ENV HOME=/home/astro

# create mount targets and hand them to astro
RUN mkdir -p /workspaces /logs && chown -R astro:astro /workspaces /logs

# switch to the non-root user (UID 1000)
USER astro

EXPOSE 8088
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8088"]

