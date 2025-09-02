FROM python:3.10-slim

RUN pip install --no-cache-dir fastapi uvicorn[standard]

WORKDIR /app
COPY app /app

EXPOSE 8090
CMD ["uvicorn", "admin:app", "--host", "0.0.0.0", "--port", "8090"]

