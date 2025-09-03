FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir gradio==4.44.0 httpx==0.27.0
COPY app.py /app/app.py
EXPOSE 7860
CMD ["python", "app.py"]

