# Serving image for Hugging Face Spaces (Docker SDK). Also runs anywhere Docker does:
#   docker build -t brain-tumor-classifier .
#   docker run -p 7860:7860 brain-tumor-classifier   ->  http://localhost:7860
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOST=0.0.0.0 \
    PORT=7860

# Spaces run the container as UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user model/ model/
COPY --chown=user web/ web/

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"

CMD ["python", "web/serve.py"]
