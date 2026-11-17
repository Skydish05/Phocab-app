# Model service image. Build from the repo root (the package must be importable
# as `model_service`):
#   docker build -f model_service/Dockerfile -t phocab-model .

# --- builder: install the ML dependencies into a self-contained venv ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# requirements.txt points pip at PyTorch's CPU wheel index, so this installs the
# CPU-only torch build (no CUDA libraries, which would be several GB).
COPY model_service/requirements.txt .
RUN pip install -r requirements.txt

# --- runtime ---
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache

RUN useradd --create-home --uid 10001 app \
    # Created and owned by `app` here so an empty named volume mounted on it
    # inherits that ownership and the non-root user can write weights into it.
    && mkdir /hf_cache && chown app:app /hf_cache
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY model_service/__init__.py model_service/main.py model_service/classifier.py model_service/labels_ko.csv ./model_service/

USER app
# Weights live in a volume, not the image: the image stays small and rebuilds
# don't re-download them. Compose mounts a named volume here.
VOLUME ["/hf_cache"]
EXPOSE 8001

# uvicorn only accepts connections after the lifespan hook has loaded the model,
# so "healthy" means "ready to classify". The long start period covers the first
# run, which downloads the weights.
HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)"]

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
