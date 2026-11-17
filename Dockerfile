# API image. Build from the repo root:
#   docker build -t phocab-api .

# --- builder: install dependencies into a self-contained venv ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only the dependency list from pyproject.toml, not the phocab package itself:
# app/config.py locates labels_ko.csv relative to the source tree, which breaks
# once `app` is installed into site-packages. Copying only pyproject.toml also
# means this layer is rebuilt when dependencies change, not on every code edit.
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
    && pip install -r requirements.txt

# --- runtime: no pip cache or build leftovers, non-root ---
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# A fixed UID so file ownership is predictable on mounted volumes.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Explicit paths instead of `COPY . .`, so nothing unexpected (a stray .env,
# eval photos) can end up in the image even if .dockerignore misses it.
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
# The vocabulary source of truth (CLAUDE.md rule 1); the API reads the Korean
# word, romanization and TOPIK level from it.
COPY model_service/labels_ko.csv ./model_service/labels_ko.csv

USER app
EXPOSE 8000

# The slim image has no curl, so the check uses Python's stdlib.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# Migrate on startup: one API instance, and Alembic upgrades are idempotent, so
# a fresh database (local, CI, EC2) gets its schema with no manual step. With
# several replicas this should become a separate one-off step to avoid races.
# `exec` makes uvicorn PID 1 so it receives SIGTERM and shuts down cleanly.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
