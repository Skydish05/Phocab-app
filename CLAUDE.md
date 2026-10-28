# Korean Photo Vocab App

Web app where a user photographs an everyday object and gets a Korean vocabulary card (word, romanization, example sentences, audio). Cards go into a personal deck reviewed with Leitner boxes. This is a portfolio project for Backend / AI Engineer internship applications, so it must be deployed, tested, and evaluated with real numbers.

The full plan (architecture, schema, endpoints, 14-day checklist) is in `docs/PLAN.md`. Follow it. Work on ONE checklist item at a time and tell me which item you are on.

## Stack
- Python 3.12, FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL
- Pydantic v2 for all request, response, and LLM-output schemas
- Frontend: Jinja2 templates + HTMX + Tailwind (DaisyUI). No React.
- Model service (separate container): Hugging Face `transformers`, zero-shot image classification
- Storage: S3 (boto3). Deploy: Docker Compose on EC2, images in ECR, GitHub Actions with OIDC
- Tooling: Ruff (lint + format), pytest, pydantic-settings for config

## Layout
- `app/` FastAPI service (routers/, services/, templates/, static/)
- `model_service/` separate container with `labels_ko.csv` (vocabulary source of truth)
- `eval/` evaluation photos, ground-truth labels, `run_eval.py`, saved results
- `migrations/`, `tests/`, `infra/`, `.github/workflows/`, `docs/`

## Commands
Update these as they become real; keep them accurate.
- Setup: `python3.12 -m venv .venv && source .venv/bin/activate` (system pip is PEP 668-locked)
- Install: `pip install -e ".[dev]"`
- Lint/format: `ruff check . && ruff format --check .`
- Test: `pytest -q`
- Run locally (no Docker yet): `uvicorn app.main:app --reload`
- Run locally with Docker (not yet available): `docker compose -f infra/docker-compose.yml up --build`
- Migrate (not yet available): `alembic upgrade head`
- Eval (not yet available): `python eval/run_eval.py`

## Rules
1. Korean word, romanization, and TOPIK level ALWAYS come from `labels_ko.csv`. The LLM may only write example sentences and mnemonics, and must never change the vocabulary fields.
2. All LLM output is validated against a Pydantic schema, with at most 2 retries.
3. Every behavior change comes with a test. Run lint and tests before saying a task is done.
4. No secrets in the repo. Config comes from environment variables; keep `.env.example` current.
5. Never commit user photos or eval photos that show people or personal information.
6. Do not add Redis, a job queue, or new services unless `docs/PLAN.md` says so or I ask.
7. Never state a metric (accuracy, latency, cost, cache hit rate) that was not produced by a script in this repo.
8. Keep changes small. Propose a short plan first for anything touching more than 3 files, and wait for my go-ahead.
9. Write short comments explaining WHY for non-obvious decisions. I need to explain every design choice in interviews.

## Git
Small commits, one per checklist item, message format `feat: ...`, `fix: ...`, `test: ...`, `chore: ...`.
