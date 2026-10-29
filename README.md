# Phocab

Photograph an everyday object and get a Korean vocabulary card (word, romanization,
example sentences, audio), reviewed with Leitner boxes.

**Status:** in progress. See `docs/PLAN.md`.

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                      # then fill in the required values (JWT_SECRET, S3_BUCKET, GEMINI_API_KEY)

ruff check . && ruff format --check .   # lint
pytest -q                               # tests
uvicorn app.main:app --reload           # run at http://localhost:8000 (docs at /docs)
```
