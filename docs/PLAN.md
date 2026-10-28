# Korean Photo Vocab App: Build Plan

Oct 27, 2026 · @Pavel

## Goals and scope

The goal is a deployed, tested photo-to-Korean-vocabulary web app with a measured AI pipeline, finished in 14 days, so it can anchor a Backend / AI Engineer CV.

**In scope**

- Photo upload from a phone, object recognition, and a generated Korean card (word, romanization, 1 Korean example sentence)
- Login, personal decks, and simple Leitner-box review
- Caching of generated cards, a "this is wrong" feedback button, and per-user rate limits
- FastAPI + PostgreSQL + S3, Docker Compose, GitHub Actions to AWS (EC2 + ECR)
- A Hugging Face model for recognition, with a hosted vision-language model as fallback
- An evaluation script run on about 50 of your own photos (accuracy, latency, cost per card)

**Out of scope for now**

- Redis and a job queue (add later only if time remains)
- Native mobile app, React frontend, Grafana/Prometheus
- Full spaced-repetition algorithm (SM-2)
- Custom-trained models

**Assumptions**

- About 3-4 hours per day
- A mobile-first web app using Jinja2 templates, HTMX, and Tailwind with DaisyUI
- A 4 GB RAM EC2 instance is enough to run the Hugging Face model on CPU (confirm by measuring memory use on day 2)

## Architecture

The API is the only service the browser talks to. It stores the photo in S3, asks the model service for labels, checks the word cache in PostgreSQL, and calls the hosted model only on low confidence or a cache miss.

&#91;embedded content: architecture · 4 containers, 2 external services\]

## Repo structure

One repository, two deployable services (the web/API app and a separate model service), so the model can be resized or replaced without touching the API.

```text
korean-vocab/
├── app/                      # FastAPI web + API service
│   ├── main.py               # app factory, router includes
│   ├── config.py             # settings from env (pydantic-settings)
│   ├── db.py                 # SQLAlchemy engine/session
│   ├── models.py             # ORM tables
│   ├── schemas.py            # Pydantic request/response models
│   ├── security.py           # password hashing, JWT
│   ├── deps.py               # current_user, rate limit dependencies
│   ├── routers/
│   │   ├── auth.py           # register, login
│   │   ├── cards.py          # upload photo, create/list/delete cards
│   │   ├── review.py         # next due card, submit answer
│   │   └── feedback.py       # "this is wrong" reports
│   ├── services/
│   │   ├── storage.py        # S3 upload/delete, presigned URLs
│   │   ├── recognizer.py     # calls model service, fallback logic
│   │   ├── card_generator.py # LLM call, schema validation, retries
│   │   ├── cache.py          # lookup/store by normalized label
│   │   └── leitner.py        # box promotion/demotion, due dates
│   ├── templates/            # Jinja2 + HTMX partials
│   └── static/               # Tailwind build output, icons
├── model_service/            # separate container
│   ├── main.py               # POST /classify -> labels + scores
│   ├── labels_ko.csv         # English label, Korean word, romanization, TOPIK level
│   └── Dockerfile
├── eval/
│   ├── photos/               # your ~50 labeled photos (not committed if private)
│   ├── labels.csv            # ground truth
│   ├── run_eval.py           # accuracy, latency, cost per card
│   └── results/              # saved reports (committed)
├── migrations/               # Alembic
├── tests/
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   └── nginx.conf            # reverse proxy + HTTPS
├── .github/workflows/
│   ├── ci.yml                # lint + tests on every push
│   └── deploy.yml            # build, push to ECR, deploy to EC2
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Data model

Five tables cover the whole product. The key design choice is splitting shared, cacheable content (`words`) from per-user state (`user_cards`), so one generated card serves every user.

| Table | Columns | Notes |
| --- | --- | --- |
| users | id, email (unique), password\_hash, created\_at, daily\_quota | Quota drives the rate limit |
| words | id, label\_en (unique, normalized), korean, romanization, topik\_level, example (text), audio\_url, source (hf / vlm), created\_at | The cache: one row per object label |
| photos | id, user\_id, s3\_key, word\_id (nullable), confidence, model\_used, latency\_ms, created\_at | Records every upload and which model path handled it |
| user\_cards | id, user\_id, word\_id, photo\_id, box (1-5), next\_review\_at, times\_correct, times\_wrong, created\_at | Unique on (user\_id, word\_id); Leitner state |
| feedback | id, user\_id, photo\_id, word\_id, reason (wrong\_object / wrong\_korean / other), note, created\_at | Your corrections dataset for evaluation |

**Indexes to add:** `words(label_en)`, `user_cards(user_id, next_review_at)`, `photos(user_id, created_at)`.

**Leitner rule:** a correct answer moves the card up one box and sets `next_review_at` to now plus the box interval (for example 1, 2, 4, 8, 16 days); a wrong answer returns it to box 1 for tomorrow.

## API endpoints

The JSON API is the contract; the Jinja/HTMX pages call the same routes and swap in HTML fragments. FastAPI's auto-generated `/docs` page doubles as live documentation for recruiters.

| Method | Route | Auth | Purpose |
| --- | --- | --- | --- |
| POST | /auth/register | no | Create account (email, password) |
| POST | /auth/login | no | Return JWT access token |
| POST | /cards/from-photo | yes | Upload image; returns the generated card and saves it to the deck |
| GET | /cards | yes | List the user's cards (paged, filter by box) |
| GET | /cards/{id} | yes | One card with photo URL |
| DELETE | /cards/{id} | yes | Remove a card from the deck |
| GET | /review/next | yes | Next due card, or none |
| POST | /review/{card\_id} | yes | Submit `correct: true/false`; updates box and next review |
| POST | /feedback | yes | Report a wrong recognition or wrong Korean |
| GET | /me | yes | Profile and remaining daily quota |
| GET | /health | no | Liveness check for the load balancer and CI |

**Upload handling for `/cards/from-photo`**

1. Check file type (JPEG/PNG/WebP) and size (cap at 5 MB); resize to a max of 1024 px on the long side.
2. Check the daily quota; return 429 when exceeded.
3. Store the image in S3, then call the model service for labels.
4. Normalize the top label, then look it up in `words`. A hit skips the LLM call.
5. On a miss, generate the card, validate it against a Pydantic schema, retry up to 2 times, and store it.
6. Create the `user_cards` row and return the card.

**Error codes to implement:** 400 bad file, 401 unauthenticated, 413 file too large, 422 validation, 429 quota, 502 model or LLM failure (with a clear message).

## AI pipeline and evaluation

The pipeline's job is to be grounded: the open model picks from a fixed vocabulary list, so the Korean word always comes from a curated source and not from free generation. The LLM only writes the extra: one Korean example sentence.

**Recognition (model\_service)**

- Load a CLIP-family model through Hugging Face `transformers` for zero-shot classification. Look up the exact model ID and its license on the Hub before you commit to one.
- Candidate labels come from `labels_ko.csv`: about 300-500 everyday object nouns, each with its Korean word, romanization, and TOPIK level. Build it by hand or from a word list with a permissive license, and spot-check the Korean yourself.
- Return the top 3 labels with scores. If the top score is below a threshold (tune it on your eval set), the API falls back to a hosted vision-language model that names the object freely; that result is flagged `source = vlm` and can be added to the vocabulary after review.

**Card generation (card\_generator)**

- Ask the LLM for strict JSON: 1 example sentence in Korean only (beginner level, using the word, no English translation). Validate with Pydantic; on failure, retry with the error message appended, up to 2 times.
- Never let the LLM change the Korean word or romanization taken from the vocabulary file.
- Cache by normalized `label_en` in the `words` table.

**Pronunciation**

- Version 1: the browser's built-in speech synthesis (`ko-KR` voice), which costs nothing and needs no storage.
- Optional later: a Hugging Face Korean TTS model, after checking its license allows your use.

**Evaluation script (`eval/run_eval.py`)**

1. Take about 50 photos of everyday objects in your room, kitchen, and campus, with a ground-truth label for each in `labels.csv`. Include a few hard cases (bad lighting, cluttered backgrounds, objects not in the vocabulary).
2. Run each photo through three configurations: Hugging Face only, hosted VLM only, and the hybrid (HF first, VLM fallback).
3. Record top-1 and top-3 accuracy, median and 95th-percentile latency, and cost per card (from the API's reported token usage and its current price list).
4. Save the results as a table in `eval/results/` and paste the summary into the README.
5. Re-run after changing the threshold or the vocabulary, and note which change helped.

The finding you want to report is a trade-off with numbers, such as "the hybrid kept accuracy within a few points of VLM-only while cutting cost per card by X%".

## Docker, CI/CD and AWS

Deploy a bare-bones version by day 5, then keep redeploying. Most deployment pain (HTTPS, env vars, permissions, memory) shows up on the first deploy, so find it early.

**AWS account setup (do this first)**

1. Create the account, enable MFA on the root user, then stop using root: create an admin IAM user or an IAM Identity Center user.
2. Set a billing budget with an email alert at a low threshold before creating any resources. Check AWS's current free-tier terms, because they differ for newer accounts.
3. Pick one region and stay in it (Seoul, `ap-northeast-2`, is the natural choice for Korean users).

**Resources**

- **S3 bucket** for photos: private, with a lifecycle rule that deletes photos after a set number of days.
- **ECR repositories** for the API image and the model-service image.
- **EC2 instance** (about 4 GB RAM; confirm after measuring the model) running Docker Compose with the API, model service, Postgres, and Nginx. Attach an IAM role to the instance for S3 access instead of storing keys on it.
- **Security group** allowing 80 and 443 from anywhere, and 22 from your IP only.
- **Domain and HTTPS:** a cheap domain plus Let's Encrypt (Certbot) through Nginx. A camera upload from a phone browser requires HTTPS on a real domain.
- **Later, if you want RDS on the CV:** move Postgres to RDS. It costs more than a container, so treat it as an upgrade.

**Docker**

- Use multi-stage builds and run as a non-root user.
- Keep the Hugging Face model cache in a volume so containers don't re-download weights on each restart.
- `docker-compose.yml` for local development (with a local Postgres), `docker-compose.prod.yml` for the server.
- Provide a `.env.example` and never commit real secrets.

**GitHub Actions**

- `ci.yml` on every push and pull request: install dependencies, run Ruff and the tests against a Postgres service container, and build the images.
- `deploy.yml` on pushes to `main`: authenticate to AWS with GitHub's OIDC provider and an IAM role (no long-lived access keys), build and push images to ECR, then SSH or use AWS Systems Manager to run `docker compose pull && up -d` on the instance.
- Run a smoke test against `/health` after each deploy and fail the workflow if it doesn't respond.

**Operations basics**

- JSON-structured logs with a request ID; Sentry for exceptions.
- Nightly Postgres dump to S3 with a simple cron job.
- A short load test (Locust or k6) against the card endpoint with the LLM mocked, so you can report requests per second and p95 latency.

**Privacy**

Delete photos after processing or after a short retention period, say so on the upload page, and don't keep face-containing images longer than needed. Because this serves users in Korea, read the basics of the Personal Information Protection Act before inviting outside users.

## 14-day checklist

Start on Tue Oct 27 (day 1) and finish on Mon Nov 9 (day 14). The rule: a deployed thin version by day 5, then deepen.

**Days 1-2: foundation and AI core**

- [ ] Create the GitHub repo, `pyproject.toml`, Ruff, pytest, and the folder layout above
- [ ] Write `labels_ko.csv` with 300+ everyday objects (Korean word, romanization, TOPIK level) and spot-check it
- [ ] Build `model_service` with a Hugging Face zero-shot model; run it on 10 test photos
- [ ] Measure the model's memory use and latency on CPU (this decides the EC2 size)
- [ ] Script the full pipeline: photo in, validated JSON card out

**Days 3-4: backend**

- [ ] Postgres + SQLAlchemy models + Alembic migration for the 5 tables
- [ ] Register/login with password hashing and JWT
- [ ] `/cards/from-photo` with validation, resizing, S3 upload, and the `words` cache
- [ ] Tests for auth, cache hit vs miss, and the card schema
- [ ] Dockerfiles and `docker-compose.yml` working locally

**Day 5: first deploy**

- [ ] Minimal upload page (Jinja2 + HTMX + Tailwind/DaisyUI) that shows the card
- [ ] AWS account hardening, budget alert, S3 bucket, ECR, EC2 with IAM role
- [ ] Deploy by hand once, with Nginx and HTTPS on your domain
- [ ] Test the camera upload from your phone

**Days 6-8: product features**

- [ ] Deck page and card detail page
- [ ] Leitner review flow (`/review/next`, `/review/{id}`) with tests for the box logic
- [ ] "This is wrong" feedback button and the `feedback` table
- [ ] Per-user daily quota and 429 handling
- [ ] Browser text-to-speech for the Korean word
- [ ] Mobile layout polish; add an empty state and error messages

**Days 9-10: evaluation**

- [ ] Photograph and label about 50 objects in `eval/labels.csv`
- [ ] Write `run_eval.py` and run the three configurations
- [ ] Tune the confidence threshold for the fallback; record the results
- [ ] Summarize the findings in a table for the README

**Days 11-12: automation and reliability**

- [ ] `ci.yml`: lint, tests with a Postgres service, image build
- [ ] `deploy.yml`: OIDC to AWS, push to ECR, redeploy on EC2, `/health` smoke test
- [ ] Structured logging and Sentry
- [ ] Nightly Postgres backup to S3
- [ ] Short load test and note the results

**Days 13-14: ship and share**

- [ ] README: pitch, architecture diagram, run-locally steps, eval results, trade-offs, limitations
- [ ] 60-90 second demo video or GIF
- [ ] Add a short privacy notice to the upload page
- [ ] Invite 10-20 users (classmates, international students) and collect feedback
- [ ] Pin the repo, add the live link and metrics to your CV and LinkedIn

**If you fall behind**, cut in this order: load test, nightly backup, Sentry, then the hybrid fallback (use the Hugging Face path alone and report that limitation). Don't cut the deployment, tests, or the evaluation.

## README, CV bullets and risks

The README is what recruiters actually read, so write it as you build, not at the end.

**README outline**

1. One-sentence pitch and a live link, with a screenshot or demo GIF at the top
2. Architecture diagram and a short explanation of why the model service is separate
3. Tech stack
4. Evaluation results table and the main trade-off you found
5. How to run locally (one `docker compose up` command)
6. Design decisions: why caching, why grounding the vocabulary, why FastAPI + HTMX
7. Limitations and what you would build next

**CV bullets (fill in with your real numbers)**

- Built and deployed a photo-to-Korean-vocabulary web app (FastAPI, PostgreSQL, Docker, AWS EC2/S3/ECR) used by N international students
- Designed a hybrid recognition pipeline using a Hugging Face zero-shot model with a vision-language-model fallback; evaluated on a 50-image labeled set, reaching X% top-3 accuracy at Y% lower cost per card than VLM-only
- Added a PostgreSQL-backed card cache that cut LLM calls by Z%, plus per-user rate limiting
- Set up CI/CD with GitHub Actions (tests, image builds, OIDC-authenticated deploys to AWS) and a post-deploy health check
- Implemented JWT auth, Leitner spaced-repetition review, structured logging, and automated tests

Only include a number once you have measured it. Interviewers will ask how you got it.

**Risks and fallbacks**

| Risk | What to do |
| --- | --- |
| Model too slow or too large for the instance | Use a smaller model, a larger instance for the demo period, or a VLM-only path |
| Wrong or unnatural Korean | Keep the vocabulary file as the source of truth; ask a Korean friend to review the file |
| AWS cost surprise | Budget alert on day 5, stop the instance when idle, delete test resources |
| Scope creep | Follow the cut order in the checklist; finish and deploy before adding features |
| Privacy concerns with user photos | Short retention, a visible notice, no photos in the public repo |
| Internship applications stall | Apply with the day-5 deployed version while you polish; don't wait for day 14 |
