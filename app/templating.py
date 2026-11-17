"""Jinja2 environment shared by the page routes and the HTMX fragments."""

from pathlib import Path

from fastapi.templating import Jinja2Templates

# One instance for the whole app, so every router renders with the same loader
# and autoescaping (on for .html, which keeps LLM-written sentences inert).
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")
