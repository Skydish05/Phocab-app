"""Jinja2 environment shared by the page routes and the HTMX fragments."""

import math
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

# One instance for the whole app, so every router renders with the same loader
# and autoescaping (on for .html, which keeps LLM-written sentences inert).
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def wants_html(request: Request) -> bool:
    """True for HTMX requests, which get an HTML fragment instead of JSON.

    One route serves both clients (PLAN.md: the JSON API is the contract), so
    there is a single copy of each query and of the ownership checks.
    """
    return request.headers.get("HX-Request") == "true"


def due_label(when: datetime, now: datetime | None = None) -> str:
    """'due now', 'due in 5 h', 'due tomorrow', 'due in 4 days'."""
    now = now or datetime.now(UTC)
    seconds = (when - now).total_seconds()
    if seconds <= 0:
        return "due now"
    # Rounded up, so a card due in 30 minutes never reads as "due in 0 h".
    if seconds < 86400:
        return f"due in {math.ceil(seconds / 3600)} h"
    days = math.ceil(seconds / 86400)
    return "due tomorrow" if days == 1 else f"due in {days} days"


templates.env.filters["due_label"] = due_label
