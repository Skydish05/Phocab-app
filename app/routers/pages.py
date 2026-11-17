"""HTML pages. They hold no data of their own: they call the JSON API from the browser."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.templating import templates

# Hidden from /docs: the schema there documents the JSON API, not the pages.
router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    # No auth dependency: the JWT lives in the browser, so the page decides on the
    # client whether to show the login form or the upload form.
    return templates.TemplateResponse(request, "index.html")
