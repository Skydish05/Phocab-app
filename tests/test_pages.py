from fastapi.testclient import TestClient

from app.main import create_app


def test_index_serves_the_upload_page_without_login():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # The page's upload form must post to the JSON API route, not a page-only copy.
    assert 'hx-post="/cards/from-photo"' in response.text
    assert 'id="auth-form"' in response.text


def test_pages_are_not_in_the_api_docs():
    client = TestClient(create_app())

    assert "/" not in client.get("/openapi.json").json()["paths"]
