from fastapi.testclient import TestClient

from app.main import create_app
from app.templating import templates


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


def test_index_reads_listen_buttons_aloud_in_korean():
    page = TestClient(create_app()).get("/").text

    # One page-level handler serves every [data-speak] button, and the buttons stay
    # hidden until a Korean voice is found.
    assert "speechSynthesis" in page and '"ko-KR"' in page
    assert "body:not(.can-speak) [data-speak]" in page


def test_listen_button_escapes_the_text():
    speak = templates.env.get_template("_speak.html").module.speak

    html = str(speak('그는 "안녕" <b>', "Listen to the sentence"))

    assert 'data-speak="그는 &#34;안녕&#34; &lt;b&gt;"' in html
    assert 'type="button"' in html and 'aria-label="Listen to the sentence"' in html
