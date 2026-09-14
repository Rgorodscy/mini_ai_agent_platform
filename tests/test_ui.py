"""
The web console and the two endpoints it depends on.
"""

from pathlib import Path

import pytest

from app.config import GROQ_MODEL
from app.core.tool_implementations import TOOL_REGISTRY

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


# --- GET /models ---


def test_models_lists_what_the_api_accepts(client):
    body = client.get("/models").json()

    assert GROQ_MODEL in body["models"]
    assert body["default"] == GROQ_MODEL


def test_every_listed_model_is_accepted_by_a_run(client, fake_llm):
    """The list is only useful if the run endpoint agrees with it."""
    from tests.conftest import make_llm_response

    fake_llm.queue(make_llm_response(content="Done."))
    agent = client.post(
        "/agents",
        json={"name": "A", "role": "r", "description": "d", "tools": []},
    ).json()

    for model in client.get("/models").json()["models"]:
        response = client.post(
            f"/agents/{agent['id']}/run", json={"task": "hi", "model": model}
        )
        assert response.status_code == 201, model


def test_models_requires_authentication(raw_client):
    assert raw_client.get("/models").status_code == 422


# --- GET /tools/available ---


def test_available_tools_match_the_registry(client):
    names = [t["name"] for t in client.get("/tools/available").json()]

    assert names == sorted(TOOL_REGISTRY)


def test_available_tools_carry_descriptions(client):
    tools = client.get("/tools/available").json()

    assert all(t["description"] for t in tools)


def test_available_is_not_mistaken_for_a_tool_id(client):
    """
    Regression guard for route order: registered after /tools/{tool_id},
    "available" would be looked up as an id and answered with a 404.
    """
    assert client.get("/tools/available").status_code == 200


def test_tool_lookup_by_id_still_works(client):
    tool = client.post(
        "/tools", json={"name": "calculator", "description": "Math"}
    ).json()

    assert client.get(f"/tools/{tool['id']}").json()["name"] == "calculator"


def test_available_tools_requires_authentication(raw_client):
    assert raw_client.get("/tools/available").status_code == 422


# --- Serving the console ---


def test_root_redirects_to_the_console(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_console_without_trailing_slash_redirects(client):
    """Relative asset URLs resolve against the wrong directory without it."""
    response = client.get("/ui", follow_redirects=False)

    assert response.headers["location"] == "/ui/"


def test_console_page_is_served_without_authentication(raw_client):
    response = raw_client.get("/ui/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Agent Console" in response.text


@pytest.mark.parametrize(
    "asset,content_type",
    [("app.js", "javascript"), ("styles.css", "text/css")],
)
def test_console_assets_are_served(raw_client, asset, content_type):
    response = raw_client.get(f"/ui/{asset}")

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_console_sends_a_restrictive_content_security_policy(raw_client):
    csp = raw_client.get("/ui/").headers["content-security-policy"]

    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp


def test_console_assets_carry_the_security_headers(raw_client):
    headers = raw_client.get("/ui/app.js").headers

    assert "content-security-policy" in headers
    assert headers["x-content-type-options"] == "nosniff"


def test_security_headers_are_scoped_to_the_console(raw_client):
    """
    The headers come from the static-files mount, not a global middleware,
    so the API — and the streaming endpoint in particular — is untouched.
    """
    assert "content-security-policy" not in raw_client.get("/health").headers


def test_unknown_console_asset_is_a_404(raw_client):
    assert raw_client.get("/ui/does-not-exist.js").status_code == 404


# --- Static guarantees about the page itself ---


def test_page_has_no_inline_script_or_style():
    """
    The CSP forbids inline script and style, so any that crept in would be
    silently blocked in the browser rather than fail loudly here.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert "<script>" not in html
    assert "<style" not in html
    assert "style=" not in html
    assert " on" + "click=" not in html


def test_script_never_assigns_html():
    """
    Model answers and tool results are untrusted text — a retrieved
    document can contain markup. The console must only ever set text.
    """
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
    ):
        assert sink not in script, sink


def test_script_never_puts_the_key_in_a_url():
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "localStorage" not in script
    assert "?api_key" not in script
    assert "x-api-key" in script


@pytest.mark.parametrize("path", ["/", "/ui"])
def test_redirects_answer_head_requests(client, path):
    """Uptime checkers commonly probe with HEAD rather than GET."""
    response = client.head(path, follow_redirects=False)

    assert response.status_code in (302, 307)
