import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
from tests.conftest import TestingSessionLocal


API_KEY_A = os.getenv("API_KEY_TENANT_A")
API_KEY_B = os.getenv("API_KEY_TENANT_B")
API_KEY_C = os.getenv("API_KEY_TENANT_C")


@pytest.fixture
def raw_client(setup_database):
    """Client with no auth override — tests real auth middleware."""
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_missing_api_key_returns_422(raw_client):
    """No x-api-key header at all."""
    response = raw_client.get("/tools")
    assert response.status_code == 422


def test_missing_api_key_on_post(raw_client):
    response = raw_client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    })
    assert response.status_code == 422


def test_missing_api_key_on_delete(raw_client):
    response = raw_client.delete("/tools/some-id")
    assert response.status_code == 422


def test_invalid_api_key_returns_401(raw_client):
    response = raw_client.get("/tools", headers={"x-api-key": "invalid-key"})
    assert response.status_code == 401


def test_invalid_api_key_error_message(raw_client):
    response = raw_client.get("/tools", headers={"x-api-key": "invalid-key"})
    assert "invalid" in response.json()["detail"].lower()


def test_empty_api_key_returns_401(raw_client):
    response = raw_client.get("/tools", headers={"x-api-key": ""})
    assert response.status_code == 401


def test_wrong_api_key_returns_401(raw_client):
    response = raw_client.get("/tools",
                              headers={"x-api-key": "key-tenant-wrong-999"})
    assert response.status_code == 401


def test_valid_api_key_a_returns_200(raw_client):
    response = raw_client.get("/tools", headers={"x-api-key": API_KEY_A})
    assert response.status_code == 200


def test_valid_api_key_b_returns_200(raw_client):
    response = raw_client.get("/tools", headers={"x-api-key": API_KEY_B})
    assert response.status_code == 200


def test_valid_api_key_c_returns_200(raw_client):
    response = raw_client.get("/tools", headers={"x-api-key": API_KEY_C})
    assert response.status_code == 200


def test_different_keys_are_isolated(raw_client):
    """Tools created by tenant a must not be visible to tenant b."""
    raw_client.post("/tools",
                    headers={"x-api-key": API_KEY_A},
                    json={"name": "a-tool", "description": "a's tool"}
                    )

    response = raw_client.get("/tools", headers={"x-api-key": API_KEY_B})
    assert response.status_code == 200
    assert len(response.json()) == 0


def test_same_key_can_see_own_tools(raw_client):
    """Tenant can see their own tools."""
    raw_client.post("/tools",
                    headers={"x-api-key": API_KEY_A},
                    json={"name": "a-tool", "description": "a's tool"}
                    )

    response = raw_client.get("/tools", headers={"x-api-key": API_KEY_A})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_health_endpoint_requires_no_auth(raw_client):
    """Public health endpoint must be accessible without an API key."""
    response = raw_client.get("/health")
    assert response.status_code == 200
