import pytest
from unittest.mock import patch
from sqlalchemy.exc import SQLAlchemyError
from fastapi.testclient import TestClient
from app.main import app
from app.middleware.auth import get_tenant
from app.database import get_db
from tests.conftest import TestingSessionLocal


@pytest.fixture
def error_client():
    """
    TestClient with dependency overrides and
    raise_server_exceptions=False.
    """
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def override_get_tenant():
        return "tenant_test"

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_tenant] = override_get_tenant

    yield TestClient(app, raise_server_exceptions=False)

    app.dependency_overrides.clear()


def test_sqlalchemy_error_returns_500(error_client):
    with patch(
        "app.repositories.tool_repo.ToolRepository.get_all",
        side_effect=SQLAlchemyError("DB connection lost")
    ):
        response = error_client.get("/tools")
        assert response.status_code == 500
        assert "database error" in response.json()["detail"].lower()


def test_sqlalchemy_error_returns_json(error_client):
    with patch(
        "app.repositories.tool_repo.ToolRepository.get_all",
        side_effect=SQLAlchemyError("DB connection lost")
    ):
        response = error_client.get("/tools")
        assert "application/json" in response.headers["content-type"]


def test_generic_exception_returns_500(error_client):
    with patch(
        "app.repositories.tool_repo.ToolRepository.get_all",
        side_effect=Exception("Something totally unexpected")
    ):
        response = error_client.get("/tools")
        assert response.status_code == 500
        assert "unexpected error" in response.json()["detail"].lower()


def test_generic_exception_returns_json(error_client):
    with patch(
        "app.repositories.tool_repo.ToolRepository.get_all",
        side_effect=Exception("Something totally unexpected")
    ):
        response = error_client.get("/tools")
        assert "application/json" in response.headers["content-type"]


def test_http_exception_not_swallowed_by_global_handler(client):
    """HTTPExceptions must still return their correct status code."""
    response = client.get("/tools/non-existent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
