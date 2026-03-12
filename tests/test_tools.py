from app.middleware.auth import get_tenant


def test_create_tool(client):
    response = client.post("/tools", json={
        "name": "Test Tool",
        "description": "A tool for testing",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Tool"
    assert data["description"] == "A tool for testing"
    assert "id" in data
    assert "tenant_id" in data
    assert data["tenant_id"] == "test_tenant"


def test_get_tool(client):
    created = client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()

    response = client.get(f"/tools/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_all_tools(client):
    client.post("/tools",
                json={"name": "web-search", "description": "Searches the web"})
    client.post("/tools",
                json={"name": "summarizer", "description": "Summarizes text"})

    response = client.get("/tools")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_update_tool(client):
    created = client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()

    response = client.put(f"/tools/{created['id']}", json={
        "description": "Updated description"
    })
    assert response.status_code == 200
    assert response.json()["description"] == "Updated description"
    assert response.json()["name"] == "web-search"


def test_delete_tool(client):
    created = client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()

    response = client.delete(f"/tools/{created['id']}")
    assert response.status_code == 204

    response = client.get(f"/tools/{created['id']}")
    assert response.status_code == 404


def test_get_tool_not_found(client):
    response = client.get("/tools/nonexistent-id")
    assert response.status_code == 404


def test_tenant_isolation(db, client):
    created = client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()

    from app.main import app

    def other_tenant():
        return "tenant_other"

    app.dependency_overrides[get_tenant] = other_tenant

    from fastapi.testclient import TestClient
    other_client = TestClient(app)

    response = other_client.get(f"/tools/{created['id']}")
    assert response.status_code == 404
