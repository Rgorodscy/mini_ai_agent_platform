import pytest


@pytest.fixture
def tool(client):
    return client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()


def test_create_agent(client, tool):
    response = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Researches topics",
        "tools": [tool["id"]]
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Research Agent"
    assert len(data["tools"]) == 1
    assert data["tools"][0]["id"] == tool["id"]


def test_create_agent_no_tools(client):
    response = client.post("/agents", json={
        "name": "Simple Agent",
        "role": "assistant",
        "description": "A basic agent",
        "tools": []
    })
    assert response.status_code == 201
    assert response.json()["tools"] == []


def test_create_agent_invalid_tool(client):
    response = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Researches topics",
        "tools": ["non-existent-tool-id"]
    })
    assert response.status_code == 404


def test_get_agent(client, tool):
    created = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Researches topics",
        "tools": [tool["id"]]
    }).json()

    response = client.get(f"/agents/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_all_agents(client, tool):
    client.post(
        "/agents",
        json={"name": "Agent 1", "role": "r1", "description": "d1",
              "tools": []})
    client.post(
        "/agents",
        json={"name": "Agent 2", "role": "r2", "description": "d2",
              "tools": []})

    response = client.get("/agents")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_filter_agents_by_tool_name(client, tool):
    client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Researches topics",
        "tools": [tool["id"]]
    })
    client.post("/agents", json={
        "name": "Simple Agent",
        "role": "assistant",
        "description": "A basic agent",
        "tools": []
    })

    response = client.get("/agents?tool_name=web-search")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == "Research Agent"


def test_update_agent(client, tool):
    created = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Researches topics",
        "tools": []
    }).json()

    response = client.put(f"/agents/{created['id']}", json={
        "name": "Updated Agent",
        "tools": [tool["id"]]
    })
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Agent"
    assert len(response.json()["tools"]) == 1


def test_delete_agent(client):
    created = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Researches topics",
        "tools": []
    }).json()

    response = client.delete(f"/agents/{created['id']}")
    assert response.status_code == 204

    response = client.get(f"/agents/{created['id']}")
    assert response.status_code == 404
