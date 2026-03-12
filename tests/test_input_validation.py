import pytest


def test_create_tool_empty_name_rejected(client):
    response = client.post("/tools", json={
        "name": "",
        "description": "Valid description"
    })
    assert response.status_code == 422


def test_create_tool_whitespace_name_rejected(client):
    response = client.post("/tools", json={
        "name": "   ",
        "description": "Valid description"
    })
    assert response.status_code == 422


def test_create_tool_empty_description_rejected(client):
    response = client.post("/tools", json={
        "name": "valid-name",
        "description": ""
    })
    assert response.status_code == 422


def test_create_tool_name_too_long_rejected(client):
    response = client.post("/tools", json={
        "name": "x" * 201,
        "description": "Valid description"
    })
    assert response.status_code == 422


def test_create_tool_name_stripped(client):
    response = client.post("/tools", json={
        "name": "  web-search  ",
        "description": "Searches the web"
    })
    assert response.status_code == 201
    response_json = response.json()
    assert response_json.get("name") == "web-search"


def test_update_tool_empty_name_rejected(client):
    tool = client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()
    response = client.put(f"/tools/{tool['id']}", json={"name": ""})
    assert response.status_code == 422


def test_update_tool_none_name_allowed(client):
    """Partial update — omitting a field is fine."""
    tool = client.post("/tools", json={
        "name": "web-search",
        "description": "Searches the web"
    }).json()
    response = client.put(f"/tools/{tool['id']}", json={
        "description": "Updated description"
    })
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("name") == "web-search"


def test_create_agent_empty_name_rejected(client):
    response = client.post("/agents", json={
        "name": "",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    })
    assert response.status_code == 422


def test_create_agent_empty_role_rejected(client):
    response = client.post("/agents", json={
        "name": "Research Agent",
        "role": "",
        "description": "Valid description",
        "tool_ids": []
    })
    assert response.status_code == 422


def test_create_agent_empty_description_rejected(client):
    response = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "",
        "tool_ids": []
    })
    assert response.status_code == 422


def test_create_agent_name_too_long_rejected(client):
    response = client.post("/agents", json={
        "name": "x" * 201,
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    })
    assert response.status_code == 422


def test_create_agent_name_stripped(client):
    response = client.post("/agents", json={
        "name": "  Research Agent  ",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    })
    assert response.status_code == 201
    response_json = response.json()
    assert response_json.get("name") == "Research Agent"


def test_run_agent_empty_task_rejected(client):
    agent = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    }).json()
    response = client.post(f"/agents/{agent['id']}/run", json={
        "task": "",
        "model": "gpt-4o"
    })
    assert response.status_code == 422


def test_run_agent_whitespace_task_rejected(client):
    agent = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    }).json()
    response = client.post(f"/agents/{agent['id']}/run", json={
        "task": "   ",
        "model": "gpt-4o"
    })
    assert response.status_code == 422


def test_run_agent_task_too_long_rejected(client):
    agent = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    }).json()
    response = client.post(f"/agents/{agent['id']}/run", json={
        "task": "x" * 2001,
        "model": "gpt-4o"
    })
    assert response.status_code == 422


def test_run_agent_invalid_model_returns_400(client):
    """Model validation is in the service — returns 400 not 422."""
    agent = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    }).json()
    response = client.post(f"/agents/{agent['id']}/run", json={
        "task": "valid task",
        "model": "invalid-model"
    })
    assert response.status_code == 400


def test_run_agent_missing_model_returns_422(client):
    """Missing model entirely — schema validation, returns 422."""
    agent = client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    }).json()
    response = client.post(f"/agents/{agent['id']}/run", json={
        "task": "valid task"
    })
    assert response.status_code == 422


@pytest.fixture
def agent_for_pagination(client):
    return client.post("/agents", json={
        "name": "Research Agent",
        "role": "researcher",
        "description": "Valid description",
        "tool_ids": []
    }).json()


def test_page_zero_rejected(client, agent_for_pagination):
    response = client.get(
        f"/agents/{agent_for_pagination['id']}/history?page=0"
    )
    assert response.status_code == 422


def test_negative_page_rejected(client, agent_for_pagination):
    response = client.get(
        f"/agents/{agent_for_pagination['id']}/history?page=-1"
    )
    assert response.status_code == 422


def test_page_size_zero_rejected(client, agent_for_pagination):
    response = client.get(
        f"/agents/{agent_for_pagination['id']}/history?page_size=0"
    )
    assert response.status_code == 422


def test_page_size_over_limit_rejected(client, agent_for_pagination):
    response = client.get(
        f"/agents/{agent_for_pagination['id']}/history?page_size=101"
    )
    assert response.status_code == 422


def test_page_size_at_limit_accepted(client, agent_for_pagination):
    response = client.get(
        f"/agents/{agent_for_pagination['id']}/history?page_size=100"
    )
    assert response.status_code == 200


def test_valid_pagination_accepted(client, agent_for_pagination):
    response = client.get(
        f"/agents/{agent_for_pagination['id']}/history?page=1&page_size=10"
    )
    assert response.status_code == 200
