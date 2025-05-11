from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_semantic_search_basic():
    response = client.post("/semantic-search", json={"query": "heat"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    assert all("english" in r and "elvish" in r and "definition" in r for r in results)

def test_semantic_search_with_exact_word():
    response = client.post("/semantic-search", json={"query": "fire"})
    assert response.status_code == 200
    results = response.json()["results"]
    terms = [r["english"] for r in results]
    assert "fire" in terms or any("fire" in r["definition"].lower() for r in results)

def test_semantic_search_empty_query():
    response = client.post("/semantic-search", json={"query": ""})
    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)

def test_semantic_search_nonexistent_word():
    response = client.post("/semantic-search", json={"query": "xyzblargle"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)
    assert len(results) > 0

def test_semantic_search_special_characters():
    response = client.post("/semantic-search", json={"query": "@#$%&*()"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)

def test_semantic_search_missing_query_field():
    response = client.post("/semantic-search", json={})
    assert response.status_code == 422
