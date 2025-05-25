from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_manual_citation_minimal():
    payload = {
      "paper_id": None,
      "authors": ["Alice Smith"],
      "title": "My Paper",
      "journal": "JournaL",
      "year": 2022,
      "location": "City",
      "pages": "100-110",
      "doi": "10.xxxx/yyy",
      "type": "journal",
      "volume": 5,
      "issue": 2,
      "url": None
    }
    resp = client.post("/generate_manual_citation/", json=payload)
    assert resp.status_code == 200
    out = resp.json()["citation"]
    assert "A. Smith" in out
    assert "My Paper" in out
    assert "2022" in out
