import pytest
from fastapi.testclient import TestClient
from main import app   # assuming your FastAPI instance lives in main.py

client = TestClient(app)

def test_generate_multiple_citations_not_found():
    resp = client.post("/generate_citations/", json={"selected_paper_ids": ["not-a-real-id"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "citations" in data
    assert data["citations"][0]["citation"] == "Paper ID not found."

def test_generate_multiple_citations_happy_path(monkeypatch):
    # monkeypatch the dataframe lookup so you don't need a real model or CSV
    fake_row = {
        "authors": "['Alice Smith']",
        "title": "Test Title",
        "journal": "Test Journal",
        "year": 2021,
        "Conference Location": "Town",
        "pages": "1-10",
        "doi": "10.1234/test",
        "url": None,
        "volume": None,
        "issue": None,
        "type": "journal"
    }
    # Patch pandas.read_csv to return a DataFrame with one paper_id
    import pandas as pd
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame(
        [{**fake_row, "paper_id": "42"}]
    ))
    resp = client.post("/generate_citations/", json={"selected_paper_ids": ["42"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["citations"]) == 1
    assert "Alice" in data["citations"][0]["citation"]
