import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from semantic_search_api import (
    app,  # assuming you mount `router` on FastAPI()
    extract_keywords,
    filter_papers_by_keywords,
    get_paper_details,
    find_all_matching_sentences
)

# --- Fixtures ---
@pytest.fixture(autouse=True)
def dummy_df(monkeypatch):
    # Create a small DataFrame with known values
    data = pd.DataFrame([
        {'paper_id': 'A', 'fine-tuned': 'nlp, ai', 'title': 'A', 'authors': 'X', 'year': 2020, 'journal': 'J', 'type': 'Conf', 'Conference Location': '', 'pages': '', 'volume': '', 'issue': '', 'doi': '', 'Abstract': 'This is an NLP paper about AI.', 'keywords': ''},
        {'paper_id': 'B', 'fine-tuned': 'vision, ml', 'title': 'B', 'authors': 'Y', 'year': 2021, 'journal': 'J2', 'type': 'Journal', 'Conference Location': '', 'pages': '', 'volume': '', 'issue': '', 'doi': '', 'Abstract': 'Computer vision and ML.', 'keywords': ''},
    ])
    monkeypatch.setattr('semantic_search_api.df', data)
    return data

# --- Unit tests for utility functions ---
def test_extract_keywords_simple():
    text = "Research by OpenAI on GPT-3 and BERT"
    kws = extract_keywords(text)
    assert 'openai' in kws
    assert 'gpt-3' in kws or 'gpt' in kws


def test_filter_papers_by_keywords_match(dummy_df):
    kws = ['nlp']
    filtered = filter_papers_by_keywords(kws, dummy_df, threshold=0.01)
    assert not filtered.empty
    assert 'A' in filtered['paper_id'].values


def test_filter_papers_by_keywords_no_match(dummy_df):
    kws = ['quantum']
    filtered = filter_papers_by_keywords(kws, dummy_df)
    assert filtered.empty


def test_get_paper_details_present(dummy_df):
    details = get_paper_details('A')
    assert details['paper_id'] == 'A'
    assert details['title'] == 'A'


def test_get_paper_details_missing(dummy_df):
    details = get_paper_details('Z')
    assert details is None

# --- Endpoint test ---
@pytest.fixture

def client(monkeypatch):
    from fastapi import FastAPI
    from semantic_search_api import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_search_endpoint_no_match(client, dummy_df):
    response = client.post('/search/', json={'text': 'quantum computing'})
    assert response.status_code == 200
    assert response.json() == {"message": "No matching papers found"}

# --- Additional tests for sentence matching (mocking SBERT) ---
class DummyEmbedder:
    def encode(self, texts, batch_size=None, show_progress_bar=None):
        # Return identity-like vectors for reproducibility
        return [[float(len(t)) for _ in range(4)] for t in texts]

@pytest.fixture(autouse=True)
def patch_sbert(monkeypatch):
    # Replace the SBERT model with dummy
    import semantic_search_api
    monkeypatch.setattr(semantic_search_api, 'sbert_model', DummyEmbedder())
    # Also patch FAISS index search to return a simple match
    class DummyIndex:
        def search(self, qemb, top_k):
            return np.array([[1.0]]), np.array([[0]])
    monkeypatch.setattr(semantic_search_api, 'index', DummyIndex())
    # Provide minimal abstracts and ids
    monkeypatch.setattr(semantic_search_api, 'paper_ids', ['X'])
    monkeypatch.setattr(semantic_search_api, 'abstracts', ['Sentence one. Sentence two!'])
    return


def test_find_all_matching_sentences_basic():
    emb = np.array([[1.0, 1.0, 1.0, 1.0]])
    matches = find_all_matching_sentences(emb, "Hello world. Testing.")
    assert isinstance(matches, list)
    assert all('sentence' in m for m in matches)
    assert all('similarity' in m for m in matches)