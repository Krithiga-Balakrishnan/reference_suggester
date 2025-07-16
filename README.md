
# Reference Suggesting & Citation Generation API

This repository provides a deployed FastAPI backend for generating IEEE-style citations using a fine-tuned FLAN-T5 model and performing semantic search on research papers with keyword extraction (NER) + SBERT embeddings + FAISS.

It uses **custom fine-tuned models** hosted on Hugging Face Hub:
- **Citation Model:** [`krithigadb/fine-tuned-Flan-T5-citation`](https://huggingface.co/krithigadb/fine-tuned-Flan-T5-citation)
- **Keyword Suggestion Model:** [`krithigadb/suggestionKeywords_V2`](https://huggingface.co/krithigadb/suggestionKeywords_V2)
- **Semantic Search SBERT Model:** [`krithigadb/fine_tuned_sbert`](https://huggingface.co/krithigadb/fine_tuned_sbert)

---

 **Live on Hugging Face Spaces:** [reference-suggesting-citation](https://huggingface.co/spaces/krithigadb/reference-suggesting-citation)

---

## Features

- **Automatic Citation Generation**: Generates IEEE citations for selected papers.
- **Manual Citation**: Users can manually input details to generate citations.
- **Semantic Search**: Find relevant papers by extracting keywords + SBERT semantic similarity + sentence-level matches.
- **FastAPI**: Organized modular routes with `/generate`, `/manual`, and `/semantic` endpoints.

---

## Project Structure

```
.
├── app.py                # Entry point for Hugging Face Spaces (runs main.py app instance)
├── main.py               # Creates FastAPI app, includes routers
├── app/
│   ├── citation_api.py         # Multi-paper citation API
│   ├── manual_citation_api.py  # Manual single citation API
│   ├── semantic_search_api.py  # Semantic search with keyword filter + SBERT + FAISS
├── models/
│   ├── InteractiveSheet_2025-03-12_16_01_53 - Sheet1.csv   # Papers metadata
│   ├── sbert_embeddings_latest_v1.pkl  # Precomputed SBERT embeddings
│   ├── sbert_faiss.index               # FAISS index for fast similarity search
├── requirements.txt     # Python dependencies
├── pytest.ini           # Pytest configuration
└── Test/
    └──data
        └──qrels.jsonl
        └──queries.jsonl
    ├── test_citation_api.py
    ├── test_manual_citation_api.py
    ├── test_semantic_search_api.py
    ├── test_utils.py
    ├── evaluate.py
```

---

## API Endpoints

### Citation API (`/generate`)
- POST `/generate/citation/`  
  Generate citations for selected paper IDs.

### Manual Citation API (`/manual`)
- POST `/manual/generatecitation/`  
  Generate citation by manually providing metadata.

### Semantic Search API (`/semantic`)
- POST `/semantic/search/`  
  Extracts keywords → filters by keyword match → reranks using SBERT + FAISS → returns top sentences.

---

## Models Used

- **FLAN-T5**: Fine-tuned for citation generation.
- **Keyword Extractor**: Custom NER model for domain-specific keyword extraction.
- **Sentence-BERT**: Fine-tuned for research paper similarity.
- **FAISS**: For fast approximate nearest neighbors search.

---

## Deployment

Deployed on Hugging Face Spaces.
Ensure `HF_TOKEN` is set for private models.

**Files to ensure:**  
- `models/InteractiveSheet_2025-03-12_16_01_53 - Sheet1.csv`
- `models/sbert_embeddings_latest_v1.pkl`
- `models/sbert_faiss.index`

---

## Testing

Tests are inside the `Test/` folder:  
- `pytest` is configured via `pytest.ini`.

Run all tests:  
```bash
pytest
```

---

## Installation

```bash
pip install -r requirements.txt
```

Start locally:  
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Contributing

PRs & suggestions are welcome!

---

---

## Tech Stack

- **FastAPI** — Web framework
- **Transformers** — For model inference
- **Sentence Transformers** — For semantic similarity
- **FAISS** — Vector index for fast similarity search
- **Torch** — Deep learning backend
- **pandas**, **numpy** — Data processing

---

## Author

Made by **Krithiga D B**

- [Hugging Face Profile](https://huggingface.co/krithigadb)

---

##  Author

**Krithiga D. Balakrishnan**  
[GitHub](https://github.com/Krithiga-Balakrishnan) | [Portfolio](https://krithiga-balakrishnan.vercel.app)

---
