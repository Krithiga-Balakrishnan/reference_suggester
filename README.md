
# Reference Suggestor & Citation Generator API

This repository contains the **backend microservice** for a research paper **reference suggestor** and **IEEE-style citation generator**, built with **FastAPI**, **Transformers**, and **Sentence Transformers**.

It uses **custom fine-tuned models** hosted on Hugging Face Hub:
- **Citation Model:** [`krithigadb/fine-tuned-Flan-T5-citation`](https://huggingface.co/krithigadb/fine-tuned-Flan-T5-citation)
- **Keyword Suggestion Model:** [`krithigadb/suggestionKeywords_V2`](https://huggingface.co/krithigadb/suggestionKeywords_V2)
- **Semantic Search SBERT Model:** [`krithigadb/fine_tuned_sbert`](https://huggingface.co/krithigadb/fine_tuned_sbert)

---

## Live Deployment

Try it live on **Hugging Face Spaces**: [krithigadb/reference-suggesting-citation](https://huggingface.co/spaces/krithigadb/reference-suggesting-citation)

---

## Project Structure

```
.
├── app/
│   ├── citation_api.py           # API for dataset-based citation generation
│   ├── manual_citation_api.py    # API for manual citation generation
│   ├── semantic_search_api.py    # API for keyword extraction & semantic search
├── models/
│   ├── InteractiveSheet_*.csv    # Paper metadata
│   ├── sbert_embeddings_latest_v1.pkl  # Precomputed SBERT embeddings
├── main.py                       # FastAPI app entrypoint
├── requirements.txt              # Python dependencies
```

---

## Features

 **Semantic Search** — Extracts keywords from input, filters relevant papers, finds semantically similar papers using **FAISS** and **SBERT**, and highlights matching sentences.

 **Citation Generation** — Generates IEEE citations automatically using a fine-tuned **Flan-T5** model.

 **Manual Citation** — Create citations manually by providing custom details.

---

##  Installation

1 **Clone the repo**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2 **Add your `.env`**
```
HF_TOKEN=your_huggingface_access_token
```

3 **Install dependencies**
```bash
pip install -r requirements.txt
```

4 **Run locally**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Endpoint | Method | Description |
|-------------------------------|--------|---------------------------------------------|
| `/` | GET | Health check |
| `/semantic/search/` | POST | Perform semantic keyword extraction & search |
| `/citation/generate_citations/` | POST | Generate citations for selected paper IDs |
| `/manual/generate_manual_citation/` | POST | Generate IEEE citation from manual input |

---

## Example: Semantic Search

**Request**
```json
POST /semantic/search/

{
  "text": "I want papers on transformer models for text summarization"
}
```

**Response**
- Extracted keywords
- Filtered papers
- Similarity scores
- Most similar sentences

---

## Example: Manual Citation

**Request**
```json
POST /manual/generate_manual_citation/

{
  "authors": ["John Doe", "Jane Smith"],
  "title": "Deep Learning for NLP",
  "journal": "Journal of AI Research",
  "year": 2024,
  "location": "New York",
  "pages": "12-20",
  "doi": "10.1234/example-doi"
}
```

---

## Deployment

This backend is already live on **Hugging Face Spaces**:  
 [krithigadb/reference-suggesting-citation](https://huggingface.co/spaces/krithigadb/reference-suggesting-citation)

You can also deploy it on **Azure**, **Render**, or any VM with Python.

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

Made with by **Krithiga D B**

- [Hugging Face Profile](https://huggingface.co/krithigadb)

---

## Contributing

PRs and issues are welcome — help improve the reference suggestor and citation generator!
