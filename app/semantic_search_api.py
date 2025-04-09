# semantic_search_api.py
import os  # Added import for os
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import faiss
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer

# === Setup === #
router = APIRouter()
hf_token = os.getenv("HF_TOKEN", None)  # Ensure you set HF_TOKEN in your environment

# Load models and data from local paths
# keyword_model_path = "./models/suggestionKeywords_V2"
keyword_model_path = "krithigadb/suggestionKeywords_V2"
sbert_model_path = "krithigadb/fine_tuned_sbert"
csv_path = "./models/InteractiveSheet_2025-03-12_16_01_53 - Sheet1.csv"
embedding_path = "./models/sbert_embeddings_latest_v1.pkl"

# Load models
tokenizer = AutoTokenizer.from_pretrained(keyword_model_path, token=hf_token)
model = AutoModelForTokenClassification.from_pretrained(keyword_model_path, token=hf_token)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

sbert_model = SentenceTransformer(sbert_model_path, use_auth_token=hf_token)
df = pd.read_csv(csv_path)

# Load FAISS index
with open(embedding_path, "rb") as f:
    data = pickle.load(f)

paper_ids = data["paper_ids"]
abstracts = data["abstracts"]
titles = data["titles"]
sbert_embeddings = np.array(data["embeddings"])

# Normalize and build FAISS index
faiss.normalize_L2(sbert_embeddings)
dimension = sbert_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(sbert_embeddings)

global_filtered_papers = {}

# === Request Schema === #
class QueryRequest(BaseModel):
    text: str

# === Utilities === #
def get_paper_details(paper_id):
    paper_row = df[df["paper_id"] == paper_id]
    if paper_row.empty:
        return None
    paper_row = paper_row.iloc[0]
    return {
        "paper_id": str(paper_row["paper_id"]),
        "title": str(paper_row["title"]),
        "authors": paper_row["authors"] if pd.notna(paper_row["authors"]) else "Unknown",
        "year": int(paper_row["year"]) if pd.notna(paper_row["year"]) else "Unknown",
        "journal": str(paper_row["journal"]) if pd.notna(paper_row["journal"]) else "Unknown",
        "type": str(paper_row["type"]) if pd.notna(paper_row["type"]) else "Unknown",
        "Conference Location": str(paper_row["Conference Location"]) if pd.notna(paper_row["Conference Location"]) else "N/A",
        "pages": str(paper_row["pages"]) if pd.notna(paper_row["pages"]) else "N/A",
        "volume": str(paper_row["volume"]) if pd.notna(paper_row["volume"]) else "N/A",
        "issue": str(paper_row["issue"]) if pd.notna(paper_row["issue"]) else "N/A",
        "doi": str(paper_row["doi"]) if pd.notna(paper_row["doi"]) else "N/A",
        "Abstract": str(paper_row["Abstract"]) if pd.notna(paper_row["Abstract"]) else "No Abstract Available",
        "keywords": str(paper_row["keywords"]) if pd.notna(paper_row["keywords"]) else "No Keywords Available"
    }

def extract_keywords(text):
    ner_results = nlp(text)
    keywords = {result["word"].lower().strip() for result in ner_results if result["entity_group"] in ['ORG', 'LABEL_1', 'LABEL_2', 'MISC']}
    return list(keywords)

def filter_papers_by_keywords(input_keywords, df, threshold=0.05):
    if not input_keywords or "fine-tuned" not in df.columns:
        return pd.DataFrame()
    input_keywords_normalized = {kw.lower().strip() for kw in input_keywords}
    def compute_match(row):
        if pd.isna(row["fine-tuned"]) or not isinstance(row["fine-tuned"], str):
            return 0, set()
        paper_keywords = {kw.lower().strip() for kw in row["fine-tuned"].split(", ")}
        matched_keywords = input_keywords_normalized.intersection(paper_keywords)
        if matched_keywords:
            return 1, matched_keywords
        match_ratio = len(matched_keywords) / len(input_keywords_normalized)
        return match_ratio, matched_keywords
    df["match_ratio"], df["matched_keywords"] = zip(*df.apply(compute_match, axis=1))
    df["matched_keywords"] = df["matched_keywords"].apply(lambda x: ", ".join(x) if x else "No Match")
    return df[(df["match_ratio"] >= threshold) | (df["match_ratio"] == 1)].sort_values(by="match_ratio", ascending=False)

def find_all_matching_sentences(query_embedding, abstract, top_n=3):
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    if not sentences:
        return []
    sentence_embeddings = sbert_model.encode(sentences, batch_size=32, show_progress_bar=False)
    sentence_embeddings = np.array(sentence_embeddings)
    faiss.normalize_L2(sentence_embeddings)
    scores = sentence_embeddings @ query_embedding[0]
    sorted_indices = np.argsort(-scores)
    return [{"sentence": sentences[i], "similarity": round(scores[i] * 100, 2)} for i in sorted_indices[:top_n]]

def search_similar_papers_all_sentences(query, top_k=10, top_n=3):
    query_embedding = sbert_model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        sim_score = distances[0][rank]
        paper_id = paper_ids[idx]
        abstract = abstracts[idx]
        matched_sentences = find_all_matching_sentences(query_embedding, abstract, top_n)
        details = get_paper_details(paper_id) or {
            "paper_id": paper_id,
            "title": titles[idx],
            "authors": "Unknown",
            "year": "Unknown",
            "journal": "Unknown",
            "type": "Unknown",
            "Conference Location": "N/A",
            "pages": "N/A",
            "volume": "N/A",
            "issue": "N/A",
            "doi": "N/A",
            "Abstract": abstract,
            "keywords": "No Keywords Available",
        }
        details["Abstract"] = abstract
        details["matched_sentences"] = matched_sentences
        details["similarity_percent"] = float(round(sim_score * 100, 2))
        results.append(details)
    return sorted(results, key=lambda x: x["similarity_percent"], reverse=True)

def fully_convert(obj):
    if isinstance(obj, dict):
        return {k: fully_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fully_convert(item) for item in obj]
    if isinstance(obj, tuple):
        return [fully_convert(item) for item in obj]
    if isinstance(obj, set):
        return [fully_convert(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

# === Endpoint === #
@router.post("/search/")
def process_query(request: QueryRequest):
    global global_filtered_papers
    keywords = extract_keywords(request.text)
    filtered_papers = filter_papers_by_keywords(keywords, df)
    if filtered_papers.empty:
        return {"message": "No matching papers found"}
    results = search_similar_papers_all_sentences(request.text, top_k=10, top_n=3)
    final_results = [fully_convert(paper) for paper in results]
    global_filtered_papers = {paper["paper_id"]: paper for paper in results}
    return {
        "keywords": keywords,
        "results": final_results
    }
