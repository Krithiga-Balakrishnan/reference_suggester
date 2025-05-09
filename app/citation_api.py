# citation_api.py
import os
from fastapi import APIRouter
from pydantic import BaseModel
import torch
import pandas as pd
import ast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# === Setup === #
router = APIRouter()

# Use CPU (no GPU needed)
device = torch.device("cpu")
hf_token = os.getenv("HF_TOKEN", None)  # Ensure you set HF_TOKEN in your environment


# Load model from local path
citation_model_path = "krithigadb/fine-tuned-Flan-T5-citation"
citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path, token=hf_token)
citation_model = AutoModelForSeq2SeqLM.from_pretrained(citation_model_path, token=hf_token).to(device)

# Load dataset
df = pd.read_csv("./models/InteractiveSheet_2025-03-12_16_01_53 - Sheet1.csv")


# === Utility Functions === #
def parse_authors(authors):
    if isinstance(authors, str):
        try:
            authors_list = ast.literal_eval(authors)
            if isinstance(authors_list, list):
                return authors_list
        except Exception:
            return [a.strip() for a in authors.split(",")]
    return authors


def format_author_name(full_name):
    parts = full_name.split()
    if len(parts) > 1:
        initials = " ".join(f"{p[0]}." for p in parts[:-1])
        last_name = parts[-1]
        return f"{initials} {last_name}"
    return full_name


def generate_citation(paper_details):
    authors = parse_authors(paper_details.get("authors", []))
    formatted_authors = ", ".join(format_author_name(author) for author in authors)

    title = paper_details.get("title", "Unknown Title")
    year = paper_details.get("year", "Unknown Year")
    journal = paper_details.get("journal", "Unknown Journal")
    location = paper_details.get("Conference Location", "Unknown Location")
    pages = paper_details.get("pages", "N/A")
    doi = paper_details.get("doi", "N/A")

    input_text = (
        f"Generate an IEEE citation for a research paper titled '{title}' with details: "
        f"Authors: {formatted_authors}, Year: {year}, "
        f"Journal: {journal}, Location: {location}, "
        f"Pages: {pages}, DOI: {doi}."
    )

    input_ids = citation_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        output_ids = citation_model.generate(
            input_ids, max_length=512, num_beams=8, repetition_penalty=2.0, early_stopping=True
        )

    generated_citation = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return generated_citation


# === API Request Schemas === #
class MultiCitationRequest(BaseModel):
    selected_paper_ids: list[str]


@router.post("/generate_citations/")
async def generate_multiple_citations(request: MultiCitationRequest):
    citations = []
    for paper_id in request.selected_paper_ids:
        paper_row = df[df["paper_id"].astype(str).str.strip() == str(paper_id).strip()]
        if paper_row.empty:
            citations.append({"paper_id": paper_id, "citation": "Paper ID not found."})
            continue

        paper_row = paper_row.iloc[0]
        paper_details = {
            "authors": parse_authors(paper_row["authors"]),
            "title": paper_row["title"],
            "journal": paper_row["journal"],
            "year": paper_row["year"],
            "Conference Location": paper_row.get("Conference Location", "Unknown Location"),
            "pages": paper_row.get("pages", "N/A"),
            "doi": paper_row.get("doi", "N/A"),
        }
        citation = generate_citation(paper_details)
        citations.append({"paper_id": paper_id, "citation": citation})

    return {"citations": citations}
