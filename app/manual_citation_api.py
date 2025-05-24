# manual_citation_api.py
import os
from fastapi import APIRouter
from pydantic import BaseModel
import torch
import pandas as pd
import ast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# === Setup === #
router = APIRouter()

device = torch.device("cpu")  # Use CPU
hf_token = os.getenv("HF_TOKEN", None)  # Ensure you set HF_TOKEN in your environment


# Load model and tokenizer
citation_model_path = "krithigadb/fine-tuned-citation-flan-t5"
# citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path)
citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path, token=hf_token)
citation_model = AutoModelForSeq2SeqLM.from_pretrained(citation_model_path, token=hf_token).to(device)

# citation_model = AutoModelForSeq2SeqLM.from_pretrained(citation_model_path).to(device)


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


# def format_author_name(full_name):
#     parts = full_name.split()
#     if len(parts) > 1:
#         initials = " ".join(f"{p[0]}." for p in parts[:-1])
#         last_name = parts[-1]
#         return f"{initials} {last_name}"
#     return full_name
def format_author_name(full_name):
    parts = full_name.split()
    if len(parts) > 1:
        initials = " ".join(f"{p[0]}." for p in parts[:-1])
        last_name = parts[-1]
        return f"{initials} {last_name}"
    return full_name

def format_author_list(authors):
    formatted = [format_author_name(a) for a in authors]
    n = len(formatted)
    if n == 0:
        return "Unknown Author"
    elif n == 1:
        return formatted[0]
    elif n == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        # commas between 1st and 2nd ... no comma before 'and' between last two
        return ", ".join(formatted[:-2]) + ", " + formatted[-2] + " and " + formatted[-1]

def get_valid_field(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in {"nan", "n/a", "#n/a", ""}:
        return None
    return s



# def generate_citation(paper_details):
#     authors = parse_authors(paper_details.get("authors", []))
#     formatted_authors = ", ".join(format_author_name(author) for author in authors)

#     title = paper_details.get("title", "Unknown Title")
#     year = paper_details.get("year", "Unknown Year")
#     journal = paper_details.get("journal", "Unknown Journal")
#     location = paper_details.get("Conference Location", "Unknown Location")
#     pages = paper_details.get("pages", "N/A")
#     doi = paper_details.get("doi", "N/A")

#     input_text = (
#         f"Generate an IEEE citation for a research paper titled '{title}' with details: "
#         f"Authors: {formatted_authors}, Year: {year}, "
#         f"Journal: {journal}, Location: {location}, "
#         f"Pages: {pages}, DOI: {doi}."
#     )

#     input_ids = citation_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)

#     with torch.no_grad():
#         output_ids = citation_model.generate(
#             input_ids, max_length=512, num_beams=8, repetition_penalty=2.0, early_stopping=True
#         )

#     generated_citation = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
#     return generated_citation

def generate_citation(paper_details):
    # — Get formatted fields —
    authors = parse_authors(paper_details.get("authors", []))
    formatted_authors = format_author_list(authors)
    title     = paper_details.get("title", "Unknown Title")
    year      = get_valid_field(paper_details.get("year"))
    journal   = get_valid_field(paper_details.get("journal"))
    location  = get_valid_field(paper_details.get("Conference Location"))
    pages     = get_valid_field(paper_details.get("pages"))
    doi       = get_valid_field(paper_details.get("doi"))
    citation_type = paper_details.get("type", "").lower()
    volume    = get_valid_field(paper_details.get("volume"))
    issue     = get_valid_field(paper_details.get("issue"))
    url = get_valid_field(paper_details.get("url"))


    # — Compose prompt based on citation type —
    if "conference" in citation_type:
        type_description = "conference paper"
        input_text = (
            f"Generate an IEEE citation for a {type_description} titled '{title}' with details: "
            f"Authors: {formatted_authors}, Year: {year}, Conference: {journal}, "
            f"Location: {location}, Pages: {pages}, DOI: {doi}, URL: {url}."
        )
        
    elif "journal" in citation_type:
        type_description = "journal article"
        input_text = (
            f"Generate an IEEE citation for a {type_description} titled '{title}' with details: "
            f"Authors: {formatted_authors}, Year: {year}, Journal: {journal}, "
            f"Volume: {volume}, Issue: {issue}, Pages: {pages}, DOI: {doi}, URL: {url}."
        )
       
    else:
        type_description = "research paper"
        input_text = (
            f"Generate an IEEE citation for a {type_description} titled '{title}' with details: "
            f"Authors: {formatted_authors}, Year: {year}, Publication: {journal}, "
            f"Location: {location}, Pages: {pages}, DOI: {doi}."
        )

    # # — Generate from model —
    # input_ids = citation_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    # with torch.no_grad():
    #     output_ids = citation_model.generate(
    #         input_ids,
    #         max_length=512,
    #         num_beams=8,
    #         repetition_penalty=2.0,
    #         early_stopping=True
    #     )

    # generated_citation = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # # — Post-process output —
    # generated_citation = re.sub(r'\bAvailable:\s*', '', generated_citation)
    # generated_citation = generated_citation.replace("*", "")

    # if "journal" in citation_type and not re.search(r'\bin\s+\b' + re.escape(journal), generated_citation):
    #     generated_citation = re.sub(
    #         r'("[^"]+", )([A-Z][^,]+,)',
    #         lambda m: f'{m.group(1)}in {m.group(2)}',
    #         generated_citation,
    #         count=1
    #     )

    # return generated_citation

    # Run Model
    input_ids = citation_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = citation_model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            repetition_penalty=2.0,
            early_stopping=True
        )

    generated_citation = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Post-processing cleanup
    generated_citation = re.sub(r'\bAvailable:\s*', '', generated_citation)
    generated_citation = generated_citation.replace("*", "")

    if "journal" in citation_type and not re.search(r'\bin\s+\b' + re.escape(journal), generated_citation):
        generated_citation = re.sub(
            r'("[^"]+", )([A-Z][^,]+,)',
            lambda m: f'{m.group(1)}in {m.group(2)}',
            generated_citation,
            count=1
        )

    return generated_citation


# === API Schema === #
class CitationRequest(BaseModel):
    paper_id: str = None
    authors: list = None
    title: str = None
    journal: str = None
    year: int = None
    location: str = None
    pages: str = None
    doi: str = None
    type: str = None  # ADD THIS
    volume: int = None
    issue: int = None
    url: str = None


# === API Endpoint === #
@router.post("/generate_manual_citation/")
async def generate_manual_citation(request: CitationRequest):
    paper_details = {
        "authors": request.authors or [],
        "title": request.title or "Unknown Title",
        "journal": request.journal or "Unknown Journal",
        "year": request.year or "Unknown Year",
        "Conference Location": request.location or "Unknown Location",
        "pages": request.pages or "N/A",
        "doi": request.doi or "N/A",
        "Conference Location": request.location or "Unknown Location",
        "type": request.type or "",
        "volume": request.volume,
        "issue": request.issue,
        "url": request.url

    }

    citation = generate_citation(paper_details)
    return {"citation": citation}
