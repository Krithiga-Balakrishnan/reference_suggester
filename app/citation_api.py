# citation_api.py
import os
from fastapi import APIRouter
from pydantic import BaseModel
import torch
import pandas as pd
import ast
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# === Setup === #
router = APIRouter()

# Use CPU (no GPU needed)
device = torch.device("cpu")
hf_token = os.getenv("HF_TOKEN", None)  # Ensure you set HF_TOKEN in your environment


# Load model from local path
citation_model_path = "krithigadb/fine-tuned-citation-flan-t5"
# citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path, token=hf_token)
citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path, use_fast=False)
citation_model = AutoModelForSeq2SeqLM.from_pretrained(citation_model_path).to(device)

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
    low = s.lower()
    # treat literally missing, nan, n/a, or any "unknown ..." as invalid
    if not s or low in {"nan", "n/a", "#n/a"} or low.startswith("unknown"):
        return None
    return s

def generate_citation(paper_details):
    import re

    # — Parse and format authors —
    authors = parse_authors(paper_details.get("authors", []))
    formatted_authors = format_author_list(authors)

    # — Clean and validate fields —
    title     = get_valid_field(paper_details.get("title", "Unknown Title"))
    year      = get_valid_field(paper_details.get("year"))
    journal   = get_valid_field(paper_details.get("journal"))
    location  = get_valid_field(paper_details.get("Conference Location"))
    pages     = get_valid_field(paper_details.get("pages"))
    volume    = get_valid_field(paper_details.get("volume"))
    issue     = get_valid_field(paper_details.get("issue"))
    citation_type = get_valid_field(paper_details.get("type", "")).lower()
    doi = get_valid_field(paper_details.get("doi"))
    url = get_valid_field(paper_details.get("url"))

    # 🛡️ Remove URL if it is just the expanded form of the DOI
    if doi and url:
        clean_doi = doi.strip().lower().replace("https://doi.org/", "")
        clean_url = url.strip().lower()
        if clean_url == f"https://doi.org/{clean_doi}":
            url = None  # prevent duplicate


    # — Build prompt based on type —
    if "conference" in citation_type:
        input_text = (
            f"Generate an IEEE citation for a conference paper titled '{title}' with details: "
            f"Authors: {formatted_authors}, Year: {year}, Conference: {journal}, "
            f"Location: {location}, Pages: {pages}, doi: {doi}, URL: {url}."
        )
    elif "journal" in citation_type:
        input_text = (
            f"Generate an IEEE citation for a journal article titled '{title}' with details: "
            f"Authors: {formatted_authors}, Year: {year}, Journal: {journal}, "
            f"Volume: {volume}, Issue: {issue}, Pages: {pages}, doi: {doi}, URL: {url}."
        )
    else:
        input_text = (
            f"Generate an IEEE citation for a research paper titled '{title}' with details: "
            f"Authors: {formatted_authors}, Year: {year}, Publication: {journal}, "
            f"Location: {location}, Pages: {pages}, doi: {doi}, URL: {url}."
        )

    # — Run model inference —
    input_ids = citation_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = citation_model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            repetition_penalty=2.0,
            early_stopping=True
        )
    generated = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # generated = generated.replace('*', '')
    # Remove markdown and malformed DOI+URL
    generated = re.sub(r'\*(.*?)\*', r'\1', generated)
    generated = re.sub(r'(doi:\s*\S+?)https?://doi\.org/\S+', r'\1', generated)

    # Strip trailing period safely
    generated = generated.rstrip('. ')

    # Append DOI if missing
    if doi and f"doi: {doi}" not in generated:
        generated += f", doi: {doi}"

    # Append URL if present
    if url and f"Available: {url}" not in generated:
        generated += f". [Online]. Available: {url}"

    # Final punctuation
    generated += "."

    # — Validate model output —
    if len(generated.split()) < 4 or not any(char.isdigit() for char in generated):
        # Fallback manual citation
        parts = [formatted_authors, f"\"{title},\""]
        if "journal" in citation_type:
            if journal: parts.append(journal)
            if volume: parts.append(f"vol. {volume}")
            if issue: parts.append(f"no. {issue}")
            if year:  parts.append(str(year))
            if pages: parts.append(f"pp. {pages}")
        elif "conference" in citation_type:
            if journal: parts.append(journal)
            if location: parts.append(location)
            if year:     parts.append(str(year))
            if pages:    parts.append(f"pp. {pages}")
        else:
            if journal: parts.append(journal)
            if year:    parts.append(str(year))
            if pages:   parts.append(f"pp. {pages}")
        if doi: parts.append(f"doi: {doi}")
        if url: parts.append(url)
        return ", ".join(p for p in parts if p) + "."

    # — Post-process model output —
    # generated = re.sub(r'\bAvailable:\s*', '', generated)
    generated = re.sub(r'\*(.*?)\*', r'\1', generated)
    if "journal" in citation_type and not re.search(r'\bin\s+\b' + re.escape(journal), generated):
        generated = re.sub(
            r'("[^"]+", )([A-Z][^,]+,)',
            lambda m: f'{m.group(1)}in {m.group(2)}',
            generated,
            count=1
        )


    return generated

import re

def get_valid_field(val):
    """
    Return a cleaned string, or None for any 'empty' placeholder:
    None, '', 'nan', 'n/a', '#n/a', or anything starting with 'unknown'.
    """
    if val is None:
        return None
    s = str(val).strip()
    low = s.lower()
    if not s or low in {"nan", "n/a", "#n/a"} or low.startswith("unknown"):
        return None
    return s

def generate_citation(paper_details):
    # — Parse & format authors —
    authors = parse_authors(paper_details.get("authors", []))
    formatted_authors = format_author_list(authors)

    # — Clean & validate each field —
    title     = get_valid_field(paper_details.get("title"))
    year      = get_valid_field(paper_details.get("year"))
    journal   = get_valid_field(paper_details.get("journal"))
    location  = get_valid_field(paper_details.get("Conference Location"))
    pages     = get_valid_field(paper_details.get("pages"))
    # doi       = get_valid_field(paper_details.get("doi"))
    # url       = get_valid_field(paper_details.get("url"))
    volume    = get_valid_field(paper_details.get("volume"))
    issue     = get_valid_field(paper_details.get("issue"))
    citation_type = (get_valid_field(paper_details.get("type")) or "").lower()
    doi = get_valid_field(paper_details.get("doi"))
    url = get_valid_field(paper_details.get("url"))

    # 🛡️ Remove URL if it is just the expanded form of the DOI
    if doi and url:
        clean_doi = doi.strip().lower().replace("https://doi.org/", "")
        clean_url = url.strip().lower()
        if clean_url == f"https://doi.org/{clean_doi}":
            url = None  # prevent duplicate

    # — Collect only the fields we actually have —
    details = []
    if formatted_authors:
        details.append(f"Authors: {formatted_authors}")
    if year:
        details.append(f"Year: {year}")

    if "conference" in citation_type:
        if journal:  details.append(f"Conference: {journal}")
        if location: details.append(f"Location: {location}")
    elif "journal" in citation_type:
        if journal: details.append(f"Journal: {journal}")
        if volume:  details.append(f"Volume: {volume}")
        if issue:   details.append(f"Issue: {issue}")
    else:
        if journal: details.append(f"Publication: {journal}")

    # common
    if pages: details.append(f"Pages: {pages}")
    if doi:   details.append(f"doi: {doi}")
    if url:   details.append(f"URL: {url}")

    details_text = ", ".join(details)

    # — Build the model prompt —
    kind = "conference paper" if "conference" in citation_type else "journal article"
    if title:
        prompt = f"Generate an IEEE citation for a {kind} titled '{title}'"
    else:
        prompt = f"Generate an IEEE citation for a {kind}"
    if details_text:
        prompt += f" with details: {details_text}."
    else:
        prompt += "."

    # — Run model inference —
    input_ids = citation_tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = citation_model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            repetition_penalty=2.0,
            early_stopping=True
        )
    generated = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # Remove trailing period for safe appending
    # Remove markdown and malformed DOI+URL
    generated = re.sub(r'\*(.*?)\*', r'\1', generated)
    generated = re.sub(r'(doi:\s*\S+?)https?://doi\.org/\S+', r'\1', generated)

    # Strip trailing period safely
    generated = generated.rstrip('. ')

    # Append DOI if missing
    if doi and f"doi: {doi}" not in generated:
        generated += f", doi: {doi}"

    # Append URL if present
    if url and f"Available: {url}" not in generated:
        generated += f". [Online]. Available: {url}"

    # Final punctuation
    generated += "."

    # — Fallback if the model output looks invalid —
    if len(generated.split()) < 4 or not any(char.isdigit() for char in generated):
        parts = [formatted_authors, f"\"{title},\""] if title else []
        if "journal" in citation_type:
            if journal: parts.append(journal)
            if volume:  parts.append(f"vol. {volume}")
            if issue:   parts.append(f"no. {issue}")
        elif "conference" in citation_type:
            if journal:  parts.append(journal)
            if location: parts.append(location)
        if year:   parts.append(str(year))
        if pages:  parts.append(f"pp. {pages}")
        if doi:    parts.append(f"doi: {doi}")
        if url:    parts.append(url)
        return ", ".join(p for p in parts if p) + "."

    # — Clean up any leftover markers —
    # generated = re.sub(r'\bAvailable:\s*', '', generated)
    generated = re.sub(r'\b(None|N/A|Unknown\s+\w+)\b,?\s*', '', generated)
    

    return generated

# === API Request Schemas === #
class MultiCitationRequest(BaseModel):
    selected_paper_ids: list[str]


@router.post("/citation/")
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
        }

         # Optional fields to check and add only if they are valid
        optional_fields = [
            "Conference Location", "pages", "doi", "url", "volume", "issue", "type"
        ]
        for field in optional_fields:
            value = paper_row.get(field)
            if isinstance(value, str):
                value = value.strip()
            if value not in [None, "", float("nan"), "N/A"] and not pd.isna(value):
                paper_details[field] = value.lower() if field == "type" and isinstance(value, str) else value
                        
        citation = generate_citation(paper_details)
        citations.append({"paper_id": paper_id, "citation": citation})

    return {"citations": citations}

