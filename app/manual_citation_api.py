# manual_citation_api.py
import os
from fastapi import APIRouter
from pydantic import BaseModel
import torch
import pandas as pd
import ast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re

# === Setup === #
router = APIRouter()

device = torch.device("cpu")  # Use CPU
hf_token = os.getenv("HF_TOKEN", None)  # Ensure you set HF_TOKEN in your environment


# Load model and tokenizer
citation_model_path = "krithigadb/fine-tuned-citation-flan-t5"
# citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path)
# citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path, token=hf_token)
citation_tokenizer = AutoTokenizer.from_pretrained(citation_model_path, use_fast=False)
# citation_model = AutoModelForSeq2SeqLM.from_pretrained(citation_model_path, token=hf_token).to(device)
citation_model = AutoModelForSeq2SeqLM.from_pretrained(citation_model_path).to(device)

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
    # drop empty, common placeholders, or anything starting with 'unknown'
    if not s or low in {"nan", "n/a", "#n/a"} or low.startswith("unknown"):
        return None
    return s


def generate_citation(paper_details):
    # Parse and clean fields
    authors = parse_authors(paper_details.get("authors", []))
    formatted_authors = format_author_list(authors)

    title = get_valid_field(paper_details.get("title"))
    year = get_valid_field(paper_details.get("year"))
    journal = get_valid_field(paper_details.get("journal"))
    location = get_valid_field(paper_details.get("Conference Location"))
    pages = get_valid_field(paper_details.get("pages"))
    volume = get_valid_field(paper_details.get("volume"))
    issue = get_valid_field(paper_details.get("issue"))
    citation_type = (get_valid_field(paper_details.get("type")) or "").lower()
    doi = get_valid_field(paper_details.get("doi"))
    url = get_valid_field(paper_details.get("url"))

    # Avoid repeating DOI in both `doi:` and `URL:` form
    if doi and url:
        doi_stripped = doi.strip().lower().lstrip("https://doi.org/")
        url_stripped = url.strip().lower().lstrip()
        if url_stripped == f"https://doi.org/{doi_stripped}":
            url = None  # Redundant URL


    # Build list of available detail strings
    details = []
    if formatted_authors:
        details.append(f"Authors: {formatted_authors}")
    if year:
        details.append(f"Year: {year}")

    # Type-specific fields
    if "conference" in citation_type:
        if journal:
            details.append(f"Conference: {journal}")
        if location:
            details.append(f"Location: {location}")
    elif "journal" in citation_type:
        if journal:
            details.append(f"Journal: {journal}")
        if volume:
            details.append(f"Volume: {volume}")
        if issue:
            details.append(f"Issue: {issue}")
    else:
        if journal:
            details.append(f"Publication: {journal}")

    # Common optional fields
    if pages:
        details.append(f"Pages: {pages}")
    if doi:
        details.append(f"doi: {doi}")
    if url:
        details.append(f"URL: {url}")

    # Build prompt
    kind = "conference paper" if "conference" in citation_type else "journal article"
    if title:
        prompt = f"Generate an IEEE citation for a {kind} titled '{title}'"
    else:
        prompt = f"Generate an IEEE citation for a {kind}"
    if details:
        prompt += f" with details: {', '.join(details)}."
    else:
        prompt += "."

    # Model inference
    input_ids = citation_tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = citation_model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            repetition_penalty=2.0,
            early_stopping=True
        )
    # Decode model output
    generated = citation_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
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

    # Remove hallucinated year/volume/pages if not provided
    if not year:
        generated = re.sub(r'(,?\s*)(19|20)\d{2}(?=,|\.|\s|$)', '', generated)
    # Insert missing volume if provided but not present
    if volume and not re.search(r'\bvol\.?\s*\d+', generated, flags=re.IGNORECASE):
        # Insert right after journal name, e.g., "in JournalName" → "in JournalName, vol. X"
        generated = re.sub(
            r'(in\s+[A-Z][^,\.]+)',
            rf'\1, vol. {volume}',
            generated,
            count=1,
            flags=re.IGNORECASE
        )

    if not pages:
        generated = re.sub(r'\bpp\.?\s*\d+(-\d+)?[.,]?', '', generated, flags=re.IGNORECASE)
    if not issue:
        generated = re.sub(r'\bno\.?\s*\d+[.,]?', '', generated, flags=re.IGNORECASE)
    if not doi:
        generated = re.sub(r'doi:\s*\S+[.,]?', '', generated, flags=re.IGNORECASE)


    # Fix missing commas between fields like "no. 1 2023" → "no. 1, 2023"
    generated = re.sub(r'(\b(?:vol\.?|no\.?)\s*\d+)\s+(19|20)\d{2}', r'\1, \2', generated, flags=re.IGNORECASE)
    # Clean residual punctuation/spacing
    generated = re.sub(r'\s{2,}', ' ', generated)
    generated = re.sub(r',\s*,', ',', generated)
    generated = generated.strip(' ,.')
    generated = re.sub(r'\*(.*?)\*', r'\1', generated)  # ✅ Strip *Science* to Science

    # Fallback if output invalid
    if len(generated.split()) < 4 or not any(ch.isdigit() for ch in generated):
        parts = []
        if formatted_authors:
            parts.append(formatted_authors)
        if title:
            parts.append(f'"{title},"')
        if "journal" in citation_type and journal:
            parts.append(journal)
            if volume:
                parts.append(f"vol. {volume}")
            if issue:
                parts.append(f"no. {issue}")
        if "conference" in citation_type and journal:
            parts.append(journal)
            if location:
                parts.append(location)
        if year:
            parts.append(str(year))
        if pages:
            parts.append(f"pp. {pages}")
        if doi:
            parts.append(f"doi: {doi}")
        if url:
            parts.append(url)
        return ", ".join(parts) + "."

    # Clean up any stray placeholders
    # generated = re.sub(r'\bAvailable:\s*', '', generated)
    generated = re.sub(r'\b(None|N/A|Unknown\s+\w+)\b,?\s*', '', generated)

    return generated

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
@router.post("/generatecitation/")
async def generate_manual_citation(request: CitationRequest):
    paper_details = {
        "authors": request.authors or [],
        "title": request.title or "",
        "journal": request.journal,
        "year": request.year,
        "Conference Location": request.location,
        "pages": request.pages or "N/A",
        "doi": request.doi,
        "Conference Location": request.location,
        "type": request.type or "",
        "volume": request.volume,
        "issue": request.issue,
        "url": request.url

    }

    citation = generate_citation(paper_details)
    return {"citation": citation}


