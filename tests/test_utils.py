import pytest
from app.citation_api import parse_authors, format_author_list, get_valid_field

def test_parse_authors_list_string():
    assert parse_authors("['Alice','Bob']") == ['Alice','Bob']

def test_parse_authors_comma_string():
    assert parse_authors("Alice, Bob, Charlie") == ['Alice','Bob','Charlie']

def test_format_author_list_empty():
    assert format_author_list([]) == "Unknown Author"

def test_format_author_list_two():
    assert format_author_list(["Alice Smith","Bob Jones"]) == "A. Smith and B. Jones"

def test_get_valid_field_none_and_nan():
    assert get_valid_field(None) is None
    assert get_valid_field("NaN") is None
    assert get_valid_field("  ") is None
    assert get_valid_field("Hello") == "Hello"
