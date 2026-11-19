"""Shared text preprocessing for ScholarLens.

Scientific text is noisy: LaTeX markup, citation markers, URLs, figure/table
references and broken whitespace all hurt tokenization and model quality.
Every module (summarization, QA, NER) uses these helpers so cleaning is
consistent across the pipeline and can be described once in the report.
"""

import re
from typing import Dict, List

# Pre-compiled patterns (compiled once, reused for every document).
_LATEX_INLINE_MATH = re.compile(r"\$[^$]{0,200}\$")
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")
_CITATION_NUMERIC = re.compile(r"\[(?:\d+\s*[,\u2013-]?\s*)+\]")
_CITATION_AUTHOR = re.compile(r"\((?:[A-Z][A-Za-z\-]+(?: et al\.)?,? \d{4}(?:;\s*)?)+\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_FIG_TABLE_REF = re.compile(r"\b(?:Fig\.|Figure|Table|Eq\.|Equation)\s*\(?\d+[a-z]?\)?", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# Placeholder tokens used by the arXiv dataset for math and citations.
_DATASET_TAGS = re.compile(r"@xmath\d*|@xcite|@\w*cite\w*")

# Sentence boundary: end punctuation followed by whitespace and an uppercase
# letter or digit. Abbreviations such as "et al." and "e.g." are protected below.
_ABBREVIATIONS = ["et al.", "e.g.", "i.e.", "Fig.", "Eq.", "vs.", "cf.", "approx."]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean_text(text: str) -> str:
    """Remove LaTeX, citation markers, URLs and dataset placeholder tags."""
    if not text:
        return ""
    text = _DATASET_TAGS.sub(" ", text)
    text = _LATEX_INLINE_MATH.sub(" ", text)
    text = _LATEX_COMMAND.sub(" ", text)
    text = _CITATION_NUMERIC.sub(" ", text)
    text = _CITATION_AUTHOR.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _FIG_TABLE_REF.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    # Tidy spaces left before punctuation by the removals above.
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, protecting common scientific abbreviations."""
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"<DOT{i}>"))
    parts = _SENT_SPLIT.split(protected)
    sentences = []
    for part in parts:
        for i, abbr in enumerate(_ABBREVIATIONS):
            part = part.replace(abbr.replace(".", f"<DOT{i}>"), abbr)
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences


def split_into_passages(text: str, max_words: int = 120, overlap_sentences: int = 1) -> List[str]:
    """Group sentences into overlapping passages for the retrieval module.

    Passages of roughly `max_words` words keep enough context for QA while
    staying small enough for the embedding model. Overlap avoids cutting an
    answer in half at a passage boundary.
    """
    sentences = split_sentences(clean_text(text))
    passages: List[str] = []
    current: List[str] = []
    count = 0
    for sent in sentences:
        n_words = len(sent.split())
        if current and count + n_words > max_words:
            passages.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            count = sum(len(s.split()) for s in current)
        current.append(sent)
        count += n_words
    if current:
        passages.append(" ".join(current))
    return passages


def is_valid_pair(source: str, target: str, min_source_words: int = 100,
                  min_target_words: int = 20, max_ratio: float = 0.6) -> bool:
    """Filter degenerate summarization pairs (too short, or summary ~ source)."""
    s_len, t_len = len(source.split()), len(target.split())
    if s_len < min_source_words or t_len < min_target_words:
        return False
    return (t_len / s_len) <= max_ratio


def preprocess_summarization_example(example: Dict[str, str]) -> Dict[str, str]:
    """Clean one arXiv example (fields: article, abstract)."""
    return {
        "article": clean_text(example["article"]),
        "abstract": clean_text(example["abstract"]),
    }
