import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"


@dataclass(frozen=True)
class KnowledgeEntry:
    text: str
    source: str
    answer: str | None = None
    title: str | None = None
    source_urls: tuple[str, ...] = ()
    citation_label: str | None = None
    as_of: str | None = None


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be non-negative and smaller than max_words")
    words = list(re.finditer(r"\S+", text))
    chunks = []
    for start in range(0, len(words), max_words - overlap):
        end = min(start + max_words, len(words))
        chunks.append(text[words[start].start():words[end - 1].end()])
        if end == len(words):
            break
    return chunks


def document_entries(document_text: str, source: str) -> list[KnowledgeEntry]:
    headings = list(re.finditer(r"(?m)^\[(DOC-\d+)\]\s*(.*?)\s*$", document_text))
    if not headings:
        return [KnowledgeEntry(text=chunk, source=source) for chunk in chunk_text(document_text)]

    entries = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(document_text)
        section = document_text[heading.end() : end]
        section_end = re.search(r"(?m)^(?:SOURCE INDEX|END OF (?:EXPANDED MATERIAL|FILE))\s*$", section)
        if section_end:
            section = section[: section_end.start()]

        title = heading.group(2).strip()
        urls = tuple(
            dict.fromkeys(
                url.rstrip(".,;:")
                for url in re.findall(r"https?://[^\s<>\"']+", section)
            )
        )
        content = f"{heading.group(1)} {title}\n{section.strip()}".strip()
        for chunk in chunk_text(content, max_words=300, overlap=60):
            entries.append(
                KnowledgeEntry(text=chunk, source=source, title=title, source_urls=urls)
            )

    return entries


def _question_answer_entry(item: object, source: str) -> KnowledgeEntry | None:
    if not isinstance(item, dict):
        raise ValueError("Each Q&A record must be an object")
    try:
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("Each Q&A record needs question and answer strings") from error
    if not question or not answer:
        return None

    def optional_text(key: str) -> str:
        value = item.get(key, "")
        return str(value).strip() if value is not None else ""

    record_id = optional_text("id")
    category = optional_text("category")
    as_of = optional_text("as_of")
    keywords = optional_text("keywords")
    source_value = optional_text("source")
    urls = tuple(
        dict.fromkeys(
            url.rstrip(".,;:")
            for url in re.findall(r"https?://[^\s<>\"']+", source_value)
        )
    )
    context = [f"Question: {question}", f"Answer: {answer}"]
    if category:
        context.insert(0, f"Category: {category}")
    if record_id:
        context.insert(0, record_id)
    if as_of:
        context.append(f"As of: {as_of}")
    if keywords:
        context.append(f"Keywords: {keywords}")

    return KnowledgeEntry(
        text="\n".join(context),
        source=source,
        answer=answer,
        source_urls=urls,
        citation_label=f"{category} Q&A" if category else None,
        as_of=as_of or None,
    )


def _text_question_answer_entries(document_text: str, source: str) -> list[KnowledgeEntry] | None:
    if not re.search(r"(?m)^ID:\s*QA-\d+\s*$", document_text):
        return None

    entries = []
    field_pattern = re.compile(
        r"(?m)^(ID|Category|Question|Answer|As-of|Source|Keywords):[ \t]*(.*)$"
    )
    for block in re.split(r"(?m)^---\s*$", document_text):
        fields = list(field_pattern.finditer(block))
        if not any(field.group(1) == "ID" for field in fields):
            continue
        values = {}
        for index, field in enumerate(fields):
            end = fields[index + 1].start() if index + 1 < len(fields) else len(block)
            value = field.group(2) + block[field.end() : end]
            values[field.group(1)] = value.strip()

        entry = _question_answer_entry(
            {
                "id": values.get("ID", ""),
                "category": values.get("Category", ""),
                "question": values.get("Question", ""),
                "answer": values.get("Answer", ""),
                "as_of": values.get("As-of", ""),
                "source": values.get("Source", ""),
                "keywords": values.get("Keywords", ""),
            },
            source,
        )
        if entry:
            entries.append(entry)

    return entries


def _normalize_search_text(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def _build_fts_query(question: str) -> str:
    tokens = [token for token in _normalize_search_text(question).split() if len(token) > 1]
    if not tokens:
        return ""
    phrase = " ".join(tokens)
    quoted_terms = [f'"{token}"' for token in tokens]
    return f'"{phrase}" OR {" OR ".join(quoted_terms)}'


