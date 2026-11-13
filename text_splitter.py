"""Structure-aware chunking that preserves document metadata."""
from __future__ import annotations

import re

_HEADING = re.compile(r"^(?:#{1,6}\s+|第[一二三四五六七八九十百0-9]+[章节部分]|\d+(?:\.\d+){0,3}[、.．\s]).{2,}$")
_SENTENCES = re.compile(r"(?<=[。！？；;])")


def split_text(text: str, chunk_size: int = 500, overlap: int = 80, *, doc_name: str = "", blocks: list[dict] | None = None) -> list[dict]:
    """Keep compatibility with plain text and accept parser blocks for richer chunks."""
    blocks = blocks if blocks is not None else [{"text": text, "kind": "paragraph"}]
    chunks, section, subsection, offset = [], "", "", 0
    for block in blocks:
        raw = (block.get("text") or "").strip()
        if not raw:
            continue
        block_section = block.get("section") or section
        block_subsection = block.get("subsection") or subsection
        paragraphs = [value.strip() for value in raw.splitlines() if value.strip()]
        groups, current_group = [], []
        for paragraph in paragraphs:
            if block.get("kind") != "table" and _HEADING.match(paragraph) and not block.get("section"):
                if current_group:
                    groups.append((block_section or section, block_subsection or subsection, current_group))
                    current_group = []
                if paragraph.startswith("#") or paragraph.startswith("第"):
                    section, subsection = paragraph, ""
                else:
                    subsection = paragraph
            else:
                current_group.append(paragraph)
        if current_group:
            groups.append((block_section or section, block_subsection or subsection, current_group))
        for group_section, group_subsection, group_paragraphs in groups:
            for piece in _split_units(group_paragraphs, chunk_size, overlap):
                chunks.append({"text": piece, "start": offset, "end": offset + len(piece), "doc_name": doc_name,
                               "section": group_section, "subsection": group_subsection,
                               "page": block.get("page"), "kind": block.get("kind", "paragraph")})
                offset += len(piece) + 1
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def _split_units(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    pieces, current = [], ""
    for paragraph in paragraphs:
        for sentence in (item for item in _SENTENCES.split(paragraph) if item):
            if len(sentence) > size:
                if current:
                    pieces.append(current.strip())
                    current = ""
                step = max(1, size - overlap)
                pieces.extend(sentence[start:start + size].strip() for start in range(0, len(sentence), step))
            elif current and len(current) + len(sentence) + 1 > size:
                pieces.append(current.strip())
                current = f"{current[-overlap:]}\n{sentence}".strip() if overlap else sentence
            else:
                current = f"{current}\n{sentence}".strip()
    if current:
        pieces.append(current.strip())
    return [piece for piece in pieces if piece]
