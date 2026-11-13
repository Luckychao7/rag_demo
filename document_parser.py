"""Document parsing with structural metadata, while retaining the text-only API."""
from __future__ import annotations

import csv
import os
import re


def parse_document(file_path: str) -> str:
    """Backward-compatible plain text extraction."""
    return "\n".join(block["text"] for block in parse_document_blocks(file_path))


def parse_document_blocks(file_path: str) -> list[dict]:
    """Return blocks with text/page/section/subsection/kind metadata."""
    extension = os.path.splitext(file_path)[1].lower()
    parsers = {".pdf": _parse_pdf, ".docx": _parse_docx, ".csv": _parse_csv,
               ".xlsx": _parse_xlsx, ".txt": _parse_text, ".md": _parse_text}
    if extension not in parsers:
        raise ValueError(f"不支持的文件格式: {extension}（支持: {list(parsers.keys())}）")
    return [item for item in parsers[extension](file_path) if item["text"].strip()]


def _block(text: str, **metadata) -> dict:
    return {"text": re.sub(r"[ \t]+", " ", text).strip(), **metadata}


def _parse_pdf(file_path: str) -> list[dict]:
    import pdfplumber
    blocks = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if text := page.extract_text():
                blocks.append(_block(text, page=page_number, kind="paragraph"))
            for table in page.extract_tables() or []:
                rows = [" | ".join(str(cell).strip() if cell else "" for cell in row) for row in table]
                if table_text := "\n".join(row for row in rows if row.strip(" |")):
                    blocks.append(_block(table_text, page=page_number, kind="table"))
    return blocks


def _parse_docx(file_path: str) -> list[dict]:
    from docx import Document
    document = Document(file_path)
    blocks, section, subsection = [], "", ""
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if "heading 1" in style or "标题 1" in style:
            section, subsection = text, ""
        elif "heading" in style or "标题" in style:
            subsection = text
        else:
            blocks.append(_block(text, section=section, subsection=subsection, kind="paragraph"))
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        if table_text := "\n".join(row for row in rows if row.strip(" |")):
            blocks.append(_block(table_text, section=section, subsection=subsection, kind="table"))
    return blocks


def _parse_csv(file_path: str) -> list[dict]:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding, errors="replace", newline="") as handle:
                rows = [" | ".join(cell.strip() for cell in row) for row in csv.reader(handle)]
            return [_block("\n".join(rows), section="CSV 数据", kind="table")]
        except (OSError, UnicodeDecodeError):
            continue
    return []


def _parse_xlsx(file_path: str) -> list[dict]:
    import openpyxl
    workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    blocks = []
    try:
        for sheet_name in workbook.sheetnames:
            rows = []
            for row in workbook[sheet_name].iter_rows(values_only=True):
                values = [str(value).strip() if value is not None else "" for value in row]
                if any(values):
                    rows.append(" | ".join(values))
            if rows:
                blocks.append(_block("\n".join(rows), section=f"工作表：{sheet_name}", kind="table"))
    finally:
        workbook.close()
    return blocks


def _parse_text(file_path: str) -> list[dict]:
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as handle:
                return [_block(handle.read(), kind="paragraph")]
        except (OSError, UnicodeDecodeError):
            continue
    raise ValueError(f"无法读取文件: {file_path}")
