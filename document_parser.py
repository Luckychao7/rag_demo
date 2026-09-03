"""
文档解析模块 - 从不同格式的文件中提取纯文本
支持: pdf / docx / csv / xlsx / txt / md
"""
import os
import csv
import io


def parse_document(file_path: str) -> str:
    """根据文件扩展名选择解析器，返回提取的文本内容"""
    ext = os.path.splitext(file_path)[1].lower()
    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".csv": _parse_csv,
        ".xlsx": _parse_xlsx,
        ".txt": _parse_text,
        ".md": _parse_text,
    }
    parser = parsers.get(ext)
    if parser is None:
        raise ValueError(f"不支持的文件格式: {ext}（支持: {list(parsers.keys())}）")
    return parser(file_path)


def _parse_pdf(file_path: str) -> str:
    import pdfplumber
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
            # 提取表格内容
            for table in page.extract_tables():
                for row in table:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    texts.append(" | ".join(cleaned))
    return "\n".join(texts)


def _parse_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text)
    # 提取表格
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            texts.append(" | ".join(cells))
    return "\n".join(texts)


def _parse_csv(file_path: str) -> str:
    texts = []
    # 尝试 utf-8，失败则用 gbk（中文 Windows 常见编码）
    for encoding in ["utf-8", "gbk", "latin-1"]:
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    texts.append(" | ".join(row))
            break
        except Exception:
            continue
    return "\n".join(texts)


def _parse_xlsx(file_path: str) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    texts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        texts.append(f"[工作表: {sheet_name}]")
        for row in ws.iter_rows(values_only=True):
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cells):
                texts.append(" | ".join(cells))
    wb.close()
    return "\n".join(texts)


def _parse_text(file_path: str) -> str:
    for encoding in ["utf-8", "gbk", "latin-1"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {file_path}")
