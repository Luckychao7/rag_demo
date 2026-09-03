"""
文本分块器 - 将长文本按段落和大小切分为检索用的小块
"""


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list:
    """
    先按段落分割，再按 chunk_size 切分，相邻块之间保留 overlap 重叠。
    返回: [{"text": str, "start": int, "end": int}, ...]
    """
    if not text or not text.strip():
        return []

    paragraphs = text.split("\n")
    chunks = []
    current = ""
    start = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 如果单段就超长，先强制切
        if len(para) > chunk_size:
            if current:
                chunks.append({"text": current.strip(), "start": start, "end": start + len(current)})
                start += len(current)
                current = ""
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i:i + chunk_size]
                chunks.append({"text": piece, "start": start + i, "end": start + i + len(piece)})
            start += len(para)
            continue

        if len(current) + len(para) + 1 > chunk_size and current:
            chunks.append({"text": current.strip(), "start": start, "end": start + len(current)})
            # 保留重叠：取上一块末尾 overlap 个字符
            overlap_text = current[-overlap:] if len(current) > overlap else ""
            start += len(current)
            current = overlap_text + "\n" + para
        else:
            current = current + "\n" + para if current else para

    if current.strip():
        chunks.append({"text": current.strip(), "start": start, "end": start + len(current)})

    return chunks
