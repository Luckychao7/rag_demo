"""可选的本地 CrossEncoder 精排，并提供零额外依赖的融合降级方案。"""
from __future__ import annotations

import re
from typing import Iterable

import config

_model = None
_model_load_failed = False


def _tokens(text: str) -> set[str]:
    entities = re.findall(r"[A-Za-z]+[A-Za-z0-9._/-]*|\d{2,}[\d._/-]*", text)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return {item.lower() for item in entities + chinese if len(item) >= 2}


def entity_overlap(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    return sum(token in text_lower for token in query_tokens) / len(query_tokens)


def _get_model():
    global _model, _model_load_failed
    if not config.ENABLE_RERANKER or _model_load_failed:
        return None
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Rerank] 正在加载本地模型: {config.RERANKER_MODEL}")
            _model = CrossEncoder(config.RERANKER_MODEL)
        except Exception as exc:
            _model_load_failed = True
            print(f"[Rerank] 模型不可用，使用融合降级: {exc}")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """以 CrossEncoder 精排；不可用时使用 dense/BM25/实体命中融合分数。"""
    if not candidates:
        return []
    candidates = candidates[:config.RERANKER_CANDIDATES]
    model = _get_model()
    if model is not None:
        scores = model.predict(
            [(query, item["text"]) for item in candidates],
            batch_size=config.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            item["rerank_method"] = "cross_encoder"
    else:
        for item in candidates:
            fusion = item.get("fusion_score", 0.0)
            lexical = entity_overlap(query, item["text"])
            item["rerank_score"] = 0.85 * fusion + 0.15 * lexical
            item["score"] = item["rerank_score"]
            item["rerank_method"] = "fusion_fallback"
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]
