"""RAG orchestration: query assessment, hybrid retrieval, reranking and grounded answers."""
from __future__ import annotations

import json
import os
import re
import time

import config
import llm_service
from reranker import rerank
from vector_store import VectorStoreManager

_vsm = None


def get_vsm() -> VectorStoreManager:
    global _vsm
    if _vsm is None:
        _vsm = VectorStoreManager()
    return _vsm


SYSTEM_PROMPT = """你是企业内部文档问答助手。回答只能依据本次提供的参考文档，历史对话不是事实来源。
若参考文档不足以支持答案，明确说“根据现有文档，我无法找到相关答案”。不要编造名称、金额、日期或来源。
回答简洁专业；可在答案中引用给定的真实文档名，不能创造文档名。"""

ANALYZE_PROMPT = """你只负责判断是否需要澄清，并在必要时补足指代上下文来生成检索改写。
绝对规则：不得删除或泛化用户问题中的编号、日期、金额、专有名词、英文缩写、型号和其他关键实体；原问题已明确时原样返回。
若无法检索（例如“那个怎么样”且历史也没有对象），返回 [NEED_CLARIFY] 加不超过30字的追问；否则只返回改写查询，不解释。

仅供理解指代的历史用户问题（不是事实来源）：
{history}

用户最新问题：{question}"""

RAG_PROMPT_TEMPLATE = """根据以下程序提供的参考文档回答问题。

{context}

用户问题：{question}

规则：只能以参考文档为事实依据；信息不足时明确说明找不到答案；不要编造引用或文档名。"""


def _history_path(conv_id: str) -> str:
    return os.path.join(config.HISTORY_DIR, f"{conv_id}.json")


def save_history(conv_id: str, messages: list):
    with open(_history_path(conv_id), "w", encoding="utf-8") as handle:
        json.dump(messages, handle, ensure_ascii=False, indent=2)


def load_history(conv_id: str) -> list:
    path = _history_path(conv_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_histories() -> list:
    histories = []
    for filename in os.listdir(config.HISTORY_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(config.HISTORY_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                messages = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        title = next((item.get("content", "")[:30] for item in messages if item.get("role") == "user"), "新对话")
        histories.append({"id": filename[:-5], "title": title, "msg_count": len(messages), "_mtime": os.path.getmtime(path)})
    return [{key: value for key, value in item.items() if key != "_mtime"} for item in sorted(histories, key=lambda item: item["_mtime"], reverse=True)]


def delete_history(conv_id: str):
    path = _history_path(conv_id)
    if os.path.exists(path):
        os.remove(path)


def _entities(text: str) -> list[str]:
    pattern = r"[A-Za-z]+[A-Za-z0-9._/-]*|\d{2,}(?:[._/-]\d+)*|\d+(?:\.\d+)?(?:元|万元|亿元|%|日|月|年)"
    return list(dict.fromkeys(re.findall(pattern, text)))


def _preserve_entities(original: str, rewritten: str) -> str:
    missing = [entity for entity in _entities(original) if entity.lower() not in rewritten.lower()]
    return f"{rewritten} {' '.join(missing)}".strip() if missing else rewritten


def assess_question(question: str, chat_history: list | None = None) -> dict:
    """Use history only for coreference; never replace the original query."""
    user_history = [item.get("content", "")[:300] for item in (chat_history or []) if item.get("role") == "user"][-3:]
    history = "\n".join(f"用户：{item}" for item in user_history)
    try:
        response = llm_service.chat([
            {"role": "system", "content": ANALYZE_PROMPT.format(history=history or "（无）", question=question)},
            {"role": "user", "content": question},
        ], temperature=0.1).strip()
        if "[NEED_CLARIFY]" in response:
            return {"need_clarify": True, "clarify_question": response.replace("[NEED_CLARIFY]", "").strip() or "请补充要查询的具体对象。", "refined_query": question}
        refined = _preserve_entities(question, response if len(response) >= 2 else question)
        print(f"[Rewrite] original={question[:80]!r} refined={refined[:100]!r}")
        return {"need_clarify": False, "clarify_question": "", "refined_query": refined}
    except Exception as exc:
        print(f"[Rewrite] 降级为原始查询: {exc}")
        return {"need_clarify": False, "clarify_question": "", "refined_query": question}


def _rrf_merge(result_lists: list[list[dict]]) -> list[dict]:
    merged = {}
    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            key = (item["doc_id"], item["chunk_index"])
            target = merged.setdefault(key, dict(item, fusion_score=0.0, retrieval_channels=[]))
            target["fusion_score"] += 1.0 / (config.RRF_K + rank)
            channel = "keyword" if "keyword_score" in item else "dense"
            if channel not in target["retrieval_channels"]:
                target["retrieval_channels"].append(channel)
            for name in ("dense_score", "keyword_score"):
                if name in item:
                    target[name] = max(target.get(name, float("-inf")), item[name])
    return sorted(merged.values(), key=lambda item: item["fusion_score"], reverse=True)[:config.FUSION_CANDIDATES]


def search_documents(original_query: str, refined_query: str | None = None, doc_ids: list | None = None) -> list[dict]:
    """Original and refined queries use dense + BM25 independently, then share reranking."""
    started = time.perf_counter()
    queries = list(dict.fromkeys(query for query in (original_query, refined_query or original_query) if query.strip()))
    store = get_vsm()
    result_lists = []
    for query in queries:
        dense = store.dense_search(query, doc_ids=doc_ids)
        keyword = store.keyword_search(query, doc_ids=doc_ids)
        result_lists.extend((dense, keyword))
        print(f"[Dense Retrieval] query={query[:60]!r} candidates={len(dense)}")
        print(f"[Keyword Retrieval] query={query[:60]!r} candidates={len(keyword)}")
    candidates = _rrf_merge(result_lists)
    reranked = rerank(original_query, candidates, config.RERANK_TOP_K)
    expanded = store.get_neighbors(reranked)
    print(f"[Rerank] candidates={len(candidates)} final={len(reranked)} method={reranked[0].get('rerank_method', '-') if reranked else '-'}")
    print(f"[Context] chunks_after_neighbor={len(expanded)}")
    print(f"[Query] total_retrieval_ms={(time.perf_counter() - started) * 1000:.0f}")
    return expanded


def _format_location(item: dict) -> str:
    fields = [f"文档：{item['doc_name']}"]
    if item.get("section"):
        fields.append(f"章节：{item['section']}")
    if item.get("subsection"):
        fields.append(f"小节：{item['subsection']}")
    if item.get("page"):
        fields.append(f"页码：{item['page']}")
    fields.append(f"文本块：{item['chunk_index']}")
    return "\n".join(fields)


def build_context(search_results: list[dict]) -> tuple[str, list[dict]]:
    """Deduplicate, cap context, and expose only real program-originated citations."""
    parts, sources, used_texts, total_chars = [], [], set(), 0
    for item in search_results:
        text = item["text"].strip()
        if not text or text in used_texts or len(sources) >= config.CONTEXT_MAX_SOURCES:
            continue
        addition = len(text)
        if total_chars + addition > config.CONTEXT_MAX_CHARS:
            continue
        used_texts.add(text)
        total_chars += addition
        parts.append(f"[来源{len(sources) + 1}]\n{_format_location(item)}\n相关内容：\n{text}")
        sources.append({key: item.get(key) for key in ("doc_name", "doc_id", "chunk_index", "page", "section", "subsection", "score")})
        sources[-1]["text"] = text[:500]
    return "\n\n---\n\n".join(parts), sources


def generate_answer(question: str, search_results: list, chat_history: list | None = None) -> dict:
    context, sources = build_context(search_results)
    if not context:
        return {"answer": "根据现有文档，我无法找到相关答案。", "sources": []}
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    started = time.perf_counter()
    answer = llm_service.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], temperature=0.2)
    print(f"[LLM] answer_ms={(time.perf_counter() - started) * 1000:.0f}; [Total] completed")
    return {"answer": answer, "sources": sources}
