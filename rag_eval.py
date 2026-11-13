"""Retrieval evaluation for labelled enterprise-document questions.

Usage: python rag_eval.py evaluation_cases.json
Each case needs question, expected_doc, and expected_keywords. ``refined_query`` is optional.
"""
from __future__ import annotations

import json
import sys
import time

import config
import rag_engine


def _relevant(result: dict, case: dict) -> bool:
    document_ok = not case.get("expected_doc") or result["doc_name"] == case["expected_doc"]
    text = result["text"].lower()
    keywords = [item.lower() for item in case.get("expected_keywords", [])]
    keywords_ok = not keywords or all(item in text for item in keywords)
    return document_ok and keywords_ok


def _metrics(results_by_case: list[tuple[dict, list[dict]]]) -> dict:
    total = len(results_by_case)
    values = {"recall@5": 0, "recall@10": 0, "mrr": 0.0}
    for case, results in results_by_case:
        ranks = [index for index, item in enumerate(results, start=1) if _relevant(item, case)]
        if ranks:
            first = ranks[0]
            values["mrr"] += 1 / first
            values["recall@5"] += first <= 5
            values["recall@10"] += first <= 10
    return {name: round(value / total, 4) for name, value in values.items()} if total else values


def _run_dense(store, query, case):
    return store.dense_search(query, per_document_k=10)


def _run_hybrid(store, original, refined):
    result_lists = []
    for query in dict.fromkeys([original, refined]):
        result_lists += [store.dense_search(query), store.keyword_search(query)]
    return rag_engine._rrf_merge(result_lists)


def main(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    required = [case for case in cases if case.get("expected_doc") or case.get("expected_keywords")]
    if not required:
        raise ValueError("没有带 expected_doc 或 expected_keywords 的可评估样本")
    store, outputs = rag_engine.get_vsm(), {"Dense": [], "Dense + Rewrite": [], "Hybrid": [], "Hybrid + Reranker": []}
    for case in required:
        original, refined = case["question"], case.get("refined_query", case["question"])
        started = time.perf_counter()
        outputs["Dense"].append((case, _run_dense(store, original, case)))
        outputs["Dense + Rewrite"].append((case, _run_dense(store, refined, case)))
        hybrid = _run_hybrid(store, original, refined)
        outputs["Hybrid"].append((case, hybrid))
        outputs["Hybrid + Reranker"].append((case, rag_engine.rerank(original, hybrid, 10)))
        print(f"[Eval] {original[:40]!r} {(time.perf_counter() - started) * 1000:.0f}ms")
    print("\n方案\tRecall@5\tRecall@10\tMRR\tHit Rate")
    for name, output in outputs.items():
        metric = _metrics(output)
        print(f"{name}\t{metric['recall@5']:.2%}\t{metric['recall@10']:.2%}\t{metric['mrr']:.2%}\t{metric['recall@5']:.2%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation_cases.json")
