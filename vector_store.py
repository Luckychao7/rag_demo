"""Per-document FAISS persistence with dense, BM25 and neighbor helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter

import faiss
import numpy as np

import config
import embeddings as emb


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z]+[A-Za-z0-9._/-]*|\d+(?:[._/-]\d+)*", text)]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run[index:index + 2] for index in range(max(0, len(run) - 1)))
        if len(run) == 1:
            tokens.append(run)
    return tokens


class VectorStoreManager:
    def __init__(self):
        self.stores: dict[str, faiss.Index] = {}
        self.metadata: dict[str, dict] = {}
        os.makedirs(config.VECTOR_DIR, exist_ok=True)

    def _index_path(self, doc_id: str) -> str:
        digest = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
        return os.path.join(config.VECTOR_DIR, f"{digest}.faiss")

    def _meta_path(self, doc_id: str) -> str:
        return os.path.join(config.VECTOR_DIR, f"{doc_id}.meta.json")

    def _document_ids(self) -> list[str]:
        disk = [name[:-10] for name in os.listdir(config.VECTOR_DIR) if name.endswith(".meta.json")]
        return list(dict.fromkeys([*self.stores, *disk]))

    def load_store(self, doc_id: str):
        if os.path.exists(self._meta_path(doc_id)):
            with open(self._meta_path(doc_id), "r", encoding="utf-8") as handle:
                self.metadata[doc_id] = json.load(handle)
            self.metadata[doc_id].setdefault("chunks", [])
        if os.path.exists(self._index_path(doc_id)):
            self.stores[doc_id] = faiss.read_index(self._index_path(doc_id))

    def _save_store(self, doc_id: str):
        faiss.write_index(self.stores[doc_id], self._index_path(doc_id))
        with open(self._meta_path(doc_id), "w", encoding="utf-8") as handle:
            json.dump(self.metadata[doc_id], handle, ensure_ascii=False, indent=2)

    def add_document(self, doc_id: str, doc_name: str, chunks: list[dict]):
        chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
        if not chunks:
            return
        vectors = np.asarray(emb.embed_texts([chunk["text"] for chunk in chunks]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != config.EMBEDDING_DIM:
            raise ValueError(f"Embedding 维度异常：期望 {config.EMBEDDING_DIM}，实际 {vectors.shape}")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        normalized = []
        for chunk_index, chunk in enumerate(chunks):
            item = dict(chunk)
            item.update({"doc_name": doc_name, "chunk_index": chunk_index})
            normalized.append(item)
        self.stores[doc_id] = index
        self.metadata[doc_id] = {"schema_version": 2, "chunks": normalized}
        self._save_store(doc_id)

    def _targets(self, doc_ids: list[str] | None) -> list[str]:
        return doc_ids if doc_ids else self._document_ids()

    def _load_target(self, doc_id: str) -> bool:
        if doc_id not in self.stores or doc_id not in self.metadata:
            self.load_store(doc_id)
        return doc_id in self.stores and doc_id in self.metadata

    def dense_search(self, query: str, *, per_document_k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]:
        query_vector = np.asarray([emb.embed_query(query)], dtype=np.float32)
        results = []
        for doc_id in self._targets(doc_ids):
            if not self._load_target(doc_id) or not self.stores[doc_id].ntotal:
                continue
            count = min(per_document_k or config.DENSE_CANDIDATES_PER_DOCUMENT, self.stores[doc_id].ntotal)
            scores, indices = self.stores[doc_id].search(query_vector, count)
            for score, chunk_index in zip(scores[0], indices[0]):
                if chunk_index >= 0:
                    item = dict(self.metadata[doc_id]["chunks"][chunk_index])
                    item.update({"doc_id": doc_id, "dense_score": float(score), "score": float(score)})
                    results.append(item)
        return sorted(results, key=lambda item: item["dense_score"], reverse=True)

    def keyword_search(self, query: str, *, top_k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]:
        query_tokens = _tokenize(query)
        corpus = []
        for doc_id in self._targets(doc_ids):
            if self._load_target(doc_id):
                corpus.extend((doc_id, chunk, _tokenize(chunk["text"])) for chunk in self.metadata[doc_id]["chunks"])
        if not query_tokens or not corpus:
            return []
        frequencies = Counter(token for _, _, tokens in corpus for token in set(tokens))
        average_length = sum(len(tokens) for _, _, tokens in corpus) / len(corpus)
        results = []
        for doc_id, chunk, tokens in corpus:
            counts, score = Counter(tokens), 0.0
            for token in query_tokens:
                occurrences = counts[token]
                if occurrences:
                    inverse_frequency = math.log(1 + (len(corpus) - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
                    score += inverse_frequency * occurrences * 2.2 / (occurrences + 1.2 * (0.25 + 0.75 * len(tokens) / max(average_length, 1)))
            if score:
                item = dict(chunk)
                item.update({"doc_id": doc_id, "keyword_score": score, "score": score})
                results.append(item)
        return sorted(results, key=lambda item: item["keyword_score"], reverse=True)[:top_k or config.BM25_CANDIDATES]

    def search(self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None, threshold: float | None = None) -> list[dict]:
        """Legacy dense-only API. ``threshold`` is deliberately not used at recall time."""
        limit = top_k or config.TOP_K
        return self.dense_search(query, per_document_k=limit, doc_ids=doc_ids)[:limit]

    def get_neighbors(self, results: list[dict], *, max_hits: int | None = None) -> list[dict]:
        expanded = {(item["doc_id"], item["chunk_index"]): dict(item) for item in results}
        for item in results[:max_hits or config.NEIGHBOR_EXPANSION_HITS]:
            if item.get("score", 0.0) < config.NEIGHBOR_EXPANSION_THRESHOLD or not self._load_target(item["doc_id"]):
                continue
            chunks = self.metadata[item["doc_id"]]["chunks"]
            for neighbor_index in (item["chunk_index"] - 1, item["chunk_index"] + 1):
                key = (item["doc_id"], neighbor_index)
                if 0 <= neighbor_index < len(chunks) and key not in expanded:
                    neighbor = dict(chunks[neighbor_index])
                    neighbor.update({"doc_id": item["doc_id"], "chunk_index": neighbor_index,
                                     "score": item["score"] * 0.85, "neighbor_of": item["chunk_index"]})
                    expanded[key] = neighbor
        return sorted(expanded.values(), key=lambda item: (-item["score"], item["doc_id"], item["chunk_index"]))

    def delete_document(self, doc_id: str):
        self.stores.pop(doc_id, None)
        self.metadata.pop(doc_id, None)
        for path in (self._index_path(doc_id), self._meta_path(doc_id)):
            if os.path.exists(path):
                os.remove(path)

    def list_documents(self) -> list[dict]:
        documents = []
        for doc_id in self._document_ids():
            if self._load_target(doc_id) and self.metadata[doc_id]["chunks"]:
                chunks = self.metadata[doc_id]["chunks"]
                documents.append({"doc_id": doc_id, "doc_name": chunks[0]["doc_name"], "chunk_count": len(chunks)})
        return sorted(documents, key=lambda item: item["doc_name"])

    def get_total_chunks(self) -> int:
        return sum(item["chunk_count"] for item in self.list_documents())
