"""
向量库管理 - 基于 FAISS，每个文档一个独立的索引文件
支持：添加文档、搜索、删除文档、持久化到磁盘
"""
import os
import json
import hashlib
import numpy as np
import faiss

import config
import embeddings as emb


class VectorStoreManager:
    """管理多个文档的 FAISS 向量索引，每个文档独立存储"""

    def __init__(self):
        self.stores = {}  # doc_id -> faiss.Index
        self.metadata = {}  # doc_id -> {chunks, doc_names, texts}
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(config.VECTOR_DIR, exist_ok=True)

    def _index_path(self, doc_id: str) -> str:
        safe_name = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
        return os.path.join(config.VECTOR_DIR, f"{safe_name}.faiss")

    def _meta_path(self, doc_id: str) -> str:
        return os.path.join(config.VECTOR_DIR, f"{doc_id}.meta.json")

    # ---------- 加载 / 持久化 ----------

    def load_store(self, doc_id: str):
        """从磁盘加载某个文档的向量索引和元数据"""
        idx_path = self._index_path(doc_id)
        meta_path = self._meta_path(doc_id)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata[doc_id] = json.load(f)
        if os.path.exists(idx_path):
            self.stores[doc_id] = faiss.read_index(idx_path)

    def _save_store(self, doc_id: str):
        if doc_id in self.stores:
            faiss.write_index(self.stores[doc_id], self._index_path(doc_id))
            with open(self._meta_path(doc_id), "w", encoding="utf-8") as f:
                json.dump(self.metadata[doc_id], f, ensure_ascii=False, indent=2)

    # ---------- 添加文档 ----------

    def add_document(self, doc_id: str, doc_name: str, chunks: list):
        """
        将文档分块结果写入向量库
        chunks: [{"text": str, "start": int, "end": int}, ...]
        """
        texts = [c["text"] for c in chunks]
        if not texts:
            return

        vectors = emb.embed_texts(texts)
        vectors_np = np.array(vectors, dtype=np.float32)

        # 创建或追加到已有索引
        if doc_id not in self.stores or self.stores[doc_id] is None:
            dim = vectors_np.shape[1]
            index = faiss.IndexFlatIP(dim)  # 内积（向量已归一化 = 余弦相似度）
            self.stores[doc_id] = index
            self.metadata[doc_id] = {"chunks": [], "doc_names": [], "texts": []}

        self.stores[doc_id].add(vectors_np)

        for i, chunk in enumerate(chunks):
            self.metadata[doc_id]["chunks"].append({
                "text": chunk["text"],
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
                "doc_name": doc_name,
                "chunk_index": len(self.metadata[doc_id]["chunks"]),
            })
            self.metadata[doc_id]["doc_names"].append(doc_name)
            self.metadata[doc_id]["texts"].append(chunk["text"])

        self._save_store(doc_id)

    # ---------- 搜索 ----------

    def search(self, query: str, top_k: int = None, doc_ids: list = None,
            threshold: float = None) -> list:
        """
        在指定文档（或全部文档）中检索最相似的片段
        返回: [{"text": str, "score": float, "doc_name": str, "doc_id": str, "chunk_index": int}, ...]
        """
        top_k = top_k or config.TOP_K
        threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
        query_vec = np.array([emb.embed_query(query)], dtype=np.float32)

        results = []
        if doc_ids:
            target_ids = doc_ids
        else:
            # 不选=全部：扫描磁盘获取所有文档 ID
            target_ids = []
            for fname in os.listdir(config.VECTOR_DIR):
                if fname.endswith(".meta.json"):
                    target_ids.append(fname[:-len(".meta.json")])

        for did in target_ids:
            if did not in self.stores:
                self.load_store(did)
            if did not in self.stores:
                continue

            index = self.stores[did]
            if index.ntotal == 0:
                continue

            k = min(top_k, index.ntotal)
            scores, indices = index.search(query_vec, k)

            for score, idx in zip(scores[0], indices[0]):
                print(f"  [检索] doc={did[:20]}... idx={idx} score={score:.4f} 阈值={threshold}")
                if idx < 0 or score < threshold:
                    continue
                meta = self.metadata[did]["chunks"][idx]
                results.append({
                    "text": meta["text"],
                    "score": float(score),
                    "doc_name": meta["doc_name"],
                    "doc_id": did,
                    "chunk_index": meta["chunk_index"],
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        print(f"[检索] 查询='{query[:50]}' 搜索文档数={len(target_ids)} 命中={len(results)}")
        return results[:top_k]

    # ---------- 删除文档 ----------

    def delete_document(self, doc_id: str):
        """删除某个文档的向量索引和元数据"""
        if doc_id in self.stores:
            del self.stores[doc_id]
        if doc_id in self.metadata:
            del self.metadata[doc_id]
        for p in [self._index_path(doc_id), self._meta_path(doc_id)]:
            if os.path.exists(p):
                os.remove(p)

    # ---------- 查询状态 ----------

    def list_documents(self) -> list:
        """返回所有已索引的文档信息"""
        docs = []
        seen_ids = set(self.stores.keys())
        for fname in os.listdir(config.VECTOR_DIR):
            if fname.endswith(".meta.json"):
                doc_id = fname[:-len(".meta.json")]
                seen_ids.add(doc_id)
        for doc_id in seen_ids:
            if doc_id not in self.metadata:
                self.load_store(doc_id)
            if doc_id in self.metadata and self.metadata[doc_id]["chunks"]:
                docs.append({
                    "doc_id": doc_id,
                    "doc_name": self.metadata[doc_id]["chunks"][0]["doc_name"],
                    "chunk_count": len(self.metadata[doc_id]["chunks"]),
                })
        return docs

    def get_total_chunks(self) -> int:
        return sum(len(m["chunks"]) for m in self.metadata.values())
