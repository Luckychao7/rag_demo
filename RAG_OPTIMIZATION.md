# 企业文档 RAG 优化说明

## 审计结论

原流程是“单个 rewrite query → 每个文档 Top-K FAISS → LLM”。其 P0 问题是：原始问题实体会丢失、每文档只保留最终 Top-K、无关键词召回/精排、纯文本 chunk 丢失来源结构、历史回答进入事实上下文。`BAAI/bge-small-zh-v1.5` 本身适合中文；其 512 维输出在 `embeddings.py` 明确启用归一化，所以 `IndexFlatIP` 才等价于余弦相似度。

## 新流程

`上传 → 结构化解析 → 标题/段落/句子切分 → 批量 normalized embedding → FAISS`

`原始 Query + 保实体 Rewrite → Dense(每文档候选) + BM25 → RRF 去重合并 → CrossEncoder/融合降级精排 → Top-K → 相邻块扩展 → 有长度上限的结构化 Context → Ollama → 程序来源引用`

## 配置与运行

1. 安装依赖：`python -m pip install -r requirements.txt`。
2. 已有文档可以继续查询；为了获得章节、页码和新 chunk 策略，请重新上传并索引。
3. 默认 `ENABLE_RERANKER = False`，不会在首个查询时下载大模型。先本地下载 `BAAI/bge-reranker-base` 后将其设为 `True`，系统会使用 CrossEncoder；加载失败时自动回退到 Dense/BM25/实体命中融合精排。
4. `EMBEDDING_QUERY_INSTRUCTION` 默认空，保证旧索引兼容。启用 BGE 查询指令后必须重建所有索引。

## 评估

复制 `evaluation_cases.sample.json` 为 `evaluation_cases.json`，以真实语料替换 20 条案例中的问题、正确文件名和关键词，运行：

`python rag_eval.py evaluation_cases.json`

脚本分别报告 Dense、Dense + Rewrite、Hybrid、Hybrid + Reranker 的 Recall@5、Recall@10、MRR 和 Hit Rate。答案准确率应人工基于标准答案单独记录，不能用 LLM 自评代替。

## 性能观测

控制台输出 `[Rewrite]`、`[Dense Retrieval]`、`[Keyword Retrieval]`、`[Rerank]`、`[Context]`、`[LLM]` 和总检索耗时；这可直接用于定位 embedding、BM25、精排或 Ollama 的瓶颈。通过 `DENSE_CANDIDATES_PER_DOCUMENT`、`BM25_CANDIDATES`、`RERANKER_CANDIDATES`、`CONTEXT_MAX_CHARS` 在真实评测集上调优，而不是固定阈值猜测。
