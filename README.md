# 企业内部文档 RAG 问答系统

这是一个本地运行的企业文档问答系统，支持 PDF、DOCX、CSV、XLSX、TXT 和 Markdown 文档。系统使用 Ollama 生成答案，使用 BGE Embedding、FAISS、BM25 和可选本地 Reranker 完成混合检索。

## 一、运行环境

推荐在 Windows 10/11 上使用 Python 3.10～3.12。首次安装需要联网下载 Python 依赖与模型；完成下载后，文档检索与问答可在本地运行。

需要的软件：

- Python（建议使用虚拟环境或 Conda 环境）
- Ollama
- 可选：NVIDIA GPU。没有 GPU 时系统仍可使用 CPU 运行。

在 PowerShell 中进入项目目录：

```powershell
cd C:\Users\86130\Desktop\rag_demo-main\rag_demo-main
```

确认正在使用预期的 Python：

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

建议创建并启用独立虚拟环境，避免和其他项目的 `numpy`、`torch` 依赖冲突：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 阻止激活脚本，可只对当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

安装完成后检查核心模块：

```powershell
python -c "import faiss, streamlit, sentence_transformers; print('依赖检查通过')"
```

## 二、配置 Ollama

1. 从 [Ollama 官网](https://ollama.com/) 安装 Ollama。
2. 打开一个新的 PowerShell，下载当前默认的生成模型：

```powershell
ollama pull qwen2:7b-instruct-q4_K_M
```

3. 确认模型已安装：

```powershell
ollama list
```

4. 若 Ollama 服务未自动启动，执行：

```powershell
ollama serve
```

默认配置位于 `config.py`：

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2:7b-instruct-q4_K_M"
```

如果本机已下载其他 Ollama 模型，请将 `OLLAMA_MODEL` 改为 `ollama list` 中显示的模型名称。

## 三、Embedding 与 Reranker 配置

默认 Embedding 模型是 `BAAI/bge-small-zh-v1.5`，适合中文检索。首次上传文档时，Sentence Transformers 会下载模型；后续会复用本地缓存。

可在 `config.py` 调整以下参数：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 文档和查询使用同一个向量空间 |
| `EMBEDDING_BATCH_SIZE` | `32` | 批量向量化大小；显存不足时调小 |
| `DENSE_CANDIDATES_PER_DOCUMENT` | `20` | 每份文档的向量召回候选数 |
| `BM25_CANDIDATES` | `30` | 关键词/BM25 候选数 |
| `RERANK_TOP_K` | `6` | 精排后用于上下文的主命中数 |
| `CONTEXT_MAX_CHARS` | `9000` | 传给 LLM 的最大检索上下文长度 |

默认不开启 CrossEncoder，以免首次查询自动下载较大的模型。系统会使用 Dense、BM25 和实体命中融合分数完成降级精排。

如需使用本地 CrossEncoder 精排，先将 `BAAI/bge-reranker-base` 下载到本机缓存，然后修改：

```python
ENABLE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-base"
```

模型无法加载时，系统会输出日志并自动回退，不会中断问答。

> 修改 `EMBEDDING_MODEL` 或启用 `EMBEDDING_QUERY_INSTRUCTION` 后，必须删除旧索引并重新上传所有文档，不能混用不同 embedding 空间生成的向量。

## 四、启动项目

确保 Ollama 已运行后，在项目目录执行：

```powershell
python -m streamlit run app.py
```

终端会显示本地访问地址，通常为：

```text
http://localhost:8501
```

用浏览器打开该地址。页面侧边栏出现“🟢 Ollama 已连接”说明生成服务可用。

若启动失败，可按以下顺序排查：

1. 执行 `python -m pip install -r requirements.txt`，确认依赖安装在当前 `python` 对应的环境中。
2. 执行 `ollama list`，确认 `config.py` 中的模型已下载。
3. 执行 `ollama serve`，确认 11434 端口服务已启动。
4. 若端口 8501 被占用，使用 `python -m streamlit run app.py --server.port 8502`。

## 五、文档上传与建库

1. 在页面左侧“文档管理”选择一个或多个文件。
2. 点击“✅ 处理上传文档”。
3. 系统会依次完成：

```text
文件保存
→ 文档解析（文字、页码、表格、工作表、Word 标题）
→ 标题/段落/句子分块
→ 批量 Embedding
→ FAISS 建库
→ 保存 chunk 的文档名、章节、小节、页码等 metadata
```

4. 成功后，“已索引文档”会显示文件名及文本块数量。

同名文件再次上传时，系统会先删除旧索引，再建立新索引，避免重复检索结果。若更改了切分或 embedding 配置，也应重新上传相关文件。

索引文件保存在：

```text
data\vector_stores\
```

原始上传文件保存在：

```text
data\uploads\
```

## 六、项目应用：文档问答

在“对话问答”页面输入问题。例如：

```text
A-2025-0318 项目的预算是多少？
员工报销住宿费的标准是什么？
合同第 3.2 条规定了什么违约责任？
```

系统执行以下流程：

```text
原始问题
 + 保留编号/金额/日期/专有名词的改写问题
→ Dense 向量召回 + BM25 关键词召回
→ RRF 候选合并与去重
→ CrossEncoder 或融合精排
→ 高置信命中的相邻 chunk 扩展
→ 去重、限长、结构化 Context
→ Ollama 基于文档生成答案
→ 程序展示真实来源、章节和页码
```

### 搜索范围

在“已索引文档”中选择一个或多个文档，可限制检索范围；不选择时搜索全部已索引文档。可点击文档右侧的“🗑️”删除对应索引。

### 多轮对话

历史问题仅用于理解“它”“该项目”等指代和省略内容，历史回答不会作为事实来源写入最终上下文。最终答案仍只能根据本轮实际召回的文档内容生成。

### 来源核验

每个回答下方的“📎 查看引用来源”由程序生成，展示文档名、章节/小节、页码、文本块编号和实际片段，不依赖模型自行编造来源。

## 七、日志与性能排查

项目终端会输出以下阶段日志：

```text
[Rewrite]
[Dense Retrieval]
[Keyword Retrieval]
[Rerank]
[Context]
[LLM]
[Query]
```

排查建议：

- 首次上传慢：通常是下载/加载 Embedding 模型；后续会复用内存模型。
- 查询慢：先查看 `[Dense Retrieval]`、`[Rerank]` 和 `[LLM]` 的耗时；大多数情况下 Ollama 生成耗时最高。
- 显存不足：降低 `EMBEDDING_BATCH_SIZE`、`RERANKER_BATCH_SIZE`，或保持 `ENABLE_RERANKER = False`。
- 回答上下文过长：降低 `CONTEXT_MAX_CHARS` 或 `CONTEXT_MAX_SOURCES`。
- 编号/日期没有召回：检查原始文档是否包含该文本，并确认没有在问题中输入错误字符；BM25 会优先处理这类精确实体。

## 八、检索效果评估

不要只根据最终回答判断 RAG 效果。先评估检索是否找到正确文档/片段，再评估精排、上下文和最终答案。

1. 复制样例文件：

```powershell
Copy-Item evaluation_cases.sample.json evaluation_cases.json
```

2. 将 20 条样例替换为真实企业语料的测试问题。每条至少填写：

```json
{
  "question": "A-2025-0318项目预算是多少？",
  "expected_doc": "项目管理办法.pdf",
  "expected_keywords": ["A-2025-0318", "预算", "100万元"]
}
```

3. 运行评估：

```powershell
python rag_eval.py evaluation_cases.json
```

该脚本会对比 `Dense`、`Dense + Rewrite`、`Hybrid`、`Hybrid + Reranker`，输出 Recall@5、Recall@10、MRR 和 Hit Rate。应基于相同的标注集调整候选数、精排模型和上下文长度，而不是凭经验修改相似度阈值。

## 九、停止项目

在运行 Streamlit 的终端按 `Ctrl + C` 停止 Web 服务。若当前窗口专门运行 `ollama serve`，也可在该窗口按 `Ctrl + C` 停止 Ollama 服务。
