"""
配置文件 - 所有可调参数集中管理
"""
import os

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_stores")
HISTORY_DIR = os.path.join(DATA_DIR, "history")

# 确保目录存在
for d in [UPLOAD_DIR, VECTOR_DIR, HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ Ollama 配置 ============
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2:7b-instruct-q4_K_M"

# ============ Embedding 配置 ============
# 首次运行会自动从 HuggingFace 下载，约 100MB
# 如果网络不通，可以手动下载模型放到本地目录，然后改成本地路径
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512
EMBEDDING_BATCH_SIZE = 32
# BGE v1.5 可选查询指令。保留为空可兼容已存在的旧索引；启用后请重新上传文档建库。
EMBEDDING_QUERY_INSTRUCTION = ""
EMBEDDING_CACHE_DIR = os.environ.get("EMBEDDING_CACHE_DIR", "")

# ============ 文本分块配置 ============
CHUNK_SIZE = 1500      # 每块最大字符数
CHUNK_OVERLAP = 225   # 块间重叠字符数
CHUNK_MIN_SIZE = 80

# ============ RAG 配置 ============
# 召回阶段刻意不使用相似度阈值，避免正确答案被过早过滤。
TOP_K = 6
DENSE_CANDIDATES_PER_DOCUMENT = 20
BM25_CANDIDATES = 30
FUSION_CANDIDATES = 50
RERANK_TOP_K = 6
RRF_K = 60

# CrossEncoder 为可选的本地精排。首次启用会下载/加载模型；无可用模型时自动退回融合精排。
ENABLE_RERANKER = False
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_BATCH_SIZE = 16
RERANKER_CANDIDATES = 30

NEIGHBOR_EXPANSION_HITS = 3
# 融合降级分数不是余弦分数，因此以进入精排 Top 命中的结果作为扩展资格。
NEIGHBOR_EXPANSION_THRESHOLD = 0.0
CONTEXT_MAX_CHARS = 9000
CONTEXT_MAX_SOURCES = 8

# 旧配置名仅为兼容外部调用；新检索流程不在召回阶段使用它。
SIMILARITY_THRESHOLD = 0.0

# ============ LLM 上下文窗口 ============
CONTEXT_WINDOW = 32768  # qwen2:7b 支持的最大上下文 token 数
OLLAMA_NUM_CTX = 8192   # Ollama 实际使用的上下文窗口（需 <= CONTEXT_WINDOW）
