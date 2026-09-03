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

# ============ 文本分块配置 ============
CHUNK_SIZE = 1000      # 每块最大字符数
CHUNK_OVERLAP = 150    # 块间重叠字符数

# ============ RAG 配置 ============
TOP_K = 5              # 检索返回的最相关片段数
SIMILARITY_THRESHOLD = 0.25  # 相似度阈值，低于此值的结果会被过滤

# ============ LLM 上下文窗口 ============
CONTEXT_WINDOW = 32768  # qwen2:7b 支持的最大上下文 token 数
OLLAMA_NUM_CTX = 8192   # Ollama 实际使用的上下文窗口（需 <= CONTEXT_WINDOW）
