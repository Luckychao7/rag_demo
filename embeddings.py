"""
Embedding 模块 - 加载 BGE 中文模型，提供向量化接口
首次运行会自动从 HuggingFace 下载模型（约 100MB）
如果网络不通，可手动下载后改 config.py 中的路径为本地目录
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', r'D:\question_answer\hf_cache')
os.environ['TRANSFORMERS_VERBOSITY'] = 'critical'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import config

_model = None


def get_embedding_model():
    """懒加载 embedding 模型，全局只加载一次"""
    global _model
    if _model is None:
        print(f"正在加载 Embedding 模型: {config.EMBEDDING_MODEL} ...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logging.disable(logging.CRITICAL)
        print("Embedding 模型加载完成")
    return _model


def embed_texts(texts: list) -> list:
    """将文本列表转为向量列表"""
    model = get_embedding_model()
    # normalize_embeddings 让余弦相似度可以用内积代替
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list:
    """将单条查询文本转为向量"""
    return embed_texts([text])[0]
