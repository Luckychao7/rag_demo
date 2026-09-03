"""
RAG 全链路测试脚本（不依赖 LLM，只测 embedding + 检索）
运行方式: D:\conda_envs\envs\doc_qa\python.exe test_rag.py
"""
import time
import os

print("=" * 60)
print("RAG 全链路测试")
print("=" * 60)

# 1. 测试 embedding 模型加载
print("\n[1/5] 测试 Embedding 模型加载...")
t0 = time.time()
import embeddings
model = embeddings.get_embedding_model()
t1 = time.time()
print(f"  ✓ 模型加载成功，耗时 {t1-t0:.1f}s")
print(f"  ✓ HF_HOME = {os.environ.get('HF_HOME', '未设置')}")

# 2. 测试 PDF 解析
print("\n[2/5] 测试 PDF 解析...")
import document_parser
pdf_path = os.path.join("data", "uploads", "李凯朝简历.pdf")
text = document_parser.parse_document(pdf_path)
print(f"  ✓ 解析成功，提取文本 {len(text)} 字符")
print(f"  前100字: {text[:100]}...")

# 3. 测试文本分块
print("\n[3/5] 测试文本分块...")
import text_splitter
import config
chunks = text_splitter.split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
print(f"  ✓ 分块完成，共 {len(chunks)} 个文本块")
for i, c in enumerate(chunks):
    print(f"    块{i}: {len(c['text'])}字, 位置 {c['start']}-{c['end']}")

# 4. 测试向量库写入
print("\n[4/5] 测试向量库写入（MD5 文件名）...")
from vector_store import VectorStoreManager
vsm = VectorStoreManager()
import hashlib
doc_id = "test_doc_" + str(int(time.time()))
vsm.add_document(doc_id, "李凯朝简历.pdf", chunks)

# 检查文件名
files = os.listdir(config.VECTOR_DIR)
faiss_file = [f for f in files if f.endswith(".faiss")]
meta_file = [f for f in files if f.endswith(".meta.json")]
expected_hash = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
print(f"  ✓ FAISS 文件: {faiss_file}")
print(f"  ✓ Meta 文件: {meta_file}")
print(f"  ✓ 期望 FAISS 文件名: {expected_hash}.faiss")
assert any(expected_hash in f for f in faiss_file), "FAISS 文件名 MD5 不匹配!"
print(f"  ✓ MD5 文件名验证通过（纯 ASCII，无中文乱码）")

# 5. 测试搜索
print("\n[5/5] 测试向量检索...")
queries = [
    "这个人的教育背景是什么",
    "技术栈有哪些",
    "实习经历",
]
for q in queries:
    results = vsm.search(q, top_k=3)
    print(f"\n  查询: 「{q}」")
    if results:
        for i, r in enumerate(results):
            print(f"    结果{i+1} [相关度: {r['score']:.4f}] {r['text'][:80]}...")
    else:
        print(f"    (无结果)")

# 列出文档
print("\n[验证] list_documents():")
docs = vsm.list_documents()
for d in docs:
    print(f"  doc_id: {d['doc_id']}, name: {d['doc_name']}, chunks: {d['chunk_count']}")

# 清理测试数据
vsm.delete_document(doc_id)
print(f"\n✓ 测试数据已清理（删除 {doc_id}）")

print("\n" + "=" * 60)
print("全链路测试通过！Embedding + 解析 + 分块 + 向量库 + 检索 均正常")
print("接下来可以用 Streamlit 测试完整的 RAG 问答（含 LLM 生成）")
print("=" * 60)
