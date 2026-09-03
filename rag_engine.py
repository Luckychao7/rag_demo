"""
RAG 引擎 - 核心业务逻辑
流程：分析问题 → 追问补全 → 检索文档 → 生成答案（附来源引用）
"""
import json
import os
import config
import llm_service
from vector_store import VectorStoreManager

# 全局向量库管理器
#vsm = VectorStoreManager()

# 改成懒加载
_vsm = None

def get_vsm():
    global _vsm
    if _vsm is None:
        from vector_store import VectorStoreManager
        _vsm = VectorStoreManager()
    return _vsm

# ---------- 系统提示词 ----------

SYSTEM_PROMPT = """你是一个专业的企业内部文档问答助手。你的职责是根据上传的公司内部文档来回答用户的问题。

当前已索引文档：
{doc_metadata}

核心规则：
1. 你只能根据提供的参考文档来回答，不能编造任何信息
2. 如果参考文档中没有相关信息，必须明确说"根据现有文档，我无法找到相关答案"
3. 回答时必须标注信息来源，格式为 [来源: 文档名]
4. 如果多个文档都有相关内容，分别标注
5. 回答要准确、简洁、专业
6. 直接给出答案，不要描述你的思考过程、推理步骤或分析逻辑
7. 如果问题与文档内容无关（如天气、时间、新闻、闲聊），直接说"这个问题超出文档范围，我无法回答"，不要编造信息
8. 当用户询问文档数量时，严格区分"文档数"（上传的文件数）和"文本块数"（文档分块后的片段数），以上已索引文档列表的条目数即为文档数
9. 如果你对任何问题的答案不确定，必须明确说"我不知道"，严禁编造数字、名称、日期等任何具体信息
10. 如果用户问的是"有几个""有多少"等数量类问题，先给出数字答案，再简要列举。不要把所有内容都罗列出来
11. 如果多个参考片段描述的是同一个项目/实体/主题的不同方面，合并为一个条目，不要重复计数或重复列举"""

ANALYZE_PROMPT = """基于对话历史，分析用户的最新问题。

任务：
1. 解析代词（他/她/它/这个/那个/该）为对话历史中的具体实体
2. 补全省略的上下文
3. 如果即使有对话历史，问题仍然太模糊无法搜索，回复 [NEED_CLARIFY] 加一句简短追问（追问不超过30字）
4. 如果问题已经足够明确，直接返回改写后的检索查询（自包含、无代词）

规则：
- 只返回检索查询文本，或 [NEED_CLARIFY] 加追问
- 不要任何解释
- 改写后的查询应该是自包含的，不依赖对话历史也能理解
- 如果问题本来就没有代词且意思明确，原样返回

对话历史：
{history}

用户最新问题：{question}
"""

RAG_PROMPT_TEMPLATE = """根据以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

回答规则：
1. 只根据上述参考文档回答，不要编造信息
2. 如果参考文档中没有相关信息，明确说"根据现有文档，我无法找到相关答案"
3. 回答中引用来源，格式 [来源: 文档名]
4. 回答要准确、简洁
5. 直接给出答案，不要描述思考过程或推理步骤
6. 如果问题与文档内容无关（如天气、时间、新闻、闲聊），直接说"这个问题超出文档范围，我无法回答"
7. 如果多个参考片段描述的是同一个项目/实体/主题，合并为一个条目回答，不要重复列举
"""


# ---------- 对话历史持久化 ----------

def _history_path(conv_id: str) -> str:
    return os.path.join(config.HISTORY_DIR, f"{conv_id}.json")


def save_history(conv_id: str, messages: list):
    with open(_history_path(conv_id), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_history(conv_id: str) -> list:
    path = _history_path(conv_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def list_histories() -> list:
    files = []
    for f in os.listdir(config.HISTORY_DIR):
        if f.endswith(".json"):
            path = os.path.join(config.HISTORY_DIR, f)
            with open(path, "r", encoding="utf-8") as fp:
                msgs = json.load(fp)
            # 取第一条用户消息作为会话标题
            title = "新对话"
            for m in msgs:
                if m["role"] == "user":
                    title = m["content"][:30]
                    break
            files.append({"id": f[:-5], "title": title, "msg_count": len(msgs), "_mtime": os.path.getmtime(path)})
    # 按修改时间降序排列，最新对话在最前面
    files.sort(key=lambda x: x["_mtime"], reverse=True)
    for f in files:
        del f["_mtime"]
    return files


def delete_history(conv_id: str):
    path = _history_path(conv_id)
    if os.path.exists(path):
        os.remove(path)


# ---------- 核心流程 ----------

def assess_question(question: str, chat_history: list = None) -> dict:
    """
    分析问题并重写查询（合并代词消解 + 清晰度判断为一步 LLM 调用）
    返回: {"need_clarify": bool, "clarify_question": str, "refined_query": str}
    """
    # 构建历史摘要（最近 6 条消息）
    history_text = ""
    if chat_history:
        for m in chat_history[-6:]:
            if m["role"] in ("user", "assistant"):
                prefix = "用户" if m["role"] == "user" else "助手"
                history_text += f"{prefix}: {m['content'][:300]}\n"

    messages = [
        {"role": "system", "content": ANALYZE_PROMPT.format(history=history_text, question=question)},
        {"role": "user", "content": question},
    ]

    try:
        response = llm_service.chat(messages, temperature=0.1)
        response = response.strip()

        if "[NEED_CLARIFY]" in response:
            clarify = response.replace("[NEED_CLARIFY]", "").strip()
            print(f"[分析] 原始='{question[:50]}' → 需要追问: {clarify[:50]}")
            return {"need_clarify": True, "clarify_question": clarify, "refined_query": question}
        else:
            rewritten = response.strip()
            if len(rewritten) < 5:
                rewritten = question
            print(f"[分析] 原始='{question[:50]}' → 重写='{rewritten[:50]}'")
            return {"need_clarify": False, "clarify_question": "", "refined_query": rewritten}
    except Exception as e:
        # LLM 调用失败，降级为原始查询
        print(f"[分析] LLM调用失败({e})，降级为原始查询")
        return {"need_clarify": False, "clarify_question": "", "refined_query": question}


def search_documents(query: str, doc_ids: list = None) -> list:
    """在向量库中搜索相关文档片段"""
    return get_vsm().search(query, doc_ids=doc_ids)


def generate_answer(question: str, search_results: list, chat_history: list = None) -> dict:
    """根据检索结果生成带引用的回答"""
    # 构建上下文
    context_parts = []
    sources = []
    for i, r in enumerate(search_results):
        context_parts.append(f"[文档{i+1}: {r['doc_name']}]\n{r['text']}")
        sources.append({
            "doc_name": r["doc_name"],
            "doc_id": r["doc_id"],
            "text": r["text"][:200],
            "score": r["score"],
        })

    context = "\n\n---\n\n".join(context_parts)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # 注入已索引文档元数据到系统提示（直接写出总数，不让 LLM 自己数）
    docs = get_vsm().list_documents()
    if docs:
        total_chunks = sum(d["chunk_count"] for d in docs)
        doc_lines = "\n".join(f"- {d['doc_name']}（{d['chunk_count']}个文本块）" for d in docs)
        doc_metadata = f"共{len(docs)}个文档，{total_chunks}个文本块：\n{doc_lines}"
    else:
        doc_metadata = "暂无已索引文档"
    system_prompt = SYSTEM_PROMPT.format(doc_metadata=doc_metadata)

    # 构建消息列表（含历史，排除最后一条用户消息避免重复）
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        history = chat_history[:-1] if chat_history[-1].get("role") == "user" else chat_history
        for m in history[-6:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": prompt})

    answer = llm_service.chat(messages, temperature=0.3)
    return {"answer": answer, "sources": sources}
