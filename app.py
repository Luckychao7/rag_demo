"""
Streamlit 主界面
功能：文档上传管理 + 对话式问答 + 历史对话记忆
启动方式：streamlit run app.py
"""
import streamlit as st
import streamlit.components.v1 as components
import os
import time
import uuid

import config
import document_parser
import text_splitter
import llm_service
import rag_engine

# ==================== 页面配置 ====================
st.set_page_config(page_title="文档问答助手", page_icon="📄", layout="wide")

# ==================== 初始化 Session State ====================

def init_state():
    defaults = {
        "messages": [],           # 当前对话消息列表 [{"role":..., "content":..., "sources":...}]
        "conv_id": str(uuid.uuid4())[:8],
        "pending_question": None, # 等待用户追问的原始问题
        "selected_docs": [],      # 侧边栏选中的文档 ID
        "ollama_ok": None,        # Ollama 连接状态缓存
        "max_tokens": 0,          # 本对话累计最大 token 数
        "last_token_info": None,  # 本对话最近一次 LLM 请求的 token 信息
        "scroll_to_bottom": False, # 是否需要自动滚动到底部
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ==================== 侧边栏 ====================

def sidebar():
    with st.sidebar:
        st.title("📄 文档问答助手")
        st.caption("基于内部文档的 AI 问答系统")

        # ---- Ollama 状态 ----
        _check_ollama()

        st.divider()

        # ---- 文档上传 ----
        st.subheader("📤 文档管理")
        uploaded = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "csv", "xlsx", "txt", "md"],
            accept_multiple_files=True,
            help="支持 PDF / Word / CSV / Excel / TXT / Markdown",
        )
        if uploaded and st.button("✅ 处理上传文档", use_container_width=True):
            _process_uploads(uploaded)

        st.divider()

        # ---- 已索引文档列表 ----
        st.subheader("📚 已索引文档")
        docs = rag_engine.get_vsm().list_documents()
        if not docs:
            st.info("暂无文档，请先上传")
        else:
            doc_names = [d["doc_name"] for d in docs]
            chosen = st.multiselect("选择搜索范围（不选=全部）", doc_names)
            st.session_state.selected_docs = [
                d["doc_id"] for d in docs if d["doc_name"] in chosen
            ]
            for d in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{d['doc_name']}")
                with col2:
                    st.caption(f"{d['chunk_count']}块")
                if st.button("🗑️", key=f"del_{d['doc_id']}", help=f"删除 {d['doc_name']}"):
                    rag_engine.get_vsm().delete_document(d["doc_id"])
                    st.rerun()

        st.divider()

        # ---- 对话管理 ----
        st.subheader("💬 对话管理")
        if st.button("➕ 新建对话", use_container_width=True):
            _new_conversation()

        histories = rag_engine.list_histories()
        if histories:
            for h in histories:
                btn_label = f"{'🟢 ' if h['id'] == st.session_state.conv_id else ''}{h['title']}"
                if st.button(btn_label, key=f"hist_{h['id']}", use_container_width=True):
                    _load_conversation(h["id"])
                    st.rerun()


def _check_ollama():
    if st.session_state.ollama_ok is None:
        st.session_state.ollama_ok = llm_service.is_available()
    if st.session_state.ollama_ok:
        st.success("🟢 Ollama 已连接", icon=None)
        # 上下文窗口监控（从 session state 读取，隔离不同对话）
        info = st.session_state.last_token_info
        if info and info["est_tokens"] > 0:
            # 追踪本对话累计最大 token 数
            if info["est_tokens"] > st.session_state.max_tokens:
                st.session_state.max_tokens = info["est_tokens"]
            max_t = st.session_state.max_tokens
            ctx = info["context_window"]
            pct = round(max_t / ctx * 100, 1)
            st.caption(f"📊 上下文窗口: {ctx} tokens")
            st.caption(f"   累计峰值: ~{max_t} tokens ({pct}%)")
            st.caption(f"   剩余: ~{ctx - max_t} tokens")
            # 进度条
            st.progress(min(pct / 100, 1.0))
            if pct > 70:
                st.warning("⚠️ 上下文使用率超过 70%")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📦 新建对话", key="ctx_new", use_container_width=True):
                        _new_conversation()
                        st.rerun()
                with col2:
                    st.caption("或减少历史轮数")
        else:
            st.caption(f"📊 上下文窗口: {config.OLLAMA_NUM_CTX} tokens")
            st.caption("   暂无请求记录")
    else:
        st.error("🔴 Ollama 未连接", icon=None)
        st.warning(
            f"请先启动 Ollama 并下载模型：\n"
            f"```\nollama pull {config.OLLAMA_MODEL}\nollama serve\n```"
        )


def _process_uploads(uploaded_files):
    for uf in uploaded_files:
        with st.spinner(f"正在处理: {uf.name} ..."):
            # 查重：同名文档已存在则删旧建新
            existing_docs = rag_engine.get_vsm().list_documents()
            for d in existing_docs:
                if d["doc_name"] == uf.name:
                    rag_engine.get_vsm().delete_document(d["doc_id"])
                    st.warning(f"⚠️ 检测到同名文档已索引，已删除旧索引并重新创建")
                    break
            # 保存文件
            save_path = os.path.join(config.UPLOAD_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getvalue())
            # 解析文本
            try:
                text = document_parser.parse_document(save_path)
            except Exception as e:
                st.error(f"解析 {uf.name} 失败: {e}")
                continue
            if not text.strip():
                st.warning(f"{uf.name} 未提取到文本内容")
                continue
            # 分块
            chunks = text_splitter.split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if not chunks:
                st.warning(f"{uf.name} 分块结果为空")
                continue
            # 写入向量库
            doc_id = uf.name.rsplit(".", 1)[0] + "_" + str(int(time.time()))
            rag_engine.get_vsm().add_document(doc_id, uf.name, chunks)
            st.success(f"✅ {uf.name} → {len(chunks)} 个文本块，已索引")


def _new_conversation():
    st.session_state.conv_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.max_tokens = 0
    st.session_state.last_token_info = None
    st.session_state.scroll_to_bottom = True


def _load_conversation(conv_id):
    st.session_state.conv_id = conv_id
    st.session_state.messages = rag_engine.load_history(conv_id)
    st.session_state.pending_question = None
    st.session_state.max_tokens = 0
    st.session_state.last_token_info = None
    st.session_state.scroll_to_bottom = True


# ==================== 主聊天区域 ====================

def chat_area():
    # 顶部标签
    docs = rag_engine.get_vsm().list_documents()
    total_chunks = sum(d["chunk_count"] for d in docs)
    token_info = st.session_state.last_token_info
    token_str = f" | 上次请求: ~{token_info['est_tokens']}/{token_info['context_window']} tokens ({token_info['usage_pct']}%)" if token_info and token_info["est_tokens"] > 0 else ""
    st.caption(f"当前对话: {st.session_state.conv_id} | 已索引文档: {len(docs)} | 总文本块: {total_chunks}{token_str}")

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            if msg.get("msg_type") == "clarify":
                st.info("💬 追问：请补充信息以便我更准确地检索文档")
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # 聊天输入
    prompt = st.chat_input("输入你的问题，我会根据文档来回答...")
    if prompt:
        _handle_user_input(prompt)

    # 自动滚动到底部
    if st.session_state.scroll_to_bottom:
        st.session_state.scroll_to_bottom = False
        components.html("""
            <script>
                setTimeout(function() {
                    const el = window.parent.document.querySelector('section.main');
                    if (el) { el.scrollTop = el.scrollHeight; }
                }, 50);
            </script>
        """, height=0)


def _render_sources(sources):
    with st.expander("📎 查看引用来源"):
        for i, src in enumerate(sources):
            st.markdown(f"**来源 {i+1}: {src['doc_name']}** (相关度: {src['score']:.2f})")
            st.text(src["text"])
            st.divider()


def _handle_user_input(prompt):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # 判断是追问场景还是新问题
    if st.session_state.pending_question:
        # 用户在回答追问 → 合并问题后走完整流程
        original = st.session_state.pending_question
        combined = f"原始问题: {original}\n补充信息: {prompt}"
        st.session_state.pending_question = None
        _do_rag(combined)
    else:
        # 新问题 → 先重写查询 + 评估是否需要追问
        # 传给 assess_question 的历史排除当前消息（避免问题重复出现）
        with st.spinner("正在分析问题..."):
            assessment = rag_engine.assess_question(prompt, st.session_state.messages[:-1])
            # assess_question 内部调用了 LLM 做查询重写，记录 token 信息
            st.session_state.last_token_info = llm_service.get_last_request_info()

        if assessment["need_clarify"]:
            # 需要追问
            st.session_state.pending_question = prompt
            clarify_msg = assessment["clarify_question"]
            st.session_state.messages.append({"role": "assistant", "content": clarify_msg, "sources": [], "msg_type": "clarify"})
            with st.chat_message("assistant", avatar="🤖"):
                st.info("💬 追问：请补充信息以便我更准确地检索文档")
                st.markdown(clarify_msg)
        else:
            # 用改写后的查询做检索，用原始问题做生成
            _do_rag(prompt, search_query=assessment["refined_query"])

    # 保存对话历史
    rag_engine.save_history(st.session_state.conv_id, st.session_state.messages)
    st.session_state.scroll_to_bottom = True
    st.rerun()


def _do_rag(query, search_query=None):
    """执行 RAG 检索 + 生成
    query: 用户原始问题（传给 LLM 生成回答）
    search_query: 改写后的检索查询（传给 embedding 检索），None 时用 query
    """
    if search_query is None:
        search_query = query
    doc_ids = st.session_state.selected_docs if st.session_state.selected_docs else None

    with st.spinner("正在搜索文档..."):
        try:
            results = rag_engine.search_documents(search_query, doc_ids=doc_ids)
        except Exception as e:
            msg = f"检索出错: {e}"
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg)
            return

    if not results:
        msg = "根据现有文档，我无法找到与您问题相关的信息。请尝试换个关键词，或者上传相关文档后再试。"
        st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg)
        return

    with st.spinner("正在生成回答..."):
        try:
            result = rag_engine.generate_answer(query, results, st.session_state.messages)
        except Exception as e:
            msg = f"生成回答时出错: {e}"
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg)
            return

    # 记录本次 LLM 请求的 token 用量到 session state
    st.session_state.last_token_info = llm_service.get_last_request_info()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(result["answer"])
        _render_sources(result["sources"])


# ==================== 历史管理标签页 ====================

def history_tab():
    st.subheader("历史对话记录")
    histories = rag_engine.list_histories()
    if not histories:
        st.info("暂无历史对话")
        return

    for h in histories:
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            st.caption(f"{h['msg_count']} 条消息")
        with col2:
            st.text(h["title"])
        with col3:
            if st.button("🗑️", key=f"del_h_{h['id']}"):
                rag_engine.delete_history(h["id"])
                if h["id"] == st.session_state.conv_id:
                    _new_conversation()
                st.rerun()


# ==================== 主入口 ====================

def main():
    sidebar()

    tab1, tab2 = st.tabs(["💬 对话问答", "📋 历史管理"])
    with tab1:
        chat_area()
    with tab2:
        history_tab()


if __name__ == "__main__":
    main()
