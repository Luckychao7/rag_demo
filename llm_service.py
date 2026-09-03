"""
LLM 服务 - 封装 Ollama REST API 调用
支持普通对话和流式输出
"""
import json
import requests
import config

# 存储上一次请求的 token 信息，供 UI 读取
_last_request_info = {
    "msg_count": 0,
    "total_chars": 0,
    "est_tokens": 0,
    "context_window": config.OLLAMA_NUM_CTX,
    "usage_pct": 0.0,
}


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数。中文约 1 字 = 1.5 token，英文约 4 字符 = 1 token。"""
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_count = len(text) - chinese_count
    return int(chinese_count * 1.5 + other_count / 4)


def _count_messages(messages: list) -> dict:
    """统计消息列表的总字符数和估算 token 数"""
    total_chars = 0
    for m in messages:
        total_chars += len(m.get("content", ""))
    est_tokens = _estimate_tokens("".join(m.get("content", "") for m in messages))
    return {"msg_count": len(messages), "total_chars": total_chars, "est_tokens": est_tokens}


def get_last_request_info() -> dict:
    """供 UI 读取上一次 LLM 请求的 token 用量"""
    return _last_request_info


def reset_last_request_info():
    """重置请求信息（新建/切换对话时调用）"""
    global _last_request_info
    _last_request_info = {
        "msg_count": 0,
        "total_chars": 0,
        "est_tokens": 0,
        "context_window": config.OLLAMA_NUM_CTX,
        "usage_pct": 0.0,
    }


def chat(messages: list, temperature: float = 0.3, stream: bool = False):
    """
    调用 Ollama 的 /api/chat 接口
    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    返回: 字符串（非流式）或生成器（流式）
    """
    global _last_request_info

    # 预估 token 用量（请求前）
    info = _count_messages(messages)
    info["context_window"] = config.OLLAMA_NUM_CTX
    info["usage_pct"] = round(info["est_tokens"] / config.OLLAMA_NUM_CTX * 100, 1)

    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=120, stream=stream)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"无法连接 Ollama（{config.OLLAMA_BASE_URL}），请确认 Ollama 已启动并运行了模型 {config.OLLAMA_MODEL}"
        )

    if stream:
        return _stream_response(resp)
    else:
        data = resp.json()
        # 用 Ollama 返回的真实 token 计数替换预估值
        actual_tokens = data.get("prompt_eval_count")
        if actual_tokens is not None and actual_tokens > 0:
            info["est_tokens"] = actual_tokens
            info["usage_pct"] = round(actual_tokens / config.OLLAMA_NUM_CTX * 100, 1)
        _last_request_info = info
        print(f"[LLM] 消息数: {info['msg_count']}, "
              f"总字符: {info['total_chars']}, "
              f"实际 tokens: {info['est_tokens']}, "
              f"输出 tokens: {data.get('eval_count', '?')}, "
              f"窗口: {config.OLLAMA_NUM_CTX}, "
              f"使用率: {info['usage_pct']}%")
        return data["message"]["content"]


def _stream_response(resp):
    """解析 Ollama 流式响应，逐块 yield 文本"""
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        if content:
            yield content
        if data.get("done", False):
            break


def is_available() -> bool:
    """检测 Ollama 服务是否可用"""
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list:
    """列出 Ollama 已下载的模型"""
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
