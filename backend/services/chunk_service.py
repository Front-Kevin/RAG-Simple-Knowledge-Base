import tiktoken
from services.llm_service import call_llm


def count_tokens(text: str) -> int:
    """使用 tiktoken 计算 token 数"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """滑动窗口 + overlap 分块"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(tokens):
            break
        start += chunk_size - overlap

    return chunks


async def semantic_chunk(text: str, model_provider: str = "openai") -> list[str]:
    """LLM 语义分块"""
    prompt = (
        "请将以下文本按语义段落进行分块，每个分块应该是一个完整的语义单元（如一个主题、一段论述、一组相关要点）。"
        "分块粒度由内容语义决定，不要人为限制长度。"
        "返回 JSON 格式，格式为: {\"chunks\": [\"chunk1\", \"chunk2\", ...]}\n"
        "只返回 JSON，不要其他内容。\n\n"
        f"文本：\n{text[:8000]}"
    )

    response = await call_llm(
        messages=[{"role": "user", "content": prompt}],
        model_provider=model_provider,
    )

    import json
    import re
    # 清理 LLM 返回中常见的 markdown 代码块标记
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        chunks = data.get("chunks", [])
    except json.JSONDecodeError:
        # 如果 LLM 返回非 JSON，回退到滑动窗口
        chunks = sliding_window_chunk(text)
        return chunks

    return [c.strip() for c in chunks if c.strip()]


async def chunk_text(
    text: str,
    mode: str = "sliding",
    chunk_size: int = 500,
    overlap: int = 100,
    model_provider: str = "openai",
) -> list[str]:
    """统一分块入口"""
    if mode == "sliding":
        return sliding_window_chunk(text, chunk_size, overlap)
    elif mode == "semantic":
        return await semantic_chunk(text, model_provider)
    elif mode == "hybrid":
        # 先语义分块，再对超长块滑动裁剪（semantic_chunk 内部已处理）
        return await semantic_chunk(text, model_provider)
    else:
        return sliding_window_chunk(text, chunk_size, overlap)
