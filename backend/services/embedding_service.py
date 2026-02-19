import os
import httpx
from openai import AsyncOpenAI
import config


def _make_http_client() -> httpx.AsyncClient | None:
    """If http_proxy is set, return an httpx client that uses it."""
    proxy = os.environ.get("http_proxy") or os.environ.get("https_proxy")
    if proxy:
        return httpx.AsyncClient(proxy=proxy)
    return None


def _get_client(model_provider: str) -> tuple[AsyncOpenAI, str]:
    """根据 provider 返回对应的 client 和 model 名"""
    http_client = _make_http_client()
    if model_provider == "bailian":
        client = AsyncOpenAI(
            api_key=config.BAILIAN_API_KEY,
            base_url=config.BAILIAN_BASE_URL,
            **({"http_client": http_client} if http_client else {}),
        )
        model = config.BAILIAN_EMBEDDING_MODEL
    else:
        client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            **({"http_client": http_client} if http_client else {}),
        )
        model = config.OPENAI_EMBEDDING_MODEL
    return client, model


async def generate_embedding(text: str, model_provider: str = "openai") -> list[float]:
    """生成单条文本的 embedding"""
    client, model = _get_client(model_provider)
    response = await client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


async def generate_embeddings(texts: list[str], model_provider: str = "openai") -> list[list[float]]:
    """批量生成 embedding"""
    client, model = _get_client(model_provider)
    response = await client.embeddings.create(
        input=texts,
        model=model,
    )
    return [item.embedding for item in response.data]
