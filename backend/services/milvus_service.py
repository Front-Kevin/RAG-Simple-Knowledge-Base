from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import config

_connected = False


def connect_milvus():
    """连接 Milvus（复用连接）"""
    global _connected
    if _connected:
        return
    connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    _connected = True


_loaded_collections: set[str] = set()


def get_or_create_collection(model_provider: str = "openai") -> Collection:
    """获取或创建 collection"""
    connect_milvus()
    dim = config.EMBEDDING_DIM.get(model_provider, 1536)
    collection_name = f"{config.MILVUS_COLLECTION}_{model_provider}"

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        if collection_name not in _loaded_collections:
            collection.load()
            _loaded_collections.add(collection_name)
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="RAG document chunks")
    collection = Collection(name=collection_name, schema=schema)

    # 创建 HNSW 索引
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": config.HNSW_M,
            "efConstruction": config.HNSW_EF_CONSTRUCTION,
        },
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    _loaded_collections.add(collection_name)
    return collection


def insert_chunks(doc_id: str, chunks: list[str], vectors: list[list[float]], model_provider: str = "openai"):
    """写入分块数据到 Milvus"""
    collection = get_or_create_collection(model_provider)
    data = [
        [doc_id] * len(chunks),   # doc_id
        chunks,                    # content
        vectors,                   # vector
    ]
    collection.insert(data)
    collection.flush()


def search_chunks(query_vector: list[float], model_provider: str = "openai", top_k: int = 5) -> list[dict]:
    """向量检索"""
    collection = get_or_create_collection(model_provider)
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": config.HNSW_EF},
    }
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["doc_id", "content"],
    )

    hits = []
    for result in results[0]:
        hits.append({
            "doc_id": result.entity.get("doc_id"),
            "content": result.entity.get("content"),
            "score": result.score,
        })
    return hits


def get_doc_chunk_count(doc_id: str, model_provider: str = "openai") -> int:
    """获取某个文档的分块数量"""
    collection = get_or_create_collection(model_provider)
    expr = f'doc_id == "{doc_id}"'
    results = collection.query(expr=expr, output_fields=["doc_id"])
    return len(results)


def get_doc_chunks(doc_id: str, model_provider: str = "openai") -> list[dict]:
    """获取某个文档在 Milvus 中存储的完整记录"""
    collection = get_or_create_collection(model_provider)
    expr = f'doc_id == "{doc_id}"'
    results = collection.query(
        expr=expr,
        output_fields=["id", "doc_id", "content", "vector"],
    )
    return results


def delete_doc_chunks(doc_id: str, model_provider: str = "openai"):
    """删除某个文档的所有分块"""
    collection = get_or_create_collection(model_provider)
    expr = f'doc_id == "{doc_id}"'
    collection.delete(expr=expr)
    collection.flush()
