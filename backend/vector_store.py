import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 3072  # gemini-embedding-001

# ── Lazy Loaded Clients & Embeddings ──────────────────────────────────────────

_embeddings = None
_qdrant_client = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_classic.embeddings import CacheBackedEmbeddings
        from langchain_classic.storage import LocalFileStore
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        base_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        embedding_file_store = LocalFileStore("./embedding_cache/")
        _embeddings = CacheBackedEmbeddings.from_bytes_store(
            base_embeddings,
            embedding_file_store,
            namespace=base_embeddings.model,
            query_embedding_cache=True,
            key_encoder="blake2b",
        )
    return _embeddings


def get_qdrant_client(timeout: int = 60) -> "QdrantClient":
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            timeout=timeout,
        )
    return _qdrant_client


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str):
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client.models import Distance, VectorParams

    collection_name = get_collection_name(session_id)
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)

