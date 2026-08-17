"""把知识库文档切块、向量化并保存到本地 FAISS 向量库。"""

from __future__ import annotations

import logging
from pathlib import Path

# 先加载配置（设置镜像环境变量），再导入 fastembed，保证模型下载使用国内镜像。
from .config import get_env, pdf_dir, vectorstore_dir
from .chunking import split_documents
from .loaders import SUPPORTED_SUFFIXES, load_file

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    provider = get_env("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider == "local":
        return FastEmbedEmbeddings(
            model_name=get_env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        )
    if provider == "openai":
        if not get_env("OPENAI_API_KEY"):
            raise RuntimeError("未配置 OPENAI_API_KEY，请先复制 .env.example 为 .env 并填写。")
        return OpenAIEmbeddings(
            model=get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=get_env("OPENAI_API_KEY"),
            base_url=get_env("OPENAI_BASE_URL") or None,
        )
    raise ValueError(f"EMBEDDING_PROVIDER 只支持 local 或 openai，当前为: {provider}")


def load_documents(directory: Path | None = None) -> list[Document]:
    directory = Path(directory) if directory else pdf_dir()
    if not directory.exists():
        raise FileNotFoundError(f"知识库目录不存在: {directory}")
    files = sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not path.name.startswith("~$")
        )
    )
    if not files:
        raise FileNotFoundError(
            f"知识库目录中没有受支持的文件: {directory}，"
            f"支持格式: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    documents: list[Document] = []
    for path in files:
        try:
            documents.extend(load_file(path))
        except Exception:
            logger.exception("读取文件失败，已跳过: %s", path)
    if not documents:
        raise RuntimeError(
            "所有文件读取后都为空，请检查 PDF 是否为扫描件（需要安装 rapidocr-onnxruntime）或文件是否已加密。"
        )
    return documents


def _existing_sources(vectorstore: FAISS) -> set[str]:
    try:
        return {
            str(doc.metadata.get("source", ""))
            for doc in vectorstore.docstore._dict.values()
        }
    except Exception:
        logger.warning("无法读取已有索引的来源信息，将追加全部内容。")
        return set()


def ingest(
    directory: Path | None = None,
    index_directory: Path | None = None,
    embedding: Embeddings | None = None,
    rebuild: bool = False,
) -> int:
    """读取目录中的文档并写入向量库，返回本次新增的文本片段数量。"""
    documents = load_documents(directory)
    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError("切块后没有可索引的内容。")

    index_dir = Path(index_directory) if index_directory else vectorstore_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedding or get_embeddings()

    if rebuild or not (index_dir / "index.faiss").exists():
        vectorstore = FAISS.from_documents(chunks, embedder)
        added_count = len(chunks)
    else:
        vectorstore = FAISS.load_local(
            str(index_dir), embedder, allow_dangerous_deserialization=True
        )
        existing = _existing_sources(vectorstore)
        new_chunks = [chunk for chunk in chunks if chunk.metadata.get("source") not in existing]
        if not new_chunks:
            return 0
        vectorstore.add_documents(new_chunks)
        added_count = len(new_chunks)

    vectorstore.save_local(str(index_dir))
    return added_count
