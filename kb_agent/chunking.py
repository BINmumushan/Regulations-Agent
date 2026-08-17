"""结构感知切块：识别中文政策法规文档的章节/条款标题，保留标题上下文。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_int_env

_HEADING_PATTERNS = (
    re.compile(r"^\s*第[一二三四五六七八九十百千0-9]+[章节部分编][\s、.．:：]"),
    re.compile(r"^\s*第[一二三四五六七八九十百千0-9]+条[\s、.．:：]"),
    re.compile(r"^\s*[一二三四五六七八九十]+[、.．]\s*\S"),
    re.compile(r"^\s*[（(][一二三四五六七八九十]+[）)]\s*\S"),
    re.compile(r"^\s*[（(][0-9]{1,2}[）)]\s*\S"),
    re.compile(r"^\s*[0-9]{1,3}[、.．]\s*\S"),
)

_TOP_HEADING = re.compile(
    r"^\s*(第[一二三四五六七八九十百千0-9]+[章节部分编]|[一二三四五六七八九十]+[、.．])\s*\S"
)

_SUB_HEADING = re.compile(
    r"^\s*([（(][一二三四五六七八九十0-9]{1,3}[）)]|第[一二三四五六七八九十百千0-9]+条)[\s、.．:：]"
)


@dataclass
class _Block:
    text: str
    source: str
    page: int | None
    section: str
    section_path: str
    extra: dict[str, object] = field(default_factory=dict)


def _detect_heading(line: str) -> bool:
    return any(pattern.match(line) for pattern in _HEADING_PATTERNS)


def _page_blocks(doc: Document, path: list[str]) -> tuple[list[_Block], list[str]]:
    source = str(doc.metadata.get("source", "未知来源"))
    page = doc.metadata.get("page")
    extra = {
        key: value
        for key, value in doc.metadata.items()
        if key not in {"source", "page", "section", "section_path"}
    }
    seed_path = str(doc.metadata.get("section_path", "")).strip()
    seed_section = str(doc.metadata.get("section", "")).strip()
    local_path = (
        [part.strip() for part in seed_path.split(">") if part.strip()]
        if seed_path
        else list(path)
    )
    blocks: list[_Block] = []
    for line in doc.page_content.splitlines():
        line = line.strip()
        if not line:
            continue
        if _detect_heading(line):
            heading = line[:80]
            if _TOP_HEADING.match(line):
                local_path = [heading]
            else:
                local_path = local_path[:2] + [heading]
            section_path = " > ".join(local_path)
            blocks.append(
                _Block(
                    text=line,
                    source=source,
                    page=page,
                    section=heading,
                    section_path=section_path,
                    extra=extra,
                )
            )
        elif blocks:
            blocks[-1].text += "\n" + line
        else:
            blocks.append(
                _Block(
                    text=line,
                    source=source,
                    page=page,
                    section=seed_section,
                    section_path=" > ".join(local_path) if local_path else "",
                    extra=extra,
                )
            )
    return blocks, local_path


def _split_oversized(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", "。", "；", "，", " ", ""],
    )
    return [part for part in splitter.split_text(text) if part.strip()]


def _merge_blocks(blocks: list[_Block], chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunks: list[Document] = []
    buffer: list[tuple[str, str, str]] = []
    buffer_len = 0
    source = ""
    page: int | None = None
    extra: dict[str, object] = {}

    def flush(keep_overlap: bool = True) -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        section = next((item[1] for item in buffer if item[1]), "")
        section_path = next((item[2] for item in buffer if item[2]), "")
        text = "\n".join(item[0] for item in buffer).strip()
        if text:
            metadata = {
                "source": source,
                "page": page,
                "section": section,
                "section_path": section_path,
            }
            metadata.update(extra)
            source_label = f"【来源：{source}】" if source else ""
            chunks.append(
                Document(
                    page_content=f"{source_label}\n{text}" if source_label else text,
                    metadata=metadata,
                )
            )
        kept: list[tuple[str, str, str]] = []
        kept_len = 0
        if keep_overlap:
            for item in reversed(buffer):
                if kept_len + len(item[0]) <= chunk_overlap:
                    kept.append(item)
                    kept_len += len(item[0])
                else:
                    break
        buffer = list(reversed(kept))
        buffer_len = kept_len

    for block in blocks:
        if block.source != source and buffer:
            flush(keep_overlap=False)
        if block.section and buffer:
            flush(keep_overlap=False)
        if not buffer:
            source = block.source
            page = block.page
            extra = block.extra

        if len(block.text) > chunk_size:
            for part in _split_oversized(block.text, chunk_size, chunk_overlap):
                if not buffer:
                    source = block.source
                    page = block.page
                    extra = block.extra
                buffer.append((part, block.section or "", block.section_path or ""))
                buffer_len += len(part)
                if buffer_len >= chunk_size:
                    flush()
            continue

        if buffer and buffer_len + len(block.text) > chunk_size:
            flush()
            if not buffer:
                source = block.source
                page = block.page
                extra = block.extra
        buffer.append((block.text, block.section or "", block.section_path or ""))
        buffer_len += len(block.text)
        if buffer_len >= chunk_size:
            flush()

    flush()
    return chunks


def split_documents(documents: list[Document]) -> list[Document]:
    """按文档结构切块，返回带 source/page/section 元数据的片段。"""
    chunk_size = get_int_env("CHUNK_SIZE", 1000)
    chunk_overlap = get_int_env("CHUNK_OVERLAP", 200)
    blocks: list[_Block] = []
    path: list[str] = []
    last_source: str | None = None
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        if source != last_source:
            path = []
        last_source = source
        page_blocks, path = _page_blocks(doc, path)
        blocks.extend(page_blocks)
    return _merge_blocks(blocks, chunk_size, chunk_overlap)
