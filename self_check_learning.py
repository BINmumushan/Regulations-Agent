from __future__ import annotations

import pickle
import re
from pathlib import Path

from kb_agent.chunking import split_documents
from kb_agent.loaders import SUPPORTED_SUFFIXES, load_file


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> None:
    with open("data/vectorstore/index.pkl", "rb") as handle:
        store_docs = list(pickle.load(handle)[0]._dict.values())
    store_by_source: dict[str, list] = {}
    for doc in store_docs:
        store_by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)

    files = sorted(
        path
        for path in Path("data/pdfs").rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")
    )
    issues: list[str] = []
    checked = 0
    duplicate_sources = 0
    seen_sources: set[str] = set()

    for path in files:
        name = path.name
        # 向量库按 source 去重：同名文件（跨目录副本）只检查一次，
        # 避免扫描件 OCR 噪声差异导致的误报。
        if name in seen_sources:
            duplicate_sources += 1
            continue
        seen_sources.add(name)
        raw = load_file(path)
        raw_text = "\n".join(doc.page_content for doc in raw)
        chunked = split_documents(raw)
        store_chunks = store_by_source.get(name, [])
        checked += 1

        problems: list[str] = []
        if not raw_text.strip():
            problems.append("raw text is empty")
        if len(chunked) != len(store_chunks):
            problems.append(f"chunked={len(chunked)} store={len(store_chunks)}")
        if not store_chunks:
            problems.append("missing in vectorstore")

        if not problems:
            store_text = normalize("\n".join(doc.page_content for doc in store_chunks))
            missing_lines: list[str] = []
            for line in raw_text.splitlines():
                stripped = line.strip()
                if len(stripped) < 8 or normalize(stripped) in store_text:
                    continue
                # 超长表格行可能被 chunk 边界从中间切开，按片段确认内容仍然完整
                parts = [
                    part.strip()
                    for part in re.split(r"[|。；;，,、：:（）()\s]+", stripped)
                    if len(part.strip()) >= 8
                ]
                if parts and all(normalize(part) in store_text for part in parts):
                    continue
                if len(stripped) >= 8:
                    missing_lines.append(stripped[:60])
            if len(missing_lines) > 0:
                sample = " | ".join(missing_lines[:5])
                problems.append(f"{len(missing_lines)} missing raw lines: {sample}")

        if problems:
            issues.append(f"{name}: {'; '.join(problems)}")
            print("ISSUE", name, "->", "; ".join(problems))

    print(
        f"checked={checked} issues={len(issues)} "
        f"duplicate_sources={duplicate_sources}"
    )
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
