"""打印问题检索到的知识库片段，用于调优检索效果。"""

from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kb_agent.ingest import get_embeddings
from kb_agent.qa import (
    _all_documents,
    _local_source_names,
    _needs_other_region_docs,
    build_retriever,
    detect_tier,
    load_vectorstore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="查看问题命中的知识库片段")
    parser.add_argument("question", help="要检查的问题")
    parser.add_argument("-k", "--top-k", type=int, default=None)
    parser.add_argument("-t", "--score-threshold", type=float, default=None)
    args = parser.parse_args()

    vectorstore = load_vectorstore(embedding=get_embeddings())
    tier = detect_tier(args.question)
    target_sources = None
    if tier is None and not _needs_other_region_docs(args.question):
        local_sources = sorted(_local_source_names(_all_documents(vectorstore)))
        if local_sources:
            target_sources = local_sources
    retriever = build_retriever(
        vectorstore,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        tier=tier,
        target_sources=target_sources,
    )
    documents = retriever.invoke(args.question)
    print(f"检索到 {len(documents)} 个片段")
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        print(f"[{index}] {source} 第 {page} 页 章节：{section}")
        print(doc.page_content[:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
