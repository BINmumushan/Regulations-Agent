"""知识库入库入口：python ingest_pdfs.py [--pdf-dir ...] [--rebuild]"""

from __future__ import annotations

import argparse

from kb_agent.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把知识库文档（PDF/文本/CSV/DOCX/Excel）切块并建立向量索引"
    )
    parser.add_argument("--pdf-dir", default=None, help="知识库目录（默认 data/pdfs）")
    parser.add_argument("--index-dir", default=None, help="向量库目录（默认 data/vectorstore）")
    parser.add_argument("--rebuild", action="store_true", help="重建索引；不指定时追加新文件")
    args = parser.parse_args()

    added = ingest(
        directory=args.pdf_dir,
        index_directory=args.index_dir,
        rebuild=args.rebuild,
    )
    print(f"入库完成，新增 {added} 个文本片段。")


if __name__ == "__main__":
    main()
