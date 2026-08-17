"""页面级 OCR 缓存与并行预识别，加速扫描 PDF 入库。"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import PROJECT_ROOT, get_int_env

logger = logging.getLogger(__name__)

_worker_engine = None


def cache_dir() -> Path:
    override = os.getenv("OCR_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (PROJECT_ROOT / "data" / "ocr_cache").resolve()


def _cache_key(path: Path, dpi: int) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|dpi={dpi}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def cached_text(path: Path, page_number: int, dpi: int = 200) -> str | None:
    cache_file = cache_dir() / _cache_key(path, dpi) / f"{page_number:04d}.txt"
    if not cache_file.exists():
        return None
    try:
        return cache_file.read_text(encoding="utf-8")
    except Exception:
        return None


def save_text(path: Path, page_number: int, text: str, dpi: int = 200) -> None:
    if not text.strip():
        return
    page_dir = cache_dir() / _cache_key(path, dpi)
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{page_number:04d}.txt").write_text(text, encoding="utf-8")


def format_ocr_result(result) -> str:
    """把 RapidOCR 结果按阅读顺序拼接成文本。"""
    items: list[tuple[float, float, str]] = []
    for box, text, _score in result:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        text = str(text).strip()
        if text:
            items.append((min(ys), min(xs), text))
    items.sort()
    return "\n".join(text for _, _, text in items)


def _worker_get_engine():
    global _worker_engine
    if _worker_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _worker_engine = RapidOCR()
    return _worker_engine


def _worker_ocr_page(args: tuple[str, int, int]) -> tuple[int, str, bool]:
    path_str, page_number, dpi = args
    path = Path(path_str)
    cached = cached_text(path, page_number, dpi)
    if cached is not None:
        return page_number, cached, True

    import numpy as np
    import pymupdf

    engine = _worker_get_engine()
    with pymupdf.open(path_str) as pdf:
        page = pdf[page_number - 1]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = pix.pil_image().convert("RGB")
        result, _ = engine(np.asarray(image)[:, :, ::-1])
    text = format_ocr_result(result) if result else ""
    save_text(path, page_number, text, dpi)
    return page_number, text, False


def scanned_page_numbers(path: Path, min_text_len: int = 8) -> list[int]:
    import pymupdf

    pages: list[int] = []
    with pymupdf.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            if len(page.get_text("text").strip()) < min_text_len:
                pages.append(page_number)
    return pages


def ocr_pdf_parallel(
    path: Path,
    dpi: int | None = None,
    workers: int | None = None,
    min_text_len: int = 8,
) -> int:
    """并行识别 PDF 中缺失缓存的扫描页，返回本次新识别的页数。"""
    pages = scanned_page_numbers(path, min_text_len)
    if not pages:
        return 0
    dpi = dpi or get_int_env("PDF_OCR_DPI", 200)
    workers = workers or max(1, min(4, os.cpu_count() or 4))
    tasks = [(str(path), page_number, dpi) for page_number in pages]
    total = len(tasks)
    new_count = 0
    done = 0

    def run_pool() -> None:
        nonlocal new_count, done
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for page_number, text, was_cached in pool.map(_worker_ocr_page, tasks, chunksize=4):
                if not was_cached and text.strip():
                    new_count += 1
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"OCR 进度: {done}/{total}", flush=True)

    try:
        run_pool()
    except Exception:
        logger.exception("并行 OCR 失败，回退到串行模式: %s", path.name)
        for page_number, text, was_cached in map(_worker_ocr_page, tasks):
            if not was_cached and text.strip():
                new_count += 1
            done += 1
            if done % 25 == 0 or done == total:
                print(f"OCR 进度: {done}/{total}", flush=True)
    return new_count
