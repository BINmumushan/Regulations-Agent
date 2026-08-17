"""集中读取 .env 与默认路径配置。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 国内网络下 HuggingFace 直连不稳定，默认走镜像；Xet 协议在国内常被阻断，改用普通 HTTP。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_int_env(name: str, default: int) -> int:
    try:
        return int(get_env(name, str(default)))
    except ValueError:
        return default


def get_float_env(name: str, default: float) -> float:
    try:
        return float(get_env(name, str(default)))
    except ValueError:
        return default


def pdf_dir() -> Path:
    return Path(get_env("PDF_DIR", str(DEFAULT_PDF_DIR))).expanduser().resolve()


def vectorstore_dir() -> Path:
    return Path(get_env("VECTORSTORE_DIR", str(DEFAULT_VECTORSTORE_DIR))).expanduser().resolve()


def demo_password() -> str:
    """演示版访问密码；未配置时表示不启用密码门。"""
    return get_env("DEMO_PASSWORD", "")
