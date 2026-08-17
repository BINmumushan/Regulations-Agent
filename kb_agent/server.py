"""FastAPI 服务：/ask 提问，/upload 上传并入库。"""

from __future__ import annotations

import secrets
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import demo_password, pdf_dir
from .ingest import ingest
from .loaders import SUPPORTED_SUFFIXES
from .pdf_export import build_conversation_pdf, build_qa_pdf
from .qa import AnswerResult, answer_question

app = FastAPI(title="知识库问答 Agent", version="0.1.0")
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MAX_DEMO_PASSWORD_ATTEMPTS = 3
DEMO_PASSWORD_LOCK_SECONDS = 180

_demo_auth_lock = threading.Lock()
_demo_failed_attempts: dict[str, int] = {}
_demo_locked_until: dict[str, float] = {}


class ConversationTurn(BaseModel):
    role: str = Field(..., description="消息角色：user 或 assistant")
    content: str = Field(..., description="消息内容")
    sources: list[str] = Field(default_factory=list, description="该回答的来源文件")
    unknown: bool = Field(default=False, description="该回答是否为委婉未答")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    history: list[ConversationTurn] = Field(
        default_factory=list, description="历史对话（user/assistant 消息）"
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    unknown: bool = False
    web: bool = False


class PdfExportMessage(BaseModel):
    role: str = Field(..., description="消息角色：user 或 assistant")
    content: str = Field(..., description="消息内容")
    sources: list[str] = Field(default_factory=list, description="来源文件列表")
    unknown: bool = Field(default=False, description="该回答是否为委婉未答")


class PdfExportRequest(BaseModel):
    question: str | None = Field(default=None, description="用户问题（单条导出时使用）")
    answer: str | None = Field(default=None, description="回答内容（单条导出时使用）")
    sources: list[str] = Field(default_factory=list, description="来源文件列表")
    messages: list[PdfExportMessage] | None = Field(
        default=None, description="完整对话记录（导出整段对话时使用）"
    )


class AuthCheckRequest(BaseModel):
    password: str = Field(default="", description="演示访问密码")


def _demo_client_key(client_id: str | None) -> str:
    return client_id or "anonymous"


def _demo_lock_remaining(client_id: str) -> float:
    with _demo_auth_lock:
        until = _demo_locked_until.get(client_id)
        if until is not None and until <= time.time():
            _demo_locked_until.pop(client_id, None)
            _demo_failed_attempts.pop(client_id, None)
            return 0.0
        if until is None:
            return 0.0
        return until - time.time()


def _demo_locked_detail(remaining: float) -> str:
    if remaining <= 0:
        return "尝试次数过多，已禁止使用。"
    minutes = int(remaining // 60) + 1
    return f"尝试次数过多，已禁止使用，请约 {minutes} 分钟后再试。"


def _demo_record_failure(client_id: str) -> bool:
    """记录一次失败；达到上限时锁定并返回 True。"""
    with _demo_auth_lock:
        count = _demo_failed_attempts.get(client_id, 0) + 1
        if count >= MAX_DEMO_PASSWORD_ATTEMPTS:
            _demo_locked_until[client_id] = time.time() + DEMO_PASSWORD_LOCK_SECONDS
            _demo_failed_attempts.pop(client_id, None)
            return True
        _demo_failed_attempts[client_id] = count
        return False


def _demo_reset(client_id: str) -> None:
    with _demo_auth_lock:
        _demo_failed_attempts.pop(client_id, None)
        _demo_locked_until.pop(client_id, None)


def _demo_remaining_attempts(client_id: str) -> int:
    with _demo_auth_lock:
        count = _demo_failed_attempts.get(client_id, 0)
        return max(0, MAX_DEMO_PASSWORD_ATTEMPTS - count)


def _check_demo_password_with_client(
    x_demo_password: str | None,
    x_demo_client_id: str | None,
) -> None:
    expected = demo_password()
    if not expected:
        return
    client_id = _demo_client_key(x_demo_client_id)
    remaining = _demo_lock_remaining(client_id)
    if remaining > 0:
        raise HTTPException(status_code=429, detail=_demo_locked_detail(remaining))
    if not x_demo_password or not secrets.compare_digest(x_demo_password, expected):
        if _demo_record_failure(client_id):
            raise HTTPException(status_code=429, detail=_demo_locked_detail(0))
        remaining = _demo_remaining_attempts(client_id)
        raise HTTPException(
            status_code=401,
            detail=f"密码错误，还剩 {remaining} 次机会。",
        )
    _demo_reset(client_id)


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(STATIC_INDEX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/status")
def auth_status(
    x_demo_client_id: str | None = Header(default=None),
) -> dict[str, object]:
    client_id = _demo_client_key(x_demo_client_id)
    remaining = _demo_lock_remaining(client_id) if demo_password() else 0.0
    return {
        "enabled": bool(demo_password()),
        "locked": remaining > 0,
        "retry_after": int(remaining),
    }


@app.post("/auth/check")
def auth_check(
    payload: AuthCheckRequest,
    x_demo_client_id: str | None = Header(default=None),
) -> dict[str, bool]:
    _check_demo_password_with_client(payload.password, x_demo_client_id)
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    x_demo_password: str | None = Header(default=None),
    x_demo_client_id: str | None = Header(default=None),
) -> AnswerResult:
    _check_demo_password_with_client(x_demo_password, x_demo_client_id)
    try:
        return answer_question(
            payload.question,
            history=[turn.model_dump() for turn in payload.history],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc


@app.post("/export-pdf", response_class=Response)
def export_pdf(
    payload: PdfExportRequest,
    x_demo_password: str | None = Header(default=None),
    x_demo_client_id: str | None = Header(default=None),
) -> Response:
    """把单条问答或整段对话记录导出为 PDF 下载。"""
    _check_demo_password_with_client(x_demo_password, x_demo_client_id)
    try:
        if payload.messages is not None:
            if not payload.messages:
                raise HTTPException(status_code=400, detail="没有可导出的对话内容")
            pdf_bytes = build_conversation_pdf(
                [message.model_dump() for message in payload.messages]
            )
        else:
            if not payload.question or not payload.answer:
                raise HTTPException(
                    status_code=400, detail="question/answer 或 messages 必须提供"
                )
            pdf_bytes = build_qa_pdf(
                question=payload.question,
                answer=payload.answer,
                sources=payload.sources,
            )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {exc}") from exc
    filename = f"知识库问答_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{quote(filename)}"
            )
        },
    )


@app.post("/upload")
def upload(
    file: UploadFile = File(...),
    x_demo_password: str | None = Header(default=None),
    x_demo_client_id: str | None = Header(default=None),
) -> dict[str, object]:
    _check_demo_password_with_client(x_demo_password, x_demo_client_id)
    directory = pdf_dir()
    directory.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "upload.pdf").name
    stem = Path(original_name).stem or "upload"
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件类型: {suffix or '无扩展名'}，"
                f"支持: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            ),
        )
    target = directory / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}"

    try:
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    try:
        added = ingest(directory=directory)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"入库失败: {exc}") from exc
    return {"filename": target.name, "added_chunks": added, "status": "ok"}
