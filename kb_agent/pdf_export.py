"""把问答结果导出为排版良好的中文 PDF（不显示来源，只保留问题与回答）。"""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_FALLBACK_FONT = "STSong-Light"
_FONT_CANDIDATES = (
    (Path("C:/Windows/Fonts/msyh.ttc"), "MSYH", 0),
    (Path("C:/Windows/Fonts/msyhbd.ttc"), "MSYH-Bold", 0),
    (Path("C:/Windows/Fonts/Deng.ttf"), "DengXian", None),
    (Path("C:/Windows/Fonts/Dengb.ttf"), "DengXian-Bold", None),
    (Path("C:/Windows/Fonts/simhei.ttf"), "SimHei", None),
    (Path("C:/Windows/Fonts/simsun.ttc"), "SimSun", 0),
)
_registered_fonts: set[str] = set()

_ACCENT = colors.HexColor("#1f6feb")
_TEXT = colors.HexColor("#1f2328")
_BODY = colors.HexColor("#24292f")
_MUTED = colors.HexColor("#6e7781")
_BORDER = colors.HexColor("#d8dee6")
_QUESTION_BG = colors.HexColor("#f4f8ff")
_QUESTION_BORDER = colors.HexColor("#c9d6ea")


def _try_register(path: Path, name: str, subfont_index: int | None) -> str | None:
    if name in _registered_fonts:
        return name
    if not path.exists():
        return None
    try:
        pdfmetrics.registerFont(
            TTFont(name, str(path), subfontIndex=subfont_index)
        )
    except Exception:
        return None
    _registered_fonts.add(name)
    return name


def _resolve_fonts() -> tuple[str, str]:
    """注册可用的中文字体，返回（常规字体名，粗体字体名）。"""
    normal = bold = None
    for path, name, index in _FONT_CANDIDATES:
        if "Bold" in name:
            bold = bold or _try_register(path, name, index)
        else:
            normal = normal or _try_register(path, name, index)
    if normal is None:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(_FALLBACK_FONT))
        except Exception:
            pass
        return _FALLBACK_FONT, _FALLBACK_FONT
    return normal, bold or normal


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline_markup(text: str) -> str:
    escaped = _escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _build_styles(normal: str, bold: str) -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFontFamily(
        normal, normal=normal, bold=bold, italic=normal, boldItalic=bold
    )
    return {
        "header_title": ParagraphStyle(
            "header_title", fontName=bold, fontSize=20, leading=27,
            textColor=colors.white, spaceAfter=2,
        ),
        "header_meta": ParagraphStyle(
            "header_meta", fontName=normal, fontSize=9.5, leading=14,
            textColor=colors.HexColor("#dbe7fb"),
        ),
        "chip": ParagraphStyle(
            "chip", fontName=bold, fontSize=11, leading=15,
            textColor=colors.white,
        ),
        "question": ParagraphStyle(
            "question", fontName=bold, fontSize=12.5, leading=19,
            textColor=_TEXT,
        ),
        "body": ParagraphStyle(
            "body", fontName=normal, fontSize=11.5, leading=19.5,
            textColor=_BODY, spaceAfter=5, alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=normal, fontSize=11.5, leading=19,
            leftIndent=16, bulletIndent=2, bulletColor=_ACCENT,
            bulletFontSize=10, spaceAfter=4,
        ),
        "numbered": ParagraphStyle(
            "numbered", fontName=normal, fontSize=11.5, leading=19,
            leftIndent=26, bulletIndent=2, bulletColor=_ACCENT,
            bulletFontSize=10, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1", fontName=bold, fontSize=16, leading=22,
            textColor=_TEXT, spaceBefore=8, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=bold, fontSize=14, leading=20,
            textColor=_TEXT, spaceBefore=7, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "h3", fontName=bold, fontSize=12.5, leading=18,
            textColor=_ACCENT, spaceBefore=6, spaceAfter=3,
        ),
    }


def _render_answer(text: str, styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    list_buffer: list[str] = []

    def flush_list() -> None:
        for item in list_buffer:
            story.append(Paragraph(item, styles["bullet"], bulletText="\u2022"))
        list_buffer.clear()

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_list()
            story.append(Spacer(1, 3 * mm))
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush_list()
            level = min(len(heading.group(1)), 3)
            story.append(Paragraph(_inline_markup(heading.group(2)), styles[f"h{level}"]))
            continue
        bullet = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if bullet:
            list_buffer.append(_inline_markup(bullet.group(1)))
            continue
        numbered = re.match(
            r"^\s*((?:\d+|[一二三四五六七八九十]+)[、.．)]"
            r"|[（(][一二三四五六七八九十\d]+[）)])\s*(.*)$",
            line,
        )
        if numbered:
            flush_list()
            story.append(
                Paragraph(
                    _inline_markup(numbered.group(2)),
                    styles["numbered"],
                    bulletText=numbered.group(1),
                )
            )
            continue
        flush_list()
        story.append(Paragraph(_inline_markup(line), styles["body"]))
    flush_list()
    return story


def _make_footer(normal: str):
    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(_BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFont(normal, 8.5)
        canvas.setFillColor(_MUTED)
        canvas.drawString(18 * mm, 9 * mm, "政策知识问答记录")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    return footer


def _header_band(story: list, styles: dict[str, ParagraphStyle], doc, subtitle: str) -> None:
    header = Table(
        [
            [Paragraph("政策知识问答记录", styles["header_title"])],
            [Paragraph(subtitle, styles["header_meta"])],
        ],
        colWidths=[doc.width],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(header)
    story.append(Spacer(1, 6 * mm))


def _label_chip(text: str, styles: dict[str, ParagraphStyle]) -> Table:
    chip = Table(
        [[Paragraph(text, styles["chip"])]],
        colWidths=[46],
        hAlign="LEFT",
    )
    chip.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _ACCENT),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return chip


def _question_card(
    text: str,
    styles: dict[str, ParagraphStyle],
    doc,
    label: str = "问题",
) -> list:
    story: list = [_label_chip(label, styles), Spacer(1, 3 * mm)]
    box = Table(
        [[Paragraph(_escape(text), styles["question"])]],
        colWidths=[doc.width],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _QUESTION_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 3, _ACCENT),
                ("BOX", (0, 0), (-1, -1), 0.8, _QUESTION_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(box)
    story.append(Spacer(1, 4 * mm))
    return story


def _answer_section(answer: str, styles: dict[str, ParagraphStyle]) -> list:
    story: list = [_label_chip("回答", styles), Spacer(1, 3 * mm)]
    story.extend(_render_answer(answer, styles))
    return story


def _drop_unknown_turns(turns: list[dict[str, object]]) -> list[dict[str, object]]:
    """过滤委婉未答的回答及其对应的用户提问，避免导出无信息量的记录。"""
    filtered: list[dict[str, object]] = []
    pending_user: dict[str, object] | None = None
    for turn in turns:
        if turn["role"] == "user":
            pending_user = turn
            continue
        if turn.get("unknown"):
            pending_user = None
            continue
        if pending_user is not None:
            filtered.append(pending_user)
            pending_user = None
        filtered.append(turn)
    if pending_user is not None:
        filtered.append(pending_user)
    return filtered


def _make_document(buffer: BytesIO) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="政策知识问答记录",
        author="知识库问答 Agent",
    )


def build_qa_pdf(
    question: str,
    answer: str,
    sources: list[str] | None = None,
    timestamp: datetime | None = None,
) -> bytes:
    """生成包含问题与回答的 A4 PDF（不显示来源），返回 PDF 字节。"""
    normal, bold = _resolve_fonts()
    styles = _build_styles(normal, bold)
    now = timestamp or datetime.now()

    buffer = BytesIO()
    doc = _make_document(buffer)
    story: list = []
    _header_band(story, styles, doc, now.strftime("%Y-%m-%d %H:%M"))
    story.extend(_question_card(question, styles, doc, label="问题"))
    story.extend(_answer_section(answer, styles))

    footer = _make_footer(normal)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def build_conversation_pdf(
    messages: list[dict],
    timestamp: datetime | None = None,
) -> bytes:
    """生成包含整段对话记录的 A4 PDF（不显示来源）。"""
    turns: list[dict[str, object]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "") or "").strip().lower()
        content = str(item.get("content", "") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        turns.append(
            {
                "role": role,
                "content": content,
                "unknown": bool(item.get("unknown")),
            }
        )
    turns = _drop_unknown_turns(turns)
    if not turns:
        raise ValueError("没有可导出的对话内容")

    normal, bold = _resolve_fonts()
    styles = _build_styles(normal, bold)
    now = timestamp or datetime.now()

    buffer = BytesIO()
    doc = _make_document(buffer)
    story: list = []
    _header_band(
        story,
        styles,
        doc,
        f"{now.strftime('%Y-%m-%d %H:%M')} · 共 {len(turns)} 轮对话",
    )

    for index, turn in enumerate(turns):
        role = turn["role"]
        content = str(turn["content"])
        if role == "user":
            story.extend(_question_card(content, styles, doc, label="用户"))
            story.append(Spacer(1, 5 * mm))
            continue
        story.extend(_answer_section(content, styles))
        if index != len(turns) - 1:
            story.append(Spacer(1, 5 * mm))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=_BORDER,
                    spaceBefore=0,
                    spaceAfter=0,
                )
            )
            story.append(Spacer(1, 6 * mm))

    footer = _make_footer(normal)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
