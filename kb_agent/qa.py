"""基于检索的严格知识库问答。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import jieba
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from .config import get_env, get_float_env, get_int_env, vectorstore_dir
from .ingest import get_embeddings
from .web_search import WebResult, search_web

logger = logging.getLogger(__name__)

UNKNOWN_ANSWER = "我还不懂这些，你可以换个问法再问一次。"
UNKNOWN_WEB_OFFER = "需要我帮你联网搜索相关的答案吗？"
UNKNOWN_ANSWER_WITH_OFFER = f"{UNKNOWN_ANSWER}{UNKNOWN_WEB_OFFER}"
WEB_SEARCH_FAILED = "我尝试了联网搜索，但没有找到可靠的相关内容。你可以换个问法再问一次。"
WEB_DISCLAIMER = "> 以下内容来自联网搜索，并非官方文件内容，仅供参考，请谨慎辨别。"
MAX_HISTORY_TURNS = 8
_LIST_QUESTION_WORDS = ("有哪些", "列举", "包括", "哪些", "几个", "多少", "什么")
_SUMMARY_QUESTION_WORDS = (
    "总结",
    "梳理",
    "汇总",
    "概括",
    "概述",
    "归纳",
    "说明",
    "介绍",
    "全部",
    "完整",
    "清单",
    "一览",
    "明细",
)

_REPORT_QUESTION_WORDS = (
    "解读",
    "报告",
    "汇报",
    "提纲",
    "白皮书",
    "一页纸",
    "主要内容",
    "内容概要",
    "要点",
    "概况",
)

TIER_ALIASES = (
    ("其他地区政策", "其他地区政策"),
    ("国家政策", "国家政策"),
    ("国家级", "国家政策"),
    ("国家层面", "国家政策"),
    ("全国", "国家政策"),
    ("省级政策", "省级政策"),
    ("省级", "省级政策"),
    ("全省", "省级政策"),
    ("市级政策", "市级政策"),
    ("市级", "市级政策"),
    ("全市", "市级政策"),
    ("园区政策", "园区政策"),
    ("其他地区", "其他地区政策"),
)

# 暂存于非对应层级目录的外省/外地文件涉及的主要地区名；
# 问题同时出现层级词和这些地区名时，跨层级检索，避免层级过滤漏掉文件。
REGION_KEYWORDS = (
    "山东",
    "济南",
    "青岛",
    "厦门",
    "宁波",
    "深圳",
    "萧山",
    "杭州",
    "上海",
    "北京",
    "朝阳区",
    "北京经济技术开发区",
    "经开区",
)

# 默认语境以苏州/江苏为主；命中这些外地标记的文档默认不进入上下文，
# 除非用户明确点名外地地区、要求横向比较，或问题属于“语料券”等外地专属主题。
OTHER_REGION_MARKERS = (
    "山东",
    "济南",
    "青岛",
    "厦门",
    "宁波",
    "深圳",
    "萧山",
    "杭州",
    "上海",
    "北京",
    "朝阳区",
    "北京经济技术开发区",
    "石景山",
    "西城",
    "普陀区",
    "自贸试验区",
    "四川",
    "成都",
    "贵阳",
    "重庆",
    "大连",
    "武汉",
    "广州",
    "浙江",
    "上城区",
    "滨江",
    "深政数规",
    "甬数管",
    "厦府办规",
    "济数字",
    "萧数据",
    "沪经信智",
    "沪",
)

_COMPARISON_WORDS = ("对比", "比较", "不同", "差异", "区别", "相比", "横向", "对照")

# “其他地区政策”目录里混有少量江苏/苏州文档，用本地地区词保住它们。
LOCAL_REGION_MARKERS = (
    "江苏",
    "苏州",
    "苏州工业园区",
    "相城区",
    "相城",
    "无锡",
    "南通",
    "南京",
    "玄武",
    "昆山",
    "常熟",
    "张家港",
    "太仓",
    "吴江",
    "姑苏",
    "虎丘",
    "吴中",
    "徐州",
    "常州",
    "连云港",
    "淮安",
    "盐城",
    "扬州",
    "镇江",
    "泰州",
    "宿迁",
    "苏南",
    "苏中",
    "苏北",
)


def _is_other_region_source(source: str) -> bool:
    """根据文档来源名判断是否属于外地（非苏州/江苏默认语境）政策。"""
    return any(marker in str(source) for marker in OTHER_REGION_MARKERS)


def _doc_mentions_other_region(doc) -> bool:
    return any(marker in (doc.page_content or "") for marker in OTHER_REGION_MARKERS)


def _doc_mentions_local_region(doc) -> bool:
    return any(marker in (doc.page_content or "") for marker in LOCAL_REGION_MARKERS)


def _local_source_names(documents: list[Document]) -> set[str]:
    """默认语境下可用的文档来源：排除外地文档，但保留混放目录里的江苏/苏州文档。"""
    local: set[str] = set()
    other: set[str] = set()
    pending: dict[str, tuple[bool, bool]] = {}
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        if not source or source in other or source in local:
            continue
        if _is_other_region_source(source):
            other.add(source)
            continue
        if str(doc.metadata.get("tier", "")) != "其他地区政策":
            local.add(source)
            continue
        has_other, has_local = pending.get(source, (False, False))
        pending[source] = (
            has_other or _doc_mentions_other_region(doc),
            has_local or _doc_mentions_local_region(doc),
        )
    for source, (has_other, has_local) in pending.items():
        if not (has_other and not has_local):
            local.add(source)
    return local


def _needs_other_region_docs(question: str) -> bool:
    """问题是否明确涉及外地政策（点名地区、横向比较或语料券主题）。"""
    question = question or ""
    if "语料券" in question:
        return True
    if any(marker in question for marker in OTHER_REGION_MARKERS):
        return True
    if any(word in question for word in ("其他地区", "外地", "外省", "不同地区", "别的地区", "各地")):
        return True
    return any(word in question for word in _COMPARISON_WORDS) and any(
        word in question for word in OTHER_REGION_MARKERS + ("其他地区", "外地", "外省", "不同地区", "别的地区", "各地")
    )


def detect_tier(question: str) -> str | None:
    """从问题中识别明确的政策层级；没有明确层级时返回 None（检索全部）。"""
    tier = None
    for alias, detected in TIER_ALIASES:
        if alias in question:
            tier = detected
            break
    if tier is not None and any(region in question for region in REGION_KEYWORDS):
        return None
    return tier


def detect_target_sources(question: str, documents: list[Document]) -> list[str]:
    """从问题中识别书名号内的文件名引用（如《某文件》），返回匹配的 source 列表。"""
    candidates = [
        re.sub(r"\s+", "", cand)
        for cand in re.findall(r"《([^》]+)》", question or "")
    ]
    if not candidates:
        return []
    sources = sorted(
        {
            str(doc.metadata.get("source", ""))
            for doc in documents
            if doc.metadata.get("source")
        }
    )
    matched: list[str] = []
    for source in sources:
        base = re.sub(
            r"\.(docx?|pptx?|pdf|wps|xlsx?|xlsm?|txt|md|csv)$",
            "",
            source,
            flags=re.IGNORECASE,
        )
        base_norm = re.sub(r"\s+", "", base)
        for candidate in candidates:
            if candidate in base_norm or base_norm in candidate:
                matched.append(source)
                break
    return matched


SYSTEM_PROMPT = """你是一个严格依据知识库内容回答问题的助手。
你只能使用下面的“知识库内容”中明确写出的信息来回答，禁止编造、推测，也禁止使用知识库之外的知识。
知识库可能把同一份文件拆成多个片段，回答时必须综合同一文件的所有相关片段；如果某个奖项、机构、数字、条款或结论没有明确出现在知识库内容中，不得补充或推测，也不得混入其他文件的无关内容。

对话历史（用户和你的历史问答，只用于理解当前问题的指代与承接；它不是知识库事实，不能作为新事实来源）：
{conversation}

如果用户的问题承接上一轮对话（例如前一轮问“数字孪生是什么？”，这一轮问“园区对它有什么期望吗？”，应理解为“园区对数字孪生有什么期望吗？”），请先把“它/其/这个/该政策/上述内容”等指代词替换成历史中明确提到的具体对象（如政策名称、文件名称、主题词），再作答；不要凭空猜测指代对象，也不要复述上一轮的整段回答。
对话历史中助手的历史回答均来自知识库；当当前问题只是承接上一轮时，可以直接沿用其中已经给出的明确信息作答，但不得新增知识库之外的内容；如果历史回答本身是委婉未答或历史中没有明确信息，请回复“{unknown}”。
请保持自然、连贯的对话语气，像正常聊天一样承接上下文，但始终只依据知识库内容作答。
回答的措辞要面向普通用户：不要出现“知识库”“库内”“知识库内容”等后台术语；需要说明依据时，用“根据我的理解”“从已有资料看”“据了解”等自然、委婉的表达。

知识库内容：
{context}

如果知识库内容足够回答用户问题，请直接、精准地用中文作答，可以引用原文中的条款或数字。
回答长度要适中：既不要冗长地逐文件铺开，也不要过于简洁；先给出核心结论或要点，再按需补充关键细节。
如果问题要求总结、梳理、汇总或介绍某类内容，请概括核心要点（通常 3~7 条），保留最重要的编号条目和关键数字；不要逐文件复制目标、任务和全部细项。
如果用户明确要求列举具体条目（如“有哪些”“列举”），请把主要条目逐条列出，每条保持简短，不要展开无关细节；只有用户明确要求“完整列出/全部条目”时才需要把全部条目列全。
回答总结/梳理或涉及多份文件的问题时，每个要点后面用（《文件名》）标注其来源文件。
当知识库内容较长、需要大段作答时，先概括核心要点再按需展开，尽量简练；概括只能依据知识库内容，不得添加知识库之外的任何信息，不得臆测、扩写或补全。
涉及多份文件时，先归纳共同的核心要点；确需区分文件时再按“《文件名》”分组，且每个文件只保留关键要点；禁止把不同文件的条目混成一份清单，也禁止把一份文件的内容挪到另一份文件下。
不要在回答中评价知识库是否完整（例如“知识库中仅保留了部分标题和片段”“具体细项未列出”）；知识库内容不足时直接回复“{unknown}”，不要补充推测，也不要描述知识库缺失了什么，更不要要求用户上传或补充文档。
如果是判断类问题（如“是否属于”“是不是”），请先明确回答“是”或“不是”，再引用知识库原文说明依据。
如果知识库内容不足以明确回答，请只回复“{unknown}”。
"""

HUMAN_PROMPT = "用户问题：{question}"

GROUNDING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)

REPORT_SYSTEM_PROMPT = """你是一个严格依据知识库内容撰写政策解读/研究报告的助手。
你只能使用下面的“知识库内容”中明确写出的信息，禁止编造、推测，也禁止使用知识库之外的知识、常识或行业信息来补全报告。
知识库可能把同一份文件拆成多个片段，撰写时必须综合同一文件的所有相关片段；如果某个奖项、机构、数字、条款或结论没有明确出现在知识库内容中，不得补充或推测，也不得混入其他文件的无关内容。

对话历史（仅用于理解当前问题的指代与承接；不能作为报告的事实来源）：
{conversation}

知识库内容：
{context}

请把报告组织为“结构化解读/报告”，按下面的模板组织小节：
1. 政策名称与基本信息（名称、发文机关、文号、发布时间、施行时间等，只写知识库中明确出现的）
2. 背景与总体目标
3. 适用对象与适用范围
4. 主要内容与重点任务
5. 支持措施（资金、奖补、优惠等，如有）
6. 申报要求与流程（如有）
7. 实施期限与保障措施
8. 依据文件与来源

写作要求：
- 每个小节只能填写知识库内容中明确写出的信息；没有知识库依据的小节必须直接省略，不要编造内容，不要使用“知识库未提及/未列出/缺失”等评价知识库完整性的表述，也不要为了凑齐模板而补内容。
- 报告措辞面向用户：不要出现“知识库”“库内”“知识库内容”等后台术语；需要说明依据时，用“根据我的理解”“从已有资料看”“据了解”等自然、委婉的表达。
- 每个要点后面用（《文件名》）标注其来源文件；涉及多份文件时，禁止把不同文件的内容混在一起。
- 先概括核心要点（通常 3~7 条），保留关键编号和数字；不要逐文件铺开全部细项，也不要过于简洁。
- 报告中的每一句话都必须能在知识库内容中找到对应原文；不得扩写、不得推测、不得添加知识库之外的任何信息。
- 如果知识库内容不足以支撑报告的任何小节，只回复“{unknown}”，不要输出空报告，也不要要求用户上传或补充文档。
"""

REPORT_GROUNDING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", REPORT_SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)

EXPAND_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识库检索助手。请针对用户问题生成 3 个不同的检索查询，"
            "覆盖同义词、章节关键词和原文用词。每行只输出一个查询，不要编号，不要解释。",
        ),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识库检索助手。请结合对话历史，把用户最新问题改写成一个独立、"
            "完整的检索查询，保留必要的文件名、地区名、层级词和关键词。"
            "如果最新问题包含指代词（如“它”“它们”“其”“这个”“该政策”“该文件”“上述内容”等），"
            "必须先用对话历史中明确提到的具体对象（如政策名称、文件名称、主题词）替换指代词，"
            "改写结果中不得残留任何指代词。"
            "只输出一行查询，不要编号，不要解释。",
        ),
        ("human", "对话历史：\n{history}\n\n用户最新问题：{question}"),
    ]
)

WEB_SEARCH_SYSTEM_PROMPT = """你是一个联网搜索整理助手。
用户资料中没有相关内容，因此你只能依据下面“搜索结果”中明确写出的公开网页信息回答，禁止编造、推测或使用搜索结果之外的知识。
回答要求：
- 中文回答，面向普通用户，先给出核心结论或要点，再按需补充关键细节；不要逐条复制所有搜索结果，也不要过于简洁。
- 需要引用时，在相关句子后附上 Markdown 链接：[标题](链接)，方便用户核对。
- 如果搜索结果不足以回答问题，只回复“{unknown}”，不要评价资料是否完整，也不要要求用户补充资料。

搜索结果：
{results}
"""

WEB_SEARCH_HUMAN_PROMPT = "用户问题：{question}"

WEB_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [("system", WEB_SEARCH_SYSTEM_PROMPT), ("human", WEB_SEARCH_HUMAN_PROMPT)]
)


@dataclass
class AnswerResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    unknown: bool = False
    web: bool = False


def get_llm() -> ChatOpenAI:
    if not get_env("OPENAI_API_KEY"):
        raise RuntimeError("未配置 OPENAI_API_KEY，请先复制 .env.example 为 .env 并填写。")
    _base_url = get_env("OPENAI_BASE_URL") or None
    _model = get_env("LLM_MODEL", "gpt-4o-mini")
    _key = get_env("OPENAI_API_KEY")
    logger.warning(
        "[LLM-DEBUG] base_url=%r | model=%r | key=%s...%s | key_len=%d",
        _base_url,
        _model,
        _key[:6],
        _key[-4:],
        len(_key),
    )
    return ChatOpenAI(
        model=_model,
        temperature=get_float_env("LLM_TEMPERATURE", 0.0),
        api_key=_key,
        base_url=_base_url,
    )


def load_vectorstore(index_directory: Path | None = None, embedding=None) -> FAISS:
    index_dir = Path(index_directory) if index_directory else vectorstore_dir()
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError("向量库不存在，请先运行 python ingest_pdfs.py。")
    embedder = embedding or get_embeddings()
    return FAISS.load_local(
        str(index_dir), embedder, allow_dangerous_deserialization=True
    )


def _zh_tokenizer(text: str) -> list[str]:
    return [token.strip() for token in jieba.cut(text) if token.strip()]


def _all_documents(vectorstore: FAISS) -> list[Document]:
    try:
        return list(vectorstore.docstore._dict.values())
    except Exception:
        logger.warning("无法读取向量库中的文档列表，关键词检索将不可用。")
        return []


class GroundedRetriever:
    """混合检索器：BM25 关键词 + 向量相似度，并按相似度阈值过滤。"""

    def __init__(
        self,
        ensemble: EnsembleRetriever,
        vectorstore: FAISS | None,
        score_threshold: float,
        tier: str | None = None,
        top_k: int = 4,
        target_sources: set[str] | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.vectorstore = vectorstore
        self.score_threshold = score_threshold
        self.tier = tier
        self.target_sources = set(target_sources) if target_sources else None
        self.all_documents = _all_documents(vectorstore) if vectorstore is not None else []
        self.scope_documents = self.all_documents
        if self.tier is not None or self.target_sources:
            self.scope_documents = [
                doc
                for doc in self.all_documents
                if (
                    self.tier is None
                    or str(doc.metadata.get("tier", "")) == self.tier
                )
                and (
                    not self.target_sources
                    or str(doc.metadata.get("source", "")) in self.target_sources
                )
            ]
        self.keyword = (
            BM25Retriever.from_documents(
                self.scope_documents, k=top_k, tokenizer=_zh_tokenizer
            )
            if self.scope_documents
            else None
        )

    def _section_hits(self, question: str) -> list[Document]:
        """按章节路径命中打分排序：命中的词越长越具体，权重越高。"""
        stopwords = {"的", "是", "吗", "哪些", "什么", "多少", "几个", "有", "中", "了", "下", "为", "和", "与", "及"}
        tokens = [
            token
            for token in _zh_tokenizer(question)
            if len(token) >= 2 and token not in stopwords
        ]
        if not tokens:
            return []
        scores: dict[int, tuple[int, Document]] = {}
        pool = self.scope_documents
        for token in tokens:
            weight = len(token) * len(token)
            for doc in pool:
                path = str(doc.metadata.get("section_path", ""))
                if token in path:
                    score, _ = scores.get(id(doc), (0, doc))
                    scores[id(doc)] = (score + weight, doc)
        ordered = sorted(scores.values(), key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ordered]

    def _expand_by_section(self, question: str, documents: list[Document]) -> list[Document]:
        matched = {id(doc) for doc in documents}
        extra = [doc for doc in self._section_hits(question) if id(doc) not in matched]
        return documents + extra

    def _structure_hits(self, question: str) -> list[Document]:
        return self._section_hits(question)

    def _invoke_scoped(self, question: str) -> list[Document]:
        """指定层级/来源时：范围内 BM25 + 全库向量排序后过滤到该范围。"""
        documents = self.keyword.invoke(question) if self.keyword else []
        dense: list[Document] = []
        if self.vectorstore is not None:
            total = len(self.all_documents)
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=max(total, 50)
            )
            scope_ids = {id(doc) for doc in self.scope_documents}
            dense = [
                doc
                for doc, score in scored
                if id(doc) in scope_ids
                and (self.score_threshold <= 0 or score >= self.score_threshold)
            ]
        seen: dict[int, Document] = {}
        for doc in dense:
            seen.setdefault(id(doc), doc)
        for doc in documents:
            if self.score_threshold <= 0 or id(doc) in seen:
                seen.setdefault(id(doc), doc)
        return list(seen.values())

    def invoke(self, question: str) -> list[Document]:
        limit = get_int_env("RETRIEVE_LIMIT", 24)
        if self.tier is not None or self.target_sources:
            documents = self._invoke_scoped(question)
        else:
            documents = self.ensemble.invoke(question)
            if self.score_threshold > 0 and self.vectorstore is not None:
                total = len(self.all_documents)
                scored = self.vectorstore.similarity_search_with_relevance_scores(
                    question, k=min(total, 50)
                )
                allowed = {
                    id(doc) for doc, score in scored if score >= self.score_threshold
                }
                documents = [doc for doc in documents if id(doc) in allowed]
        return self._merge_results(question, documents, limit)

    def _merge_results(
        self,
        question: str,
        documents: list[Document],
        limit: int,
    ) -> list[Document]:
        """同源小文件整文件带入，普通命中优先，章节结构命中用于补充。"""
        structure = self._structure_hits(question)
        merged: list[Document] = []
        seen: set[int] = set()

        by_source: dict[str, list[Document]] = {}
        for doc in self.all_documents:
            by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)

        def add(doc: Document) -> None:
            if id(doc) not in seen and len(merged) < limit:
                merged.append(doc)
                seen.add(id(doc))

        # 命中的小文件（如反馈类短文档）整文件带入，保证同文件上下文完整。
        for doc in documents:
            source = str(doc.metadata.get("source", ""))
            extras = by_source.get(source, [])
            if extras and len(extras) <= 8:
                for extra in extras:
                    add(extra)
            else:
                add(doc)
        # 普通检索剩余片段。
        for doc in documents:
            add(doc)
        # 章节结构命中补漏。
        for doc in structure:
            add(doc)
        return merged

    def invoke_multi(
        self,
        queries: list[str],
        limit: int | None = None,
        relax_threshold: bool = False,
    ) -> list[Document]:
        limit = limit or get_int_env("RETRIEVE_LIMIT", 24)
        if self.tier is not None or self.target_sources:
            per_query = [self._invoke_scoped(query) for query in queries]
        else:
            per_query = [self.ensemble.invoke(query) for query in queries]

        seen: set[int] = set()
        documents: list[Document] = []
        max_len = max((len(docs) for docs in per_query), default=0)
        for round_index in range(max_len):
            for docs in per_query:
                if round_index >= len(docs):
                    continue
                doc = docs[round_index]
                if id(doc) not in seen:
                    seen.add(id(doc))
                    documents.append(doc)
                if len(documents) >= limit:
                    break
            if len(documents) >= limit:
                break

        if (
            not relax_threshold
            and self.score_threshold > 0
            and self.vectorstore is not None
        ):
            total = len(self.all_documents)
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                queries[0], k=min(total, 50)
            )
            allowed = {id(doc) for doc, score in scored if score >= self.score_threshold}
            documents = [doc for doc in documents if id(doc) in allowed]

        return self._merge_results(queries[0], documents, limit)

def build_retriever(
    vectorstore: VectorStore,
    top_k: int | None = None,
    score_threshold: float | None = None,
    tier: str | None = None,
    target_sources: list[str] | None = None,
):
    k = top_k if top_k is not None else get_int_env("TOP_K", 4)
    threshold = (
        score_threshold
        if score_threshold is not None
        else get_float_env("SCORE_THRESHOLD", 0.05)
    )
    documents = _all_documents(vectorstore)
    target_set = set(target_sources or [])
    if tier:
        pool = [doc for doc in documents if str(doc.metadata.get("tier", "")) == tier]
    elif target_set:
        pool = [doc for doc in documents if str(doc.metadata.get("source", "")) in target_set]
    else:
        pool = documents
    keyword = (
        BM25Retriever.from_documents(pool, k=k, tokenizer=_zh_tokenizer)
        if pool
        else None
    )

    if tier or target_set:
        return GroundedRetriever(
            None,
            vectorstore,
            threshold,
            tier=tier,
            top_k=k,
            target_sources=target_set,
        )

    if keyword is None:
        return GroundedRetriever(None, vectorstore, threshold)

    if threshold and threshold > 0:
        dense = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": threshold},
        )
    else:
        dense = vectorstore.as_retriever(search_kwargs={"k": k})

    ensemble = EnsembleRetriever(
        retrievers=[keyword, dense],
        weights=[0.5, 0.5],
    )
    return GroundedRetriever(ensemble, vectorstore, threshold)


def _expand_same_source(
    documents: list[Document],
    vectorstore: VectorStore | None,
    limit: int,
) -> list[Document]:
    """把已命中文件的小型片段补齐，保证同一文件的上下文完整（如修改建议一~四）。"""
    if vectorstore is None:
        return documents[:limit]
    all_docs = _all_documents(vectorstore)
    by_source: dict[str, list[Document]] = {}
    for doc in all_docs:
        by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)
    seen = {id(doc) for doc in documents}
    merged = list(documents)
    for doc in documents:
        if len(merged) >= limit:
            break
        extras = by_source.get(str(doc.metadata.get("source", "")), [])
        if not extras or len(extras) > 8:
            continue
        for extra in extras:
            if id(extra) not in seen and len(merged) < limit:
                merged.append(extra)
                seen.add(id(extra))
    return merged[:limit]


def _complete_source_context(
    documents: list[Document],
    all_documents: list[Document],
    limit: int,
    full_source_max: int | None = None,
) -> list[Document]:
    """总结/列举类问题：补齐命中来源的相关片段，并按来源、页码排序，避免条目缺失或跨文件混淆。"""
    if not documents:
        return []
    if full_source_max is None:
        full_source_max = get_int_env("SUMMARY_FULL_SOURCE_MAX", 120)
    by_source: dict[str, list[Document]] = {}
    for doc in all_documents:
        by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)

    merged: list[Document] = []
    seen: set[int] = set()

    def sort_key(doc: Document):
        page = doc.metadata.get("page")
        return (
            page is None,
            page if isinstance(page, int) else 0,
            str(doc.metadata.get("section_path", "")),
            str(doc.metadata.get("section", "")),
            doc.page_content,
        )

    source_order: dict[str, int] = {}
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        source_order.setdefault(source, len(source_order))

    for source in source_order:
        extras = by_source.get(source, [])
        if not extras:
            continue
        if len(extras) > full_source_max:
            hit_sections = {
                str(doc.metadata.get("section_path", "")).strip()
                for doc in documents
                if str(doc.metadata.get("source", "")) == source
                and str(doc.metadata.get("section_path", "")).strip()
            }
            if hit_sections:
                extras = [
                    extra
                    for extra in extras
                    if any(
                        str(extra.metadata.get("section_path", "")).strip().startswith(hit)
                        or hit.startswith(str(extra.metadata.get("section_path", "")).strip())
                        for hit in hit_sections
                    )
                ]
        for extra in sorted(extras, key=sort_key):
            if id(extra) in seen or len(merged) >= limit:
                continue
            merged.append(extra)
            seen.add(id(extra))
        if len(merged) >= limit:
            break

    if not merged:
        return documents[:limit]
    return merged


def _format_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section", "")
        tier = doc.metadata.get("tier", "")
        label = f"{source}（第 {page} 页）" if page is not None else source
        tags = [tag for tag in (tier, section) if tag]
        if tags:
            label += "【" + "】【".join(tags) + "】"
        blocks.append(f"[来源: {label}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def _format_sources(documents: list[Document]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "未知来源")
        source_str = str(source)
        if source_str in seen:
            continue
        seen.add(source_str)
        sources.append(source_str)
    return sources


def _expand_queries(question: str, llm: BaseChatModel) -> list[str]:
    try:
        messages = EXPAND_PROMPT.format_messages(question=question)
        response = llm.invoke(messages)
        text = getattr(response, "content", str(response))
        queries: list[str] = []
        for line in text.splitlines():
            line = re.sub(r"^\s*[\d一二三四五六七八九十]+[、.．\)）]\s*", "", line).strip()
            if line and len(line) <= 80:
                queries.append(line)
        return queries[:4]
    except Exception:
        logger.warning("查询扩展失败，改用原问题检索。", exc_info=True)
        return []


def _is_list_question(question: str) -> bool:
    return any(word in question for word in _LIST_QUESTION_WORDS)


def _is_summary_question(question: str) -> bool:
    return _is_list_question(question) or any(
        word in question for word in _SUMMARY_QUESTION_WORDS
    )


def _is_summary_style_question(question: str) -> bool:
    """是否明确要求总结/梳理/汇总（不包含普通“有哪些/多少”类问题）。"""
    return any(word in question for word in _SUMMARY_QUESTION_WORDS)


def _is_report_question(question: str) -> bool:
    """是否要求生成结构化解读/报告（解读、报告、汇报、要点、主要内容等）。"""
    return any(word in question for word in _REPORT_QUESTION_WORDS)


def _normalize_history(history) -> list[dict[str, object]]:
    """把调用方传入的历史消息规范为 role/content/sources 字典列表。"""
    if not history:
        return []
    turns: list[dict[str, object]] = []
    for item in history:
        role = ""
        content = ""
        sources: list[str] = []
        if isinstance(item, dict):
            role = str(item.get("role", "") or "").strip().lower()
            content = str(item.get("content", "") or "").strip()
            raw_sources = item.get("sources") or []
            if isinstance(raw_sources, (list, tuple)):
                sources = [str(source) for source in raw_sources if str(source).strip()]
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            role = str(item[0]).strip().lower()
            content = str(item[1]).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        turns.append({"role": role, "content": content, "sources": sources})
    return turns[-MAX_HISTORY_TURNS * 2 :]


def _format_history(history) -> str:
    """把历史对话格式化成提示词里的可读文本。"""
    turns = _normalize_history(history)
    if not turns:
        return "（无）"
    lines: list[str] = []
    for turn in turns:
        label = "用户" if turn["role"] == "user" else "助手"
        lines.append(f"{label}：{turn['content']}")
    return "\n".join(lines)


def _history_sources(history) -> list[str]:
    """收集历史回答中已有的来源，用于承接问题未命中新内容时展示。"""
    seen: set[str] = set()
    sources: list[str] = []
    for turn in history:
        if turn["role"] != "assistant":
            continue
        for source in turn.get("sources") or []:
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)
    return sources


def _rewrite_question(question: str, history, llm: BaseChatModel) -> str | None:
    """结合历史对话把承接问题改写为独立检索查询；失败时返回 None。"""
    try:
        messages = REWRITE_PROMPT.format_messages(
            history=_format_history(history), question=question
        )
        response = llm.invoke(messages)
        text = getattr(response, "content", str(response)).strip()
        if text and len(text) <= 120:
            return text
    except Exception:
        logger.warning("对话问题改写失败，改用原问题检索。", exc_info=True)
    return None


_REFERENT_REGEX = re.compile(
    r"它(?:们)?|"
    r"这个(?:政策|文件|问题|说法)?|"
    r"该(?:政策|文件|问题|条款|方案)?|"
    r"上述(?:内容|文件|政策|条款)?|"
    r"(?:前面|之前|上面)(?:提到的|说的|所说的)?|"
    r"这些(?:内容|政策|文件|条款|问题)?|"
    r"那个(?:政策|文件|问题)?"
)

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "") if part.strip()]


def _extract_topic(text: str) -> str | None:
    """从问题文本中提取最可能被“它/该政策”指代的话题词。"""
    text = (text or "").strip()
    if not text:
        return None
    quoted = re.findall(r"《([^》]+)》", text)
    if quoted:
        return re.sub(r"\s+", "", quoted[-1])
    sentences = _sentences(text)
    if len(sentences) > 1:
        text = sentences[0]
    cleaned = re.sub(r"[？?！!。，,\.、；;：:\s]+", "", text)
    cleaned = re.sub(
        r"(是什么|有哪些|有什么|如何|怎么样|怎样|怎么回事|吗|呢|么|怎么|怎么看|"
        r"包含什么|包括什么|包括哪些|有什么期望吗|有什么规定吗|有什么要求吗|"
        r"有什么期望|有什么规定|有什么要求|有什么影响|有什么意义|有什么作用|"
        r"情况|内容|意义|作用|奖项|成果)\s*$",
        "",
        cleaned,
    )
    cleaned = cleaned.strip("的是了在")
    if not cleaned:
        return None
    parts = re.split(r"对|关于|在|给|与|和|及|针对|围绕|之后|以后", cleaned)
    parts = [part for part in parts if part and len(part) >= 2]
    if not parts:
        return None
    return max(parts, key=len)


def _extract_local_topic(question: str) -> str | None:
    """同一句话抛出多个问题时，取指代句之前最近的非指代句话题。"""
    last_topic: str | None = None
    for sentence in _sentences(question):
        if _REFERENT_REGEX.search(sentence):
            return last_topic
        topic = _extract_topic(sentence)
        if topic:
            last_topic = topic
    return None


def _history_topic(history) -> str | None:
    """从最近的历史用户问题中找话题；本身含指代词的轮次跳过，继续向前找。"""
    for turn in reversed(history):
        if not (isinstance(turn, dict) and turn.get("role") == "user"):
            continue
        content = str(turn.get("content", ""))
        if _REFERENT_REGEX.search(content):
            continue
        topic = _extract_topic(content)
        if topic:
            return topic
    return None


def _resolve_referents(question: str, history) -> str:
    """把承接问题里的指代词替换成具体话题：优先同句前文，其次历史。"""
    question = (question or "").strip()
    if not question or not _REFERENT_REGEX.search(question):
        return question
    topic = _extract_local_topic(question)
    if topic is None:
        topic = _history_topic(history)
    if not topic:
        return question
    return _REFERENT_REGEX.sub(topic, question)


_WEB_SEARCH_EXPLICIT_WORDS = (
    "联网搜索",
    "网上搜索",
    "在线搜索",
    "搜索一下",
    "搜一下",
    "搜索",
    "搜搜",
    "帮我搜",
    "帮我查",
    "查一下",
    "查查",
    "查一查",
    "百度",
    "上网查",
    "联网查",
    "网上查",
    "在线查",
)

_WEB_CONSENT_ONLY = {
    "可以",
    "可以啊",
    "可以的",
    "好",
    "好的",
    "好啊",
    "好的啊",
    "同意",
    "行",
    "行吧",
    "嗯",
    "嗯嗯",
    "好的好的",
    "可以可以",
    "搜索",
    "搜",
    "搜吧",
    "搜索吧",
    "查",
    "查吧",
    "查一下吧",
    "搜一下吧",
    "搜索一下",
    "搜一下",
    "查一下",
    "查查",
    "联网",
    "联网搜索",
    "帮我搜",
    "帮我查",
}

_WEB_SEARCH_STRIP_RE = re.compile(
    r"^(?:请|麻烦|帮我|帮我一下|能帮我|可以帮我|帮我联网|请帮我)?"
    r"(?:联网|网上|在线|上网)?"
    r"(?:搜索一下|搜索|搜一下|搜搜|搜|查一下|查一查|查查|查|"
    r"百度一下|百度|联网搜索|网上搜索|在线搜索|上网查|联网查|网上查)"
    r"(?:相关|一下)?"
    r"(?:的答案|答案|内容|信息|资料|问题)?"
    r"[，。！？!?、\s]*"
)


def _normalize_text(text: str) -> str:
    return re.sub(r"[，。！？!?、；;\s]+", "", text or "")


def _is_web_consent_only(question: str) -> bool:
    """是否只是同意上一轮的联网搜索邀请（不含新的搜索主题）。"""
    q = _normalize_text(question)
    if q in _WEB_CONSENT_ONLY:
        return True
    return (
        any(q.startswith(prefix) for prefix in ("可以", "好的", "好", "同意", "行", "嗯"))
        and len(q) <= 12
    )


def _has_web_search_intent(question: str) -> bool:
    return any(word in question for word in _WEB_SEARCH_EXPLICIT_WORDS)


def _should_web_search(question: str, history) -> bool:
    """用户明确要求联网，或对上一轮未答邀约表示同意时启用联网搜索。"""
    if _has_web_search_intent(question):
        return True
    if not _is_web_consent_only(question):
        return False
    for turn in reversed(history):
        if turn["role"] == "user":
            return False
        if turn["role"] == "assistant":
            content = str(turn.get("content", ""))
            return UNKNOWN_WEB_OFFER in content or _is_unknown_answer(content)
    return False


def _web_search_query(question: str, history) -> str:
    """确定实际联网搜索的关键词：同意时沿用上一个问题，否则用当前问题。"""
    resolved = _resolve_referents(question, history)
    if _is_web_consent_only(question):
        for turn in reversed(history):
            if turn["role"] == "user":
                return str(turn["content"]) or resolved
        return ""
    stripped = _WEB_SEARCH_STRIP_RE.sub("", resolved).strip("，。！？!?、 \t\r\n")
    return stripped or resolved


def _conversation_queries(
    question: str, history, llm: BaseChatModel
) -> list[str]:
    """构造检索查询：指代解析后的独立问题 + 结合历史改写的问题 + 多问句扩展。"""
    resolved = _resolve_referents(question, history)
    queries = [resolved]
    last_user = ""
    for turn in reversed(history):
        if turn["role"] == "user":
            last_user = str(turn["content"])
            break
    if last_user:
        rewritten = _rewrite_question(question, history, llm)
        queries.append(rewritten if rewritten else f"{last_user} {resolved}")
    queries.extend(_expand_queries(resolved, llm))
    return queries


_ANSWER_SANITIZE_RULES = (
    ("根据知识库内容", "根据我的理解"),
    ("依据知识库内容", "根据已有资料"),
    ("根据知识库中的", "从已有资料看"),
    ("依据知识库中的", "根据已有资料中的"),
    ("知识库中", "已有资料中"),
    ("知识库里的", "已有资料里的"),
    ("知识库之外", "已有资料之外"),
    ("知识库内容", "已有资料"),
    ("知识库里", "已有资料里"),
    ("知识库内", "已有资料内"),
    ("知识库", "已有资料"),
)


def _sanitize_answer(answer: str) -> str:
    """把回答里可能出现的后台术语“知识库”替换成面向用户的委婉表达。"""
    text = answer or ""
    placeholders: list[str] = []

    def protect(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"《[^》]+》", protect, text)
    for old, new in _ANSWER_SANITIZE_RULES:
        text = text.replace(old, new)
    for index, quoted in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", quoted)
    return text


def _with_web_offer(answer: str) -> str:
    """在委婉未答文案后追加联网搜索邀约，避免重复追加。"""
    text = (answer or "").strip()
    if UNKNOWN_WEB_OFFER in text:
        return text
    return f"{text}{UNKNOWN_WEB_OFFER}"


def _is_unknown_answer(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text or "")
    base = re.sub(r"\s+", "", UNKNOWN_ANSWER)
    full = re.sub(r"\s+", "", UNKNOWN_ANSWER_WITH_OFFER)
    return normalized in {base, full}


def answer_web_search(
    question: str,
    *,
    llm: BaseChatModel | None = None,
    history=None,
    max_results: int = 5,
) -> AnswerResult:
    """用户同意后联网搜索公开网页，并整理为带免责声明的回答。"""
    history = _normalize_history(history)
    query = _web_search_query(question, history)
    if not query:
        return AnswerResult(answer=WEB_SEARCH_FAILED, unknown=True, web=True)
    results = search_web(query, max_results=max_results)
    if not results:
        logger.info("联网搜索无结果：%s", query)
        return AnswerResult(answer=WEB_SEARCH_FAILED, unknown=True, web=True)
    blocks = [
        f"{index}. 标题：{result.title}\n   链接：{result.url}\n   摘要：{result.snippet}"
        for index, result in enumerate(results, start=1)
    ]
    messages = WEB_SEARCH_PROMPT.format_messages(
        question=query,
        results="\n\n".join(blocks),
        unknown=UNKNOWN_ANSWER_WITH_OFFER,
    )
    model = llm or get_llm()
    try:
        response = model.invoke(messages)
    except Exception as exc:
        logger.warning("联网搜索整理回答失败：%s", exc)
        return AnswerResult(answer=WEB_SEARCH_FAILED, unknown=True, web=True)
    answer = getattr(response, "content", str(response)).strip()
    if not answer or _is_unknown_answer(answer):
        return AnswerResult(answer=WEB_SEARCH_FAILED, unknown=True, web=True)
    return AnswerResult(
        answer=f"{WEB_DISCLAIMER}\n\n{_sanitize_answer(answer)}",
        sources=[],
        unknown=False,
        web=True,
    )


def _answer_from_history(
    question: str, history, llm: BaseChatModel, report_mode: bool = False
) -> AnswerResult:
    """未命中新内容时，尝试仅依据历史对话承接上一轮；报告模式不承接。"""
    if report_mode:
        return AnswerResult(answer=UNKNOWN_ANSWER_WITH_OFFER, unknown=True)
    context = (
        "（当前问题没有命中新的资料内容，只能依据对话历史中已有的回答承接，"
        "不得新增内容。）"
    )
    messages = GROUNDING_PROMPT.format_messages(
        context=context,
        question=question,
        unknown=UNKNOWN_ANSWER_WITH_OFFER,
        conversation=_format_history(history),
    )
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.warning("基于历史对话作答失败：%s", exc)
        return AnswerResult(answer=UNKNOWN_ANSWER_WITH_OFFER, unknown=True)
    answer = getattr(response, "content", str(response)).strip() or UNKNOWN_ANSWER_WITH_OFFER
    if _is_unknown_answer(answer):
        return AnswerResult(answer=UNKNOWN_ANSWER_WITH_OFFER, unknown=True)
    return AnswerResult(answer=_sanitize_answer(answer), sources=_history_sources(history))


def answer_question(
    question: str,
    *,
    vectorstore: VectorStore | None = None,
    retriever=None,
    llm: BaseChatModel | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    index_directory: Path | None = None,
    history=None,
) -> AnswerResult:
    """检索知识库并回答；可带历史对话承接上一轮，检索不到时返回委婉未答。"""
    question = (question or "").strip()
    if not question:
        return AnswerResult(answer=UNKNOWN_ANSWER_WITH_OFFER, unknown=True)
    history = _normalize_history(history)
    if _should_web_search(question, history):
        return answer_web_search(question, llm=llm, history=history)

    report_mode = _is_report_question(question)
    if retriever is None:
        vs = vectorstore or load_vectorstore(index_directory=index_directory)
        effective_top_k = top_k
        summary_mode = _is_summary_question(question) or report_mode
        if summary_mode:
            default_top_k = get_int_env("TOP_K", 10)
            effective_top_k = max(top_k or default_top_k, 60)
        tier = detect_tier(question)
        all_docs = _all_documents(vs)
        target_sources = detect_target_sources(question, all_docs)
        if tier is None and not _needs_other_region_docs(question):
            local_sources = sorted(_local_source_names(all_docs))
            if target_sources:
                local_targets = [source for source in target_sources if source in local_sources]
                target_sources = local_targets or target_sources
            elif local_sources:
                target_sources = local_sources
        retriever = build_retriever(
            vs,
            top_k=effective_top_k,
            score_threshold=score_threshold,
            tier=tier,
            target_sources=target_sources,
        )
        model = llm or get_llm()
        if hasattr(retriever, "invoke_multi"):
            queries = _conversation_queries(question, history, model)
            seed_limit = get_int_env("SUMMARY_SEED_LIMIT", 120) if summary_mode else None
            documents = retriever.invoke_multi(
                queries,
                limit=seed_limit,
                relax_threshold=_is_summary_style_question(question),
            )
        else:
            documents = retriever.invoke(question)
        if summary_mode:
            limit = get_int_env("SUMMARY_RETRIEVE_LIMIT", 200)
            documents = _complete_source_context(documents, _all_documents(vs), limit)
        else:
            documents = _expand_same_source(
                documents,
                vs,
                get_int_env("RETRIEVE_LIMIT", 24),
            )
    else:
        model = llm or get_llm()
        documents = retriever.invoke(question)

    if not documents:
        if history:
            logger.info("未检索到新内容，尝试基于历史对话回答承接问题。")
            return _answer_from_history(question, history, model, report_mode)
        logger.info("未检索到超过阈值的知识库内容，返回委婉的未答说明。")
        return AnswerResult(answer=UNKNOWN_ANSWER_WITH_OFFER, unknown=True)

    context = _format_context(documents)
    prompt = REPORT_GROUNDING_PROMPT if report_mode else GROUNDING_PROMPT
    messages = prompt.format_messages(
        context=context,
        question=question,
        unknown=UNKNOWN_ANSWER_WITH_OFFER,
        conversation=_format_history(history),
    )
    response = model.invoke(messages)
    answer = (
        getattr(response, "content", str(response)).strip()
        or UNKNOWN_ANSWER_WITH_OFFER
    )
    if _is_unknown_answer(answer):
        return AnswerResult(answer=UNKNOWN_ANSWER_WITH_OFFER, unknown=True)
    return AnswerResult(answer=_sanitize_answer(answer), sources=_format_sources(documents))
