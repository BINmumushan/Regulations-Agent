"""联网搜索：抓取公开搜索引擎结果页并解析，无需额外 API Key。"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 8
DEFAULT_MAX_RESULTS = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str = ""


def _clean_text(text: str | None) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch(url: str) -> requests.Response | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response
    except Exception as exc:
        logger.warning("联网搜索请求失败 %s: %s", url, exc)
        return None


def _search_bing_rss(query: str) -> list[WebResult]:
    """Bing RSS 结果最稳定，优先使用。"""
    url = "https://www.bing.com/search?format=rss&q=" + quote_plus(query)
    response = _fetch(url)
    if response is None:
        return []
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return []
    results: list[WebResult] = []
    for item in root.iter("item"):
        title = _clean_text(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        snippet = _clean_text(item.findtext("description"))
        if title and link.startswith(("http://", "https://")):
            results.append(WebResult(title=title, url=link, snippet=snippet))
    return results


def _search_bing_html(query: str) -> list[WebResult]:
    url = "https://cn.bing.com/search?q=" + quote_plus(query) + "&setlang=zh-CN"
    response = _fetch(url)
    if response is None:
        return []
    results: list[WebResult] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', response.text, re.S):
        anchor = re.search(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.S,
        )
        if not anchor:
            continue
        link = html.unescape(anchor.group(1)).strip()
        if not link.startswith(("http://", "https://")):
            continue
        title = _clean_text(anchor.group(2))
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        snippet = _clean_text(snippet_match.group(1)) if snippet_match else ""
        if title:
            results.append(WebResult(title=title, url=link, snippet=snippet))
    return results


def _search_sogou(query: str) -> list[WebResult]:
    url = "https://www.sogou.com/web?query=" + quote_plus(query)
    response = _fetch(url)
    if response is None:
        return []
    results: list[WebResult] = []
    for link, title in re.findall(
        r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        response.text,
        re.S,
    ):
        link = html.unescape(link).strip()
        if not link.startswith(("http://", "https://")):
            continue
        title = _clean_text(title)
        if title:
            results.append(WebResult(title=title, url=link))
    return results


def _search_baidu(query: str) -> list[WebResult]:
    url = "https://www.baidu.com/s?wd=" + quote_plus(query)
    response = _fetch(url)
    if response is None:
        return []
    results: list[WebResult] = []
    pattern = re.compile(
        r'<div[^>]*class="[^"]*c-container[^"]*"[^>]*>'
        r"(.*?)(?=<div[^>]*class=\"[^\"]*c-container|</body>)",
        re.S,
    )
    for block in pattern.findall(response.text):
        anchor = re.search(
            r'<h3[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.S,
        )
        if not anchor:
            continue
        link = html.unescape(anchor.group(1)).strip()
        if not link.startswith(("http://", "https://")):
            continue
        title = _clean_text(anchor.group(2))
        snippet_match = re.search(
            r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>',
            block,
            re.S,
        )
        snippet = _clean_text(snippet_match.group(1)) if snippet_match else ""
        if title:
            results.append(WebResult(title=title, url=link, snippet=snippet))
    return results


def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[WebResult]:
    """依次尝试多个公开搜索引擎，返回去重后的标题/链接/摘要。"""
    query = (query or "").strip()
    if not query:
        return []
    engines = (
        _search_bing_rss,
        _search_bing_html,
        _search_sogou,
        _search_baidu,
    )
    seen: set[tuple[str, str]] = set()
    results: list[WebResult] = []
    for engine in engines:
        try:
            for result in engine(query):
                key = (result.url, result.title)
                if key in seen:
                    continue
                seen.add(key)
                results.append(result)
                if len(results) >= max_results:
                    return results
        except Exception as exc:
            logger.warning("联网搜索引擎 %s 失败: %s", engine.__name__, exc)
    return results
