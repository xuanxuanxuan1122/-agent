"""Chunk raw documents into RAG-ready chunk JSON files.

This is the preprocessing stage of the pipeline:
1. read raw inputs
2. clean and split them into chunks
3. assign stable metadata such as chunk_uid
4. write *.chunks.json files for embedding and retrieval
"""

import hashlib
import html
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .embedding_qdrant import QwenLocalEmbeddingModel, collect_runtime_dependency_issues
# =========================
# 1. Config
# =========================
INPUT_PATH = Path(os.getenv("RAG_INPUT_PATH", r"D:\paddlepcr\output_cleaned"))
OUTPUT_DIR = Path(r"D:\pychram\RAG2\rag_chunks_store")
LOCAL_EMBEDDING_MODEL_PATH = os.getenv("QWEN3_EMBEDDING_MODEL_PATH", r"D:\Qwen3-Embedding-4B")
LOCAL_EMBEDDING_DEVICE = os.getenv("QWEN3_EMBEDDING_DEVICE", "auto")
LOCAL_EMBEDDING_DTYPE = os.getenv("QWEN3_EMBEDDING_DTYPE", "float16")
LOCAL_EMBEDDING_ATTN_IMPLEMENTATION = os.getenv("QWEN3_EMBEDDING_ATTN_IMPL", "sdpa")
LOCAL_EMBEDDING_MAX_LENGTH = int(os.getenv("QWEN3_EMBEDDING_MAX_LENGTH", "2048"))

MAX_CHUNK_CHARS = 700
TARGET_CHUNK_CHARS = 480
MIN_CHUNK_CHARS = 160
PARENT_CHUNK_TARGET_CHARS = int(os.getenv("RAG_PARENT_CHUNK_TARGET_CHARS", "1100"))
PARENT_CHUNK_MAX_CHARS = int(os.getenv("RAG_PARENT_CHUNK_MAX_CHARS", "1600"))
PARENT_CHUNK_MIN_CHARS = int(os.getenv("RAG_PARENT_CHUNK_MIN_CHARS", "650"))
PARENT_CHILD_MAX_COUNT = int(os.getenv("RAG_PARENT_CHILD_MAX_COUNT", "4"))

OVERLAP_SENTENCES = 1
SEMANTIC_MERGE_THRESHOLD = 0.88
MERGE_ONLY_SAME_SECTION = True

TEXT_SIM_THRESHOLD = 0.92
EMBED_SIM_THRESHOLD = 0.97
MINHASH_SHINGLE_N = 5

BATCH_SIZE = int(os.getenv("QWEN3_EMBEDDING_BATCH_SIZE", "16"))
REQUEST_INTERVAL = 0.0
MAX_RETRIES = 3
INITIAL_DELAY = 1.0
LEXICAL_MERGE_THRESHOLD = 0.18

ENABLE_PARALLEL_EMBEDDING = False
EMBEDDING_WORKERS = 1
FAST_MODE = False
FORCE_REPROCESS = False
QUALITY_FILTER_EARLY = True
OUTPUT_INCLUDE_RAW_TEXT = False
OUTPUT_INCLUDE_EMBEDDING = False
CLEAN_SOURCE_BEFORE_CHUNKING = True
EMBEDDING_REFINEMENT_MODE = (os.getenv("RAG_EMBEDDING_REFINEMENT_MODE", "off") or "off").strip().lower()
if EMBEDDING_REFINEMENT_MODE not in {"auto", "on", "off"}:
    EMBEDDING_REFINEMENT_MODE = "auto"

OUTPUT_FIELD_WHITELIST = [
    "chunk_uid",
    "doc_id",
    "doc_title",
    "source",
    "section_id",
    "chunk_in_section",
    "chunk_level",
    "source_text_profile",
    "header_path",
    "section_title",
    "page_no",
    "page_label",
    "chunk_type",
    "semantic_role",
    "table_family",
    "table_headers",
    "table_rows",
    "table_row_count",
    "table_row_texts",
    "table_specs",
    "table_spec_texts",
    "normalized_table_text",
    "text",
    "display_text",
    "retrieval_text",
    "embedding_text",
    "embedding_text_clean",
    "summary_1line",
    "summary_consistency_score",
    "text_length",
    "info_density",
    "noise_score",
    "knowledge_unit_type",
    "ocr_noise_ratio",
    "section_kind",
    "page_consistency_flags",
    "answerability_score",
    "quality_flags",
    "quality_score",
    "is_retrieval_eligible",
    "prev_chunk_uid",
    "next_chunk_uid",
    "parent_chunk_uid",
    "parent_chunk_index",
    "parent_chunk_count",
    "child_chunk_uids",
    "child_chunk_count",
    "child_chunk_start_index",
    "child_chunk_end_index",
]

OUTPUT_FIELD_BLACKLIST = {
    "summary",
    "table_summary",
    "row_summary",
    "embedding_summary",
}
SUMMARY_KEY_RE = re.compile(r"summary", re.IGNORECASE)

QUALITY_SCORE_RETRIEVAL_THRESHOLD = float(os.getenv("RAG_QUALITY_SCORE_RETRIEVAL_THRESHOLD", "0.55"))
PARENT_QUALITY_SCORE_RETRIEVAL_THRESHOLD = float(os.getenv("RAG_PARENT_QUALITY_SCORE_RETRIEVAL_THRESHOLD", "0.45"))

LIST_BLOCK_SOFT_LIMIT = 5
TABLE_BLOCK_SOFT_LIMIT = 6
FAQ_QUESTION_SOFT_LIMIT = 120
PIPE_TABLE_CELL_SOFT_LIMIT = 5
PIPE_TABLE_MAX_LINE_LENGTH = 220
PIPE_TABLE_ROW_CHAR_TARGET = 120
SOFT_BOUNDARY_SPLIT_RE = re.compile(r"(?<=[,;:，；：、/])\s+|(?<=\))\s+|(?<=\])\s+")

EMBED_CACHE: Dict[str, List[float]] = {}
EMBED_CACHE_LOCK = threading.Lock()
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_var, None)
LOCAL_EMBEDDER = None


def resolve_text_source_profile(input_file: Path) -> str:
    return "clean_text"


def should_use_embedding_refinement(source_profile: str) -> bool:
    if EMBEDDING_REFINEMENT_MODE == "on":
        return True
    if EMBEDDING_REFINEMENT_MODE == "off":
        return False
    return source_profile != "clean_text"


def empty_clean_stats() -> Dict[str, int]:
    return {
        "input_chars": 0,
        "output_chars": 0,
        "removed_ocr_lines": 0,
        "removed_html_lines": 0,
        "removed_metadata_lines": 0,
        "removed_toc_lines": 0,
        "removed_mojibake_lines": 0,
        "removed_low_value_sections": 0,
        "kept_lines": 0,
    }


def parse_simple_page_number(token: str) -> int | None:
    token = str(token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)

    digit_map = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    unit_map = {"十": 10, "百": 100}
    if any(ch not in digit_map and ch not in unit_map for ch in token):
        return None

    total = 0
    current = 0
    for ch in token:
        if ch in digit_map:
            current = digit_map[ch]
            continue
        unit = unit_map[ch]
        if current == 0:
            current = 1
        total += current * unit
        current = 0
    total += current
    return total or None


def extract_page_metadata(header_path: List[str], section_title: str = "") -> Dict[str, Any]:
    candidates = [str(value).strip() for value in list(header_path or []) + [section_title] if str(value).strip()]
    for value in reversed(candidates):
        match = re.match(r"^(第\s*([0-9一二三四五六七八九十百零〇两]+)\s*页|page\s*([0-9]+))$", value, re.IGNORECASE)
        if not match:
            continue
        token = match.group(2) or match.group(3) or ""
        page_no = parse_simple_page_number(token)
        page_label = value
        result: Dict[str, Any] = {"page_label": page_label}
        if page_no is not None:
            result["page_no"] = page_no
        return result
    return {}


def summarize_child_page_metadata(children: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [str(child.get("page_label") or "").strip() for child in children if str(child.get("page_label") or "").strip()]
    page_numbers = [int(child["page_no"]) for child in children if isinstance(child.get("page_no"), int)]
    if page_numbers:
        unique_numbers = sorted(set(page_numbers))
        if len(unique_numbers) == 1:
            result: Dict[str, Any] = {"page_no": unique_numbers[0]}
            if labels:
                result["page_label"] = labels[0]
            return result
        return {"page_label": f"第 {unique_numbers[0]}-{unique_numbers[-1]} 页"}
    if labels:
        return {"page_label": labels[0]}
    return {}


def is_clean_text_item(item: Dict[str, Any]) -> bool:
    return str(item.get("source_text_profile") or "") == "clean_text"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# =========================
# 2. Regex
# =========================
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
MULTISPACE_RE = re.compile(r"\s+")
SPACE_TAB_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")
ZH_NEWLINE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])")
INLINE_NEWLINE_RE = re.compile(r"(?<=[a-zA-Z0-9,，、])\n(?=[a-zA-Z0-9\u4e00-\u9fff])")
PUNCT_SPACE_RE = re.compile(r"\s*([。！？!?；;：:])\s*")
PAREN_LEFT_SPACE_RE = re.compile(r"\(\s+")
PAREN_RIGHT_SPACE_RE = re.compile(r"\s+\)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])|(?<=[.])\s+")
PUNCT_COUNT_RE = re.compile(r"[。！？!?；;：:]")
SENTENCE_END_RE = re.compile(r"[。！？!?]$|[)）】”]$")
BROKEN_LINEBREAK_RE = re.compile(r"(?<=[\u4e00-\u9fffA-Za-z0-9])\n+(?=[\u4e00-\u9fffA-Za-z0-9])")
EN_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[\"'(\[\u4e00-\u9fffA-Z0-9])")
ABBREV_PERIOD_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|sr|jr|st|fig|eq|no|al)\.", re.IGNORECASE)
MULTI_INITIAL_RE = re.compile(r"\b(?:[A-Z]\.){2,}")

LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*+]|"
    r"[0-9]+[.)]|"
    r"[(（]?[0-9一二三四五六七八九十]+[)）]|"
    r"[一二三四五六七八九十]+[、.]|"
    r"[①②③④⑤⑥⑦⑧⑨⑩])\s+"
)

TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$")
FAQ_QUESTION_RE = re.compile(r"^\s*(?:Q[:：]|问[:：]|问题[:：]).+")
TOC_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:第?[一二三四五六七八九十0-9]+[章节条部分、.]?\s*)?.{1,60}\.{2,}\s*\d+\s*$"
)

PAGE_NOISE_RE = re.compile(
    r"^\s*(?:第?\s*\d+\s*页|page\s*\d+|\d+\s*/\s*\d+|[-_]{2,}|扫描全能王|版权所有|仅供参考)\s*$",
    re.IGNORECASE
)

SHORT_NOISE_RE = re.compile(r"^[\W_]{1,6}$|^[A-Za-z0-9]{1,3}$")
HTML_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
HTML_MEDIA_BLOCK_RE = re.compile(r"<(?:div|p)\b[^>]*>\s*<img\b[^>]*>\s*</(?:div|p)>", re.IGNORECASE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
LONG_URL_RE = re.compile(r"https?://\S{60,}")
MERGED_TABLE_ROW_START_RE = re.compile(r"(?<=[^\s|])(?=(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩])\s*\|)")
HP_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*HP\b", re.IGNORECASE)
PRICE_VALUE_RE = re.compile(r"\b(?:RM|USD)\s*[\d,]+(?:\.\d+)?\b", re.IGNORECASE)
GEARBOX_VALUE_RE = re.compile(r"\b\d+\s*[FR]/\d+\s*[FR]\b", re.IGNORECASE)
LIFT_VALUE_RE = re.compile(r"\b\d{3,4}\s*kg\b", re.IGNORECASE)
FLOW_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*L/min\b", re.IGNORECASE)
STYLE_FRAGMENT_RE = re.compile(r"style\s*=\s*['\"][^'\"]*['\"]", re.IGNORECASE)
REFERENCE_SECTION_RE = re.compile(r"^(?:references?|参考资料|参考文献)[:：]?", re.IGNORECASE)
URL_ONLY_LINE_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
URL_FRAGMENT_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
TRACKING_PARAM_RE = re.compile(r"(?:\?|&)(?:utm_[^=\s]+|source=chatgpt\.com)[^ \n]*", re.IGNORECASE)
LONG_NUMERIC_TOKEN_RE = re.compile(r"\d{3,}|[%％]|(?:RM|USD|马币|kg|L/min)\b", re.IGNORECASE)
RANGE_UNIT_RE = re.compile(r"(\d{1,3})\s*-\s+(\d{1,3}\s*(?:HP|kg|L/min|RM))\b", re.IGNORECASE)
DECIMAL_SPLIT_RE = re.compile(r"(\d+)\.\s+(\d+\s*(?:HP|kg|L/min))\b", re.IGNORECASE)
MONEY_SPLIT_RE = re.compile(r"\b(RM|USD)\s+(\d{1,3})\s*,\s*(\d{3})(?:\s*,\s*(\d{3}))?\b", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
SOURCE_SLUG_RE = re.compile(r"(?:[a-z0-9]+[-_/]){2,}[a-z0-9]+", re.IGNORECASE)
DOMAIN_FRAGMENT_RE = re.compile(r"\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|pdf|html)\b", re.IGNORECASE)
HEADING_ONLY_RE = re.compile(r"^\s*#{1,6}\s+.+$")
COPYRIGHT_LINE_RE = re.compile(r"\b(?:copyright|all rights reserved|no part of this report|warrant|liab(?:ility)?|免责声明|版权所有)\b", re.IGNORECASE)
ACKNOWLEDGEMENT_LINE_RE = re.compile(r"\b(?:acknowledg(?:e)?ments?|acknowledge|thank and acknowledge|thanks to|funding|support of|prepared by|prepared this report|staff members of|leaders in|vision)\b", re.IGNORECASE)
AUTHOR_PAGE_LINE_RE = re.compile(r"\b(?:author|authors|by\s+[A-Z][a-z]+|report prepared by|contributors|affiliation|department|university)\b", re.IGNORECASE)
LEGAL_BOILERPLATE_LINE_RE = re.compile(r"\b(?:express or implied|assume no legal liability|no warranty|without warranty|assumes no responsibility|not necessarily represent the views)\b", re.IGNORECASE)
CITATION_PAGE_RE = re.compile(r"\b(?:please use the following citation|citation for this report|please cite|recommended citation|how to cite)\b", re.IGNORECASE)
CAPTION_INDEX_RE = re.compile(r"^\s*(?:figure|fig\.|table)\s*\d+\s*[:.]", re.IGNORECASE)
CAPTION_LINE_RE = re.compile(r"\b(?:figure|fig\.|table)\s*\d+\b", re.IGNORECASE)
PHOTO_CREDIT_RE = re.compile(r"\bphoto credit\b", re.IGNORECASE)
LAYOUT_VISUAL_RE = re.compile(
    r"(?:页面|页内|图中|图表|图片|图像|配图|背景|左侧|右侧|上方|下方|左上角|右上角|左下角|右下角|底部|顶部|"
    r"横轴|纵轴|坐标轴|颜色|色块|曲线|柱状|折线|饼图|箭头|线条|版式|布局|Logo|logo|标识|水印|二维码|"
    r"办公楼|办公大楼|城市天际线|地球仪|封面图|牌匾)",
    re.IGNORECASE,
)
DECORATIVE_SENTENCE_RE = re.compile(
    r"(?:页面(?:左|右|上|下|底|顶)|左下角|右下角|左上角|右上角|页面底部|页面顶部|背景|配图|图片|图像|Logo|logo|"
    r"标识|水印|二维码|页码|办公楼|办公大楼|城市天际线|服务大众，情系民生|公司K\s*GAL|GAL及页码)",
    re.IGNORECASE,
)
PROMOTIONAL_TEXT_RE = re.compile(r"(?:微信群|公众号|二维码|回复[“\"']?研究报告|每日免费获取|扫码|关注公众号|加入.*群|起点财经)", re.IGNORECASE)
EMPTY_PAGE_PREFIX_RE = re.compile(
    r"^(?:这是一?份|这是一?张|该(?:页面|页|图表|图|部分)?|本(?:页面|页|图表)?|此(?:页面|页|图表)?|页面(?:左侧|右侧|上方|下方)?|"
    r"图表|图中|右侧|左侧|上方|下方)(?:主要)?(?:展示|显示|呈现|列出|列示|介绍|说明|描述|展示了|显示了|呈现了|列出了|介绍了|说明了|描述了)?(?:的是|了|为|出)?\s*",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(
    r"(?:\d{2,4}\s*年|\d+(?:\.\d+)?\s*(?:%|％|个百分点|pct|bp|bps|倍|亿元|万亿|亿元|万人|万家|家|个|页|"
    r"M2|GDP|kg|HP|L/min|RM|USD)|M2\s*/\s*GDP|同比|环比|CAGR)",
    re.IGNORECASE,
)
HIGH_INFO_RE = re.compile(
    r"(?:定义|是指|指的是|所谓|意味着|核心|结论|总结|建议|政策|措施|影响|原因|由于|因为|导致|因此|从而|表明|说明|显示|"
    r"趋势|增长|下降|上升|提高|降低|超过|低于|高于|相比|对比|分别|占比|比例|结构|风险|问题|关键|主要|投资建议|"
    r"预计|预测|判断|研判|长期|短期|变化|扩张|收缩|机制|关系|模型|结果)",
    re.IGNORECASE,
)
DEFINITION_SIGNAL_RE = re.compile(r"(?:定义|是指|指的是|所谓|意味着|refers to|means)", re.IGNORECASE)
CONCLUSION_SIGNAL_RE = re.compile(r"(?:结论|总结|综上|表明|说明|显示|核心观点|主要结论|takeaway|conclusion)", re.IGNORECASE)
CAUSAL_SIGNAL_RE = re.compile(r"(?:因为|由于|导致|因此|从而|原因|成因|使得|推动|带来|引发|造成)", re.IGNORECASE)
COMPARISON_SIGNAL_RE = re.compile(r"(?:相比|对比|高于|低于|超过|不足|上升|下降|增加|减少|分别|差异|变化|较)", re.IGNORECASE)
POLICY_SIGNAL_RE = re.compile(r"(?:建议|应当|需要|必须|政策|措施|监管|推进|完善|优化|鼓励|限制|防范|投资建议)", re.IGNORECASE)
COVER_PAGE_RE = re.compile(r"(?:封面|主标题|副标题|报告.*(?:发布|出品)|发布时间|发布日期|撰写团队|研究报告封面)", re.IGNORECASE)
CATALOG_PAGE_RE = re.compile(r"(?:目录|第一部分|第二部分|第三部分|第四部分|第五部分|第六部分|第七部分|第八部分|第九部分|第十部分|起始于第\d+页|章节)", re.IGNORECASE)
MEMBER_LIST_RE = re.compile(r"(?:研究团队|课题组|成员包括|课题组长|分析师|研究助理|执业证书编号|联系邮箱|名单)", re.IGNORECASE)
HTML_FRAGMENT_RE = re.compile(r"<\s*/?\s*(?:div|p|span|br)\b[^>]*>", re.IGNORECASE)
FORMULA_NOISE_RE = re.compile(r"(?:\$\s*\^|\^\{|\}\\$|\\frac|\\cdot|km·h)", re.IGNORECASE)
TABLE_GLUE_LINE_RE = re.compile(r"\|.{200,}\|")
INLINE_FORMULA_RE = re.compile(r"\${1,2}[^$\n]{1,220}\${1,2}")
LATEX_COMMAND_RE = re.compile(r"\\(?:frac|cdot|times|sqrt|left|right|mathrm|text|begin|end|sum|int)\b", re.IGNORECASE)
METRIC_UNIT_TOKEN_RE = re.compile(r"^(?:miles?|hours?|kwh|dge|kg|gallon(?:s)?|hp|l/min|rm|%|percent)$", re.IGNORECASE)
METRIC_VALUE_TOKEN_RE = re.compile(r"^(?:\d[\d,]*(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)$")
GLUED_HEADER_TOKENS = [
    "东南亚", "泰国", "越南", "菲律宾", "马来西亚", "印尼", "新加坡", "柬埔寨", "老挝", "缅甸", "说明",
    "中国", "美国", "欧洲", "印度", "澳大利亚", "英国", "日本", "韩国",
]
CITATION_LEAD_RE = re.compile(r"^(?:citation:|cite:|editor:|author:|doi:)", re.IGNORECASE)
FORMULA_HEAVY_RE = re.compile(r"(?:\$\$|\$|\\frac|\\cdot|\\times|\\sum|\\int|\\begin|\\end)", re.IGNORECASE)
OCR_PLACEHOLDER_RE = re.compile(
    r"(?:the information on the (?:chart|image|figure|page)[^.]{0,120}?too blurry to be identified"
    r"|please provide a more clear and accurate image"
    r"|too blurry to be identified)",
    re.IGNORECASE,
)
OCR_FAILURE_RE = re.compile(
    r"(?:too blurry|cannot be identified|blurry to be identified|"
    r"please provide.*(?:clear|accurate).*image|"
    r"information.*chart.*blurry|"
    r"unable to recognize|recognition failed)",
    re.IGNORECASE,
)
IMPLICIT_HEADER_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9（）()·、:：\-]{2,40}$")
PUBLICATION_META_RE = re.compile(
    r"(?:\b(?:cip|isbn|issn|webmaster|publisher|published by|editor|printing|printed|edition|price|email|e-mail)\b"
    r"|(?:CIP|ISBN|ISSN|出版社|出版|责任编辑|责编|印刷|版次|定价|邮编|网址|电子邮箱))",
    re.IGNORECASE,
)
INLINE_HASH_HEADING_RE = re.compile(r"(?<=[。！？；;:：])\s*(#{1,6}\s+)")
LOW_VALUE_SECTION_RE = re.compile(
    r"(?:\bCIP\b|\bISBN\b|\bISSN\b|目录|鐩綍|前言|代前言|序言|序|跋|后记|后記|内容简介|内容提要|出版说明|版权页|图书在版编目|推荐语|推荐|作者简介|名家推荐)",
    re.IGNORECASE,
)
ENDORSEMENT_CREDENTIAL_RE = re.compile(
    r"(?:董事长|CEO|创始人|总经理|总编辑|主理人|院长|合伙人|著名|知名|教授|学者|总裁|銆婄幆鐞冭储缁忋€)",
    re.IGNORECASE,
)
ENDORSEMENT_PRAISE_RE = re.compile(
    r"(?:推荐(?:这本书|此书|本书)|诚挚推荐|值得(?:一读|关注)|作者以|本书(?:是一部|以|可以视为)|全书以|读者阅读此书)",
    re.IGNORECASE,
)
PREFACE_STYLE_RE = re.compile(
    r"(?:代前言|前言|序言|后记|我是[一二三四五六七八九十A-Za-z0-9_]{0,8}|本书的目标读者|最后以我撰写|君子爱财)",
    re.IGNORECASE,
)
MOJIBAKE_SUSPECT_CHARS = set("銆鈥锛鍦浠鏄鐨鍙鍏鍚鍒闂璇閫鏉缁绗闈鎴鍐閲")
MOJIBAKE_SUSPECT_FRAGMENTS = (
    "銆", "鈥", "锛", "鐨", "鏄", "鍦", "浠", "鍙", "鍏", "鍚", "鍒", "闂", "璇", "閫", "鏉", "缁", "绗",
)

OCR_TERM_FIXUPS = [
    (re.compile(r"\bMalay\s*sia\b", re.IGNORECASE), "Malaysia"),
    (re.compile(r"\bJohn\s*Dee(?:te|re)\s*(?=\d)", re.IGNORECASE), "John Deere "),
    (re.compile(r"\bJohn\s*Dee(?:te|re)\b", re.IGNORECASE), "John Deere"),
    (re.compile(r"\bNew\s*Holland\b", re.IGNORECASE), "New Holland"),
    (re.compile(r"\bMassey\s*Ferguson\b", re.IGNORECASE), "Massey Ferguson"),
    (re.compile(r"\bKubota\s*L\s+(\d{3,4})\b", re.IGNORECASE), r"Kubota L\1"),
    (re.compile(r"\bKubota\s*([LM])\s*((?:\d\s*){3,4})\b", re.IGNORECASE), lambda m: f"Kubota {m.group(1).upper()}{re.sub(r'\\s+', '', m.group(2))}"),
]

# =========================
# 3. Embedding utils
# =========================
def sleep_with_jitter(base_delay: float):
    time.sleep(base_delay)


def get_local_embedder() -> QwenLocalEmbeddingModel:
    global LOCAL_EMBEDDER
    if LOCAL_EMBEDDER is None:
        # Reuse one model instance for the whole process to avoid repeated loading.
        LOCAL_EMBEDDER = QwenLocalEmbeddingModel(
            model_name_or_path=LOCAL_EMBEDDING_MODEL_PATH,
            device=LOCAL_EMBEDDING_DEVICE,
            dtype=LOCAL_EMBEDDING_DTYPE,
            attn_implementation=LOCAL_EMBEDDING_ATTN_IMPLEMENTATION,
            max_length=LOCAL_EMBEDDING_MAX_LENGTH,
        )
    return LOCAL_EMBEDDER


def call_embedding_api_with_retry(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    delay = INITIAL_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return get_local_embedder().encode(texts, batch_size=BATCH_SIZE)
        except Exception as e:
            message = str(e)
            if attempt == MAX_RETRIES or "out of memory" not in message.lower():
                raise RuntimeError(f"Local embedding request failed: {e}") from e
            print(f"[WARN] Local embedding batch failed, retry {attempt}/{MAX_RETRIES} after {delay:.1f}s: {message}")
            sleep_with_jitter(delay)
            delay *= 2

    raise RuntimeError("Local embedding request exhausted unexpectedly")


def preflight_runtime_checks(input_files: List[Path]) -> None:
    # Only block startup when there is actually new work that needs embedding.
    pending_files = [
        input_file
        for input_file in input_files
        if FORCE_REPROCESS or not build_output_file_path(input_file, INPUT_PATH, OUTPUT_DIR).exists()
    ]
    pending_processing = bool(pending_files)
    if not pending_processing:
        return

    requires_embedding = any(
        should_use_embedding_refinement(resolve_text_source_profile(input_file))
        for input_file in pending_files
    )
    if not requires_embedding:
        return

    issues = collect_runtime_dependency_issues(
        model_name_or_path=LOCAL_EMBEDDING_MODEL_PATH,
        device=LOCAL_EMBEDDING_DEVICE,
        require_qdrant=False,
    )
    if not issues:
        return

    details = "\n".join(f"- {issue}" for issue in issues)
    hints = [
        "- 当前待处理文件会在语义合并或近重去重阶段用到本地 embedding，请先保证本地模型和依赖可用。",
        f"- 当前 embedding 模型路径配置是: {LOCAL_EMBEDDING_MODEL_PATH}",
        "- 最终 chunk 向量化和 Qdrant 入库已拆到 rag_pipeline.ingest.embedding_qdrant 单独执行。",
    ]

    raise RuntimeError(
        "启动前检查失败，当前环境还不满足切片运行条件:\n"
        f"{details}\n"
        "处理建议:\n"
        f"{chr(10).join(hints)}"
    )


def fetch_embeddings_parallel(uncached_texts: List[str]) -> Dict[str, List[float]]:
    if not uncached_texts:
        return {}
    embeddings = call_embedding_api_with_retry(uncached_texts)
    return {text: emb for text, emb in zip(uncached_texts, embeddings)}


def get_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    results = [None] * len(texts)
    uncached_texts = []
    seen_uncached = set()

    for i, text in enumerate(texts):
        key = text.strip()
        with EMBED_CACHE_LOCK:
            cached = EMBED_CACHE.get(key)
        if cached is not None:
            results[i] = cached
        elif key not in seen_uncached:
            uncached_texts.append(key)
            seen_uncached.add(key)

    if ENABLE_PARALLEL_EMBEDDING and len(uncached_texts) > BATCH_SIZE:
        parallel_results = fetch_embeddings_parallel(uncached_texts)
        with EMBED_CACHE_LOCK:
            EMBED_CACHE.update(parallel_results)
    else:
        for i in range(0, len(uncached_texts), BATCH_SIZE):
            batch = uncached_texts[i:i + BATCH_SIZE]
            batch_embeddings = call_embedding_api_with_retry(batch)
            with EMBED_CACHE_LOCK:
                for text, emb in zip(batch, batch_embeddings):
                    EMBED_CACHE[text] = emb
            if i + BATCH_SIZE < len(uncached_texts):
                time.sleep(REQUEST_INTERVAL)

    for i, text in enumerate(texts):
        with EMBED_CACHE_LOCK:
            results[i] = EMBED_CACHE[text.strip()]

    return results

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# =========================
# 4. File IO
# =========================
def supported_input_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".txt"


def iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if supported_input_file(input_path) else []
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(p for p in input_path.rglob("*") if supported_input_file(p))


def build_output_file_path(input_file: Path, input_root: Path, output_dir: Path) -> Path:
    relative_parent = input_file.relative_to(input_root).parent if input_root.is_dir() else Path()
    return output_dir / relative_parent / f"{input_file.name}.chunks.json"


def write_chunks_file(
    output_file: Path,
    source_file: Path,
    chunks: List[Dict[str, Any]],
    parent_chunks: List[Dict[str, Any]] | None = None,
):
    # This is the plain chunk output used by later embedding / vector store steps.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    parent_chunks = parent_chunks or []
    payload = {
        "source_file": str(source_file),
        "chunk_count": len(chunks),
        "parent_chunk_count": len(parent_chunks),
        "generated_at": int(time.time()),
        "chunks": [serialize_chunk_for_output(chunk, source_file=str(source_file)) for chunk in chunks],
        "parent_chunks": [serialize_chunk_for_output(chunk, source_file=str(source_file)) for chunk in parent_chunks],
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_chunks_file(output_file: Path) -> List[Dict[str, Any]]:
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Invalid chunk file format: {output_file}")
    return chunks


def prune_chunk_value(value: Any) -> Any:
    if isinstance(value, dict):
        pruned: Dict[str, Any] = {}
        for key, item in value.items():
            if key in OUTPUT_FIELD_BLACKLIST or SUMMARY_KEY_RE.search(str(key)):
                continue
            pruned[key] = prune_chunk_value(item)
        return pruned
    if isinstance(value, list):
        return [prune_chunk_value(item) for item in value]
    return value


def build_chunk_uid(source_file: str, chunk: Dict[str, Any]) -> str:
    # Keep the UID stable across reruns so JSON and vector store can be matched later.
    seed = "|".join(
        [
            str(source_file or ""),
            str(chunk.get("doc_id", "")),
            str(chunk.get("section_id", "")),
            str(chunk.get("chunk_in_section", "")),
            str(chunk.get("embedding_text", chunk.get("text", "")) or ""),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def build_parent_chunk_uid(source_file: str, chunk: Dict[str, Any]) -> str:
    # Parent UIDs should stay stable as long as the grouped child UIDs stay stable.
    child_uids = [str(uid) for uid in chunk.get("child_chunk_uids", []) if str(uid).strip()]
    seed = "|".join(
        [
            str(source_file or ""),
            str(chunk.get("doc_id", "")),
            str(chunk.get("section_id", "")),
            str(chunk.get("parent_chunk_index", chunk.get("chunk_in_section", ""))),
            str(chunk.get("chunk_level", "parent")),
            ",".join(child_uids),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def serialize_chunk_for_output(chunk: Dict[str, Any], source_file: str = "") -> Dict[str, Any]:
    # Only persist the fields needed by later embedding / retrieval stages.
    serialized = {}
    for key in OUTPUT_FIELD_WHITELIST:
        if key in chunk:
            serialized[key] = prune_chunk_value(chunk[key])

    for key in OUTPUT_FIELD_BLACKLIST:
        serialized.pop(key, None)

    if OUTPUT_INCLUDE_RAW_TEXT and "raw_text" in chunk:
        serialized["raw_text"] = chunk["raw_text"]
    if OUTPUT_INCLUDE_EMBEDDING and "embedding" in chunk:
        serialized["embedding"] = chunk["embedding"]
    chunk_level = str(chunk.get("chunk_level", "child"))
    if chunk.get("chunk_uid"):
        serialized["chunk_uid"] = str(chunk["chunk_uid"])
    elif chunk_level == "parent":
        serialized["chunk_uid"] = build_parent_chunk_uid(source_file, chunk)
    else:
        serialized["chunk_uid"] = build_chunk_uid(source_file, chunk)

    return serialized


def load_documents(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix != ".txt":
        raise ValueError(f"Unsupported file type: {suffix}")

    text = file_path.read_text(encoding="utf-8")
    return [{"doc_id": 0, "source": str(file_path), "title": file_path.stem, "text": text}]


def load_cleaned_documents(input_file: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    documents = load_documents(input_file)
    if not CLEAN_SOURCE_BEFORE_CHUNKING or input_file.suffix.lower() != ".txt":
        return documents, {}

    cleaned_documents: List[Dict[str, Any]] = []
    aggregate_stats = empty_clean_stats()
    source_profile = resolve_text_source_profile(input_file)

    for doc in documents:
        cleaned_text, clean_stats = clean_structured_text(doc.get("text", ""))
        new_doc = dict(doc)
        new_doc["text"] = cleaned_text
        new_doc["source"] = str(input_file)
        new_doc["source_text_profile"] = source_profile
        if not new_doc.get("title"):
            new_doc["title"] = input_file.stem
        cleaned_documents.append(new_doc)
        for key in aggregate_stats:
            aggregate_stats[key] += int(clean_stats.get(key, 0))

    return cleaned_documents, aggregate_stats


def clean_structured_text(text: str) -> Tuple[str, Dict[str, int]]:
    stats = empty_clean_stats()
    source_text = str(text or "")
    stats["input_chars"] = len(source_text)
    if not source_text.strip():
        return "", stats

    cleaned = HTML_MEDIA_BLOCK_RE.sub("\n\n", source_text)
    cleaned = HTML_IMG_RE.sub(" ", cleaned)
    cleaned = MARKDOWN_IMAGE_RE.sub(" ", cleaned)
    cleaned = normalize_html_tables(cleaned)
    cleaned = normalize_pipe_tables(cleaned)
    cleaned = normalize_inline_heading_boundaries(cleaned)
    cleaned = cleaned.replace("\u3000", " ").replace("\xa0", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = MULTINEWLINE_RE.sub("\n\n", cleaned)

    normalized_lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = SPACE_TAB_RE.sub(" ", raw_line).strip()
        if not line:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        if OCR_PLACEHOLDER_RE.search(line):
            stats["removed_ocr_lines"] += 1
            continue
        if HTML_TAG_RE.search(line):
            line = strip_html_tags(line)
            if not line:
                stats["removed_html_lines"] += 1
                continue
        if PAGE_NOISE_RE.match(line) and not HEADER_RE.match(line):
            continue
        if TOC_LINE_RE.match(line):
            stats["removed_toc_lines"] += 1
            continue
        normalized_lines.append(line)

    cleaned = "\n".join(normalized_lines)
    cleaned = dedup_lines(cleaned)
    cleaned = MULTINEWLINE_RE.sub("\n\n", cleaned).strip()
    stats["output_chars"] = len(cleaned)
    stats["kept_lines"] = len([line for line in cleaned.splitlines() if line.strip()])
    return cleaned, stats


def prepare_text_for_chunking(text: str) -> str:
    cleaned_text, _ = clean_structured_text(text)
    return cleaned_text


# =========================
# 5. Clean text
# =========================
def dedup_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    prev_norm = None
    for line in lines:
        raw = line.strip()
        norm = MULTISPACE_RE.sub(" ", raw)
        if not norm:
            cleaned.append("")
            prev_norm = None
            continue
        if norm == prev_norm:
            continue
        cleaned.append(raw)
        prev_norm = norm
    return "\n".join(cleaned)


def looks_like_toc_line(line: str) -> bool:
    return bool(line.strip() and TOC_LINE_RE.match(line.strip()))


def strip_low_value_lines(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        raw = line.strip()
        if not raw:
            kept.append("")
            continue
        if looks_like_toc_line(raw):
            continue
        kept.append(line)
    return "\n".join(kept)


def collapse_broken_linebreaks(text: str) -> str:
    if not text:
        return text
    # Preserve paragraph breaks and markdown block boundaries; only repair true single-line OCR wraps.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9])\n(?=[\u4e00-\u9fffA-Za-z0-9])", "", text)
    return text


def _is_sentence_terminal(text: str) -> bool:
    return bool(SENTENCE_END_RE.search((text or "").strip()))


def strip_html_tags(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def strip_ocr_placeholder_noise(text: str) -> str:
    if not text:
        return ""
    cleaned = OCR_PLACEHOLDER_RE.sub(" ", text)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip(" -|:;,，。.!?！？")
    return cleaned.strip()


def contains_ocr_placeholder(text: str) -> bool:
    return bool(text and OCR_PLACEHOLDER_RE.search(text))


def meaningful_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text or ""))


def estimate_mojibake_ratio(text: str) -> float:
    s = strip_html_tags(text).strip()
    if not s:
        return 0.0

    letters = meaningful_char_count(s)
    if letters < 16:
        return 0.0

    suspect_chars = sum(1 for ch in s if ch in MOJIBAKE_SUSPECT_CHARS)
    suspect_fragments = sum(s.count(token) for token in MOJIBAKE_SUSPECT_FRAGMENTS)
    weighted_hits = suspect_chars + suspect_fragments * 2
    return min(1.0, weighted_hits / max(letters, 1))


def looks_like_mojibake_text(text: str) -> bool:
    s = strip_html_tags(text).strip()
    if not s:
        return False

    ratio = estimate_mojibake_ratio(s)
    fragment_hits = sum(s.count(token) for token in MOJIBAKE_SUSPECT_FRAGMENTS)
    return ratio >= 0.42 or (ratio >= 0.28 and fragment_hits >= 6)


def looks_like_publication_metadata(text: str) -> bool:
    s = strip_html_tags(text).strip()
    if not s:
        return False

    hits = 0
    if PUBLICATION_META_RE.search(s):
        hits += 1
    if "@" in s or "www." in s.lower() or "http" in s.lower():
        hits += 1
    if re.search(r"\b\d{10,13}\b", s):
        hits += 1
    if len(re.findall(r"\d", s)) >= 10 and any(token in s.lower() for token in ["cip", "isbn", "price", "edition", "printed"]):
        hits += 1

    return hits >= 2 or (hits >= 1 and len(s) <= 260 and meaningful_char_count(s) <= max(80, len(s) // 2))


def is_low_information_text(text: str) -> bool:
    s = strip_html_tags(text).strip()
    if not s:
        return True

    letters = meaningful_char_count(s)
    digits = len(re.findall(r"\d", s))
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", s)

    if letters <= 2 and len(s) <= 12:
        return True
    if letters <= 8 and digits >= 6 and len(s) <= 40:
        return True
    if len(tokens) <= 3 and letters <= 10 and len(s) <= 28:
        return True
    if letters < max(6, len(s) // 5) and digits >= letters and len(s) <= 80:
        return True
    if looks_like_mojibake_text(s):
        return True
    return False


def normalize_inline_heading_boundaries(text: str) -> str:
    if not text:
        return text
    return INLINE_HASH_HEADING_RE.sub(lambda m: "\n\n" + m.group(1), text)


def normalize_html_tables(text: str) -> str:
    def replace_table(match: re.Match) -> str:
        table_html = match.group(1)
        rows = []
        for row_html in HTML_ROW_RE.findall(table_html):
            cells = [strip_html_tags(cell) for cell in HTML_CELL_RE.findall(row_html)]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        if not rows:
            fallback = strip_html_tags(table_html)
            return f"\n\n{fallback}\n\n" if fallback else "\n\n"
        return "\n\n" + "\n".join(rows) + "\n\n"

    return HTML_TABLE_RE.sub(replace_table, text)


def looks_like_pipe_table_line(text: str) -> bool:
    line = text.strip()
    if not line:
        return False
    pipe_count = line.count("|")
    if pipe_count < 2:
        return False
    if len(line) <= PIPE_TABLE_MAX_LINE_LENGTH and pipe_count >= 2:
        return True
    if pipe_count >= 4:
        return True
    return False


def is_label_like_cell(text: str) -> bool:
    s = normalize_table_cell_text(text)
    if not s:
        return False
    if len(s) > 70:
        return False
    if s.endswith((':', '：')):
        return True
    if re.search(r"\b(?:Figure|Fig\.|Table)\s*\d+\b", s, re.IGNORECASE):
        return True
    if re.search(r"[A-Za-z\u4e00-\u9fff].{0,20}[:：]$", s):
        return True
    if not is_value_like_cell(s) and len(s) <= 35:
        return True
    return False


def is_value_like_cell(text: str) -> bool:
    s = normalize_table_cell_text(text)
    if not s:
        return False
    if HP_VALUE_RE.search(s) or PRICE_VALUE_RE.search(s) or GEARBOX_VALUE_RE.search(s):
        return True
    if FLOW_VALUE_RE.search(s) or LIFT_VALUE_RE.search(s):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*%|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", s):
        return True
    if re.search(r"\b\d{4}\b", s):
        return True
    if len(s) >= 8 and any(ch.isdigit() for ch in s):
        return True
    return False


def split_merged_metric_cell(text: str) -> List[str]:
    s = normalize_table_cell_text(text)
    if not s:
        return []
    tokens = s.split()
    if len(tokens) < 4:
        return [s]

    parts: List[str] = []
    current: List[str] = []

    for idx, token in enumerate(tokens):
        if current:
            prev = current[-1]
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            should_split = (
                (
                    METRIC_UNIT_TOKEN_RE.match(prev)
                    or METRIC_VALUE_TOKEN_RE.match(prev)
                )
                and token[:1].isupper()
                and not METRIC_VALUE_TOKEN_RE.match(token)
                and (not next_token or token.lower() not in {"of", "and", "the"})
            )
            if should_split:
                parts.append(" ".join(current).strip())
                current = [token]
                continue
        current.append(token)

    if current:
        parts.append(" ".join(current).strip())

    if len(parts) > 1:
        return [part for part in parts if part]
    return [s]


def expand_compound_pipe_cells(cells: List[str]) -> List[List[str]]:
    normalized = []
    for cell in cells:
        fragments = split_merged_metric_cell(cell)
        normalized.extend([fragment for fragment in fragments if fragment])
    if len(normalized) < 6:
        return [normalized]

    lead = normalized[0]
    rest = normalized[1:]
    if len(rest) < 4:
        return [normalized]

    label_value_pairs = 0
    for idx in range(0, len(rest) - 1, 2):
        left = rest[idx]
        right = rest[idx + 1]
        if is_label_like_cell(left) and is_value_like_cell(right):
            label_value_pairs += 1

    if label_value_pairs >= 2:
        rows = []
        for idx in range(0, len(rest) - 1, 2):
            left = rest[idx]
            right = rest[idx + 1]
            if is_label_like_cell(left) and is_value_like_cell(right):
                if lead and not is_label_like_cell(lead) and not is_value_like_cell(lead):
                    rows.append([lead, left, right])
                else:
                    rows.append([left, right])
        if rows:
            return rows

    return [normalized]


def split_pipe_table_line(line: str) -> List[str]:
    expanded_line = MERGED_TABLE_ROW_START_RE.sub("\n", line.strip())
    logical_lines = [part.strip() for part in expanded_line.splitlines() if part.strip()]
    normalized_rows = []

    for logical_line in logical_lines:
        cells = [cell.strip() for cell in logical_line.split("|") if cell.strip()]
        if len(cells) < 3:
            normalized_rows.append(logical_line.strip())
            continue

        expanded_rows = expand_compound_pipe_cells(cells)
        if expanded_rows and expanded_rows != [cells]:
            normalized_rows.extend([" | ".join(row) for row in expanded_rows if row])
            continue

        rows = []
        header_width = min(len(cells), PIPE_TABLE_CELL_SOFT_LIMIT)
        rows.append(" | ".join(cells[:header_width]))
        current = []

        for cell in cells[header_width:]:
            candidate = current + [cell]
            joined = " | ".join(candidate)
            if current and len(joined) > PIPE_TABLE_ROW_CHAR_TARGET:
                rows.append(" | ".join(current))
                current = [cell]
            else:
                current.append(cell)

        if current:
            rows.append(" | ".join(current))

        normalized_rows.extend([row for row in rows if row.strip()])

    return normalized_rows


def normalize_pipe_tables(text: str) -> str:
    lines = text.splitlines()
    normalized = []
    for line in lines:
        if looks_like_pipe_table_line(line):
            normalized.extend(split_pipe_table_line(line))
        else:
            normalized.append(line)
    return "\n".join(normalized)


def clean_text(text: str) -> str:
    text = HTML_MEDIA_BLOCK_RE.sub(" ", text)
    text = HTML_IMG_RE.sub(" ", text)
    text = MARKDOWN_IMAGE_RE.sub(" ", text)
    text = LONG_URL_RE.sub(" ", text)
    text = STYLE_FRAGMENT_RE.sub(" ", text)
    text = TRACKING_PARAM_RE.sub("", text)
    text = normalize_html_tables(text)
    text = normalize_pipe_tables(text)
    text = collapse_broken_linebreaks(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_TAB_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    text = ZH_NEWLINE_RE.sub("", text)
    text = INLINE_NEWLINE_RE.sub(" ", text)
    text = PUNCT_SPACE_RE.sub(r"\1", text)
    text = PAREN_LEFT_SPACE_RE.sub("(", text)
    text = PAREN_RIGHT_SPACE_RE.sub(")", text)
    text = strip_low_value_lines(text)
    text = dedup_lines(text)
    return text.strip()


def clean_text_fast(text: str) -> str:
    return clean_text(text)


def sanitize_header_text(text: str, max_len: int = 80) -> str:
    text = strip_html_tags(text)
    text = strip_ocr_placeholder_noise(text)
    text = LONG_URL_RE.sub(" ", text)
    text = STYLE_FRAGMENT_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip(" -|:;,\u3002\uff01\uff1f")
    if not text:
        return ""
    if re.match(r"^(?:第\s*[一二三四五六七八九十0-9]+\s*[页章节部分篇]|[（(]?\d+[)）]?\s*[^。！？!?]{1,40}|[一二三四五六七八九十]+、[^。！？!?]{1,40})$", text):
        return text if len(text) <= max_len else text[:max_len].rstrip() + "..."
    if CAPTION_INDEX_RE.match(text) or CAPTION_LINE_RE.search(text):
        return ""
    if looks_like_publication_metadata(text):
        return ""
    if REFERENCE_SECTION_RE.match(text.strip()):
        return ""
    if is_low_information_text(text):
        # Preserve concise section headings such as "(2) 理解社会经济现象".
        if meaningful_char_count(text) < 4:
            return ""
    for sep in ["\u3002", "\uff01", "\uff1f", "!", "?", ";"]:
        if sep in text and len(text) > max_len // 2:
            candidate = text.split(sep, 1)[0].strip()
            if 2 <= len(candidate) <= max_len:
                text = candidate
                break
    if len(text) > 42 and ("\uff1a" in text or ":" in text):
        candidate = re.split(r"[:\uff1a]", text, maxsplit=1)[0].strip()
        if 2 <= len(candidate) <= max_len:
            text = candidate
    if "|" in text and len(text) > max_len:
        text = text.split("|", 1)[0].strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text.strip()


def split_header_title_and_inline_body(text: str, max_title_len: int = 24) -> Tuple[str, str]:
    cleaned = strip_html_tags(text or "")
    cleaned = strip_ocr_placeholder_noise(cleaned)
    cleaned = LONG_URL_RE.sub(" ", cleaned)
    cleaned = STYLE_FRAGMENT_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip(" -|:;,\u3002\uff01\uff1f")
    if not cleaned:
        return "", ""

    prefix = ""
    body = cleaned
    prefix_match = re.match(r"^([(\uff08]?\d+[)\uff09]?[.\u3001]?\s*)", cleaned)
    if prefix_match:
        prefix = prefix_match.group(1)
        body = cleaned[prefix_match.end():].strip()

    split_at = None
    trigger_match = re.search(
        r"(资产负债表|利润表|现金流量表|是描述|是反映|是衡量|主要反映|主要描述|通常反映|通常包括|用于说明|用于衡量|例如|比如|案例研究|案例|所谓|指的是)",
        body,
    )
    if trigger_match and trigger_match.start() >= 4:
        split_at = trigger_match.start()
    elif prefix and len(body) > max_title_len:
        split_at = max_title_len
    elif len(body) > max_title_len + 10 and not re.search(r"[\u3002\uff01\uff1f:：;；]", body[:max_title_len]):
        split_at = max_title_len

    title_body = body[:split_at].strip() if split_at else body
    inline_body = body[split_at:].strip() if split_at else ""

    title = sanitize_header_text(f"{prefix}{title_body}".strip(), max_len=max_title_len + len(prefix))
    if not title:
        return "", ""

    inline_body = clean_text_fast(inline_body)
    if inline_body and normalize_for_hash(inline_body) == normalize_for_hash(title):
        inline_body = ""
    return title, inline_body


def is_header_like(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if HEADER_RE.match(line):
        return True
    if len(line) <= 40 and re.match(r"^\s*(第[一二三四五六七八九十0-9]+[章节部分节]|[0-9]+(?:\.[0-9]+){0,3}|[一二三四五六七八九十]+、)", line):
        return True
    return False


def is_implicit_header_line(line: str, next_line: str, prev_was_blank: bool) -> bool:
    stripped = sanitize_header_text(line, 80)
    if not stripped:
        return False
    if not prev_was_blank or not next_line.strip():
        return False
    if HEADER_RE.match(line.strip()) or LIST_LINE_RE.match(line.strip()) or TABLE_LINE_RE.match(line.strip()):
        return False
    if len(stripped) > 40 or len(stripped) < 2:
        return False
    if not IMPLICIT_HEADER_RE.fullmatch(stripped):
        return False
    if _is_sentence_terminal(stripped):
        return False
    if looks_like_publication_metadata(stripped) or is_reference_section_title(stripped):
        return False
    return True


def is_page_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PAGE_NOISE_RE.match(s):
        return True
    if len(s) <= 12 and re.match(r"^\d+$", s):
        return True
    return False


def is_weak_text_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if SHORT_NOISE_RE.match(s):
        return True
    if len(s) <= 6 and not re.search(r"[\u4e00-\u9fffA-Za-z]", s):
        return True
    return False


def is_reference_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if REFERENCE_SECTION_RE.match(s):
        return True
    if URL_ONLY_LINE_RE.match(s):
        return True
    if "utm_source=chatgpt.com" in s.lower():
        return True
    if "source=chatgpt.com" in s.lower():
        return True
    if s.lower().startswith("http"):
        return True
    if len(s) <= 120 and URL_FRAGMENT_RE.search(s):
        return True
    return False


def is_reference_section_title(text: str) -> bool:
    s = sanitize_header_text(text, 120).lower()
    if not s:
        return False
    keywords = ["references", "reference", "bibliography", "参考", "文献", "资料来源", "sources", "works cited"]
    return any(keyword in s for keyword in keywords)


def is_meta_noise_section(text: str) -> bool:
    return is_reference_section_title(text) or looks_like_front_matter_noise(text)


def is_source_like_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if is_reference_noise_line(s):
        return True
    if DOI_RE.search(s):
        return True
    if DOMAIN_FRAGMENT_RE.search(s):
        return True
    if SOURCE_SLUG_RE.search(s) and len(s) <= 180:
        return True
    if s.lower().endswith((".pdf", ".html")):
        return True
    return False


def is_copyright_noise_line(line: str) -> bool:
    s = line.strip()
    return bool(s and COPYRIGHT_LINE_RE.search(s))


def is_acknowledgement_noise_line(line: str) -> bool:
    s = line.strip()
    return bool(s and ACKNOWLEDGEMENT_LINE_RE.search(s))


def is_author_page_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if AUTHOR_PAGE_LINE_RE.search(s):
        return True
    if LEGAL_BOILERPLATE_LINE_RE.search(s):
        return True
    return False


def looks_like_front_matter_noise(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    noise_hits = 0
    for line in lines:
        if (
            is_copyright_noise_line(line)
            or is_acknowledgement_noise_line(line)
            or is_author_page_noise_line(line)
            or CAPTION_INDEX_RE.match(line)
        ):
            noise_hits += 1

    if noise_hits >= 2:
        return True
    if noise_hits >= 1 and len(lines) <= 6:
        return True

    lowered = text.lower()
    if any(token in lowered for token in [
        "acknowledgements", "acknowledgments", "disclaimer", "copyright", "all rights reserved",
        "prepared this report", "thank and acknowledge", "not necessarily represent the views",
        "please use the following citation", "citation for this report", "recommended citation", "please cite"
    ]):
        return True
    if LOW_VALUE_SECTION_RE.search(text):
        return True
    return False


def classify_section_kind(header_path: List[str], content: str) -> str:
    context = " > ".join([part for part in header_path if part])
    probe = "\n".join([context, content[:1200]]).strip()
    if not probe:
        return "main"
    if looks_like_mojibake_text(probe) and meaningful_char_count(probe) >= 40:
        return "garbled"
    lines = [line.strip() for line in probe.splitlines() if line.strip()]
    short_probe = len(probe) <= 900
    if PROMOTIONAL_TEXT_RE.search(probe):
        return "promo_page"
    if MEMBER_LIST_RE.search(probe) and (short_probe or sum(1 for line in lines[:8] if MEMBER_LIST_RE.search(line)) >= 1):
        return "member_list"
    if CATALOG_PAGE_RE.search(probe) and (
        sum(1 for line in lines if CATALOG_PAGE_RE.search(line) or looks_like_toc_line(line)) >= 2
        or "目录" in context
        or "目录" in probe[:160]
        or len(re.findall(r"起始于第\s*\d+\s*页", probe)) >= 3
        or len(re.findall(r"第[一二三四五六七八九十]+部分", probe)) >= 4
    ):
        return "catalog"
    if COVER_PAGE_RE.search(probe) and short_probe:
        return "cover"
    if any(looks_like_toc_line(line.strip()) for line in probe.splitlines() if line.strip()):
        return "toc"
    if looks_like_publication_metadata(probe) or "图书在版编目" in probe or "CIP" in probe:
        return "publication"
    if LOW_VALUE_SECTION_RE.search(context):
        if re.search(r"(?:目录|鐩綍)", context, re.IGNORECASE):
            return "toc"
        if re.search(r"(?:前言|代前言|序言|序|跋|后记|后記)", context, re.IGNORECASE):
            return "preface"
        return "endorsement"
    credential_hits = sum(1 for line in lines[:10] if ENDORSEMENT_CREDENTIAL_RE.search(line))
    title_hits = sum(1 for line in lines[:12] if LOW_VALUE_SECTION_RE.search(line))
    has_praise_language = bool(ENDORSEMENT_PRAISE_RE.search(probe))
    has_preface_language = bool(PREFACE_STYLE_RE.search(probe))
    if has_preface_language and len(header_path) <= 1:
        return "preface"
    if credential_hits >= 1 and has_praise_language and len(header_path) <= 1:
        return "endorsement"
    if credential_hits >= 2 and title_hits >= 1:
        return "endorsement"
    if credential_hits >= 3:
        return "endorsement"
    if title_hits >= 2:
        return "preface"
    return "main"


def looks_like_caption_noise(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    caption_hits = 0
    credit_hits = 0
    short_hits = 0
    for line in lines:
        if CAPTION_LINE_RE.search(line) or CAPTION_INDEX_RE.match(line):
            caption_hits += 1
        if PHOTO_CREDIT_RE.search(line):
            credit_hits += 1
        if len(line) <= 80:
            short_hits += 1

    if caption_hits >= 2 and short_hits >= max(2, len(lines) // 2):
        return True
    if caption_hits >= 1 and credit_hits >= 1 and short_hits >= max(2, len(lines) // 2):
        return True
    if caption_hits >= 3:
        return True

    lowered = text.lower()
    if "photo credit" in lowered and caption_hits >= 1:
        return True
    return False


def looks_like_reference_chunk(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    source_like = sum(1 for line in lines if is_source_like_line(line))
    short_like = sum(1 for line in lines if len(line) <= 60)
    if source_like >= max(2, int(len(lines) * 0.35)):
        return True
    if source_like >= 2 and short_like >= max(3, int(len(lines) * 0.5)):
        return True
    return False


def looks_like_table_row(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if TABLE_LINE_RE.match(s):
        return True
    if len(s) <= 60 and len(re.findall(r"\s{2,}", s)) >= 1:
        return True
    return False


def looks_like_table_candidate(lines: List[str]) -> bool:
    if len(lines) < 2:
        return False
    effective = [x.strip() for x in lines if x.strip()]
    if len(effective) < 2:
        return False
    short_lines = [x for x in effective if len(x) <= 40]
    row_like = [x for x in effective if looks_like_table_row(x)]
    return (
        len(short_lines) / len(effective) >= 0.8 or
        len(row_like) / len(effective) >= 0.6
    )


def remove_repeated_headers_footers(lines: List[str]) -> List[str]:
    freq = {}
    for line in lines:
        s = line.strip()
        if 0 < len(s) <= 30:
            freq[s] = freq.get(s, 0) + 1

    common_noise = {
        k for k, v in freq.items()
        if v >= 3 and (
            is_page_noise(k) or
            looks_like_toc_line(k) or
            len(k) <= 12
        )
    }

    return [line for line in lines if line.strip() not in common_noise]


def merge_broken_lines(lines: List[str]) -> List[str]:
    merged = []
    buffer = ""

    for raw in lines:
        line = raw.strip()

        if not line:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append("")
            continue

        if is_page_noise(line):
            continue

        if is_header_like(line) or LIST_LINE_RE.match(line) or TABLE_LINE_RE.match(line):
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(line)
            continue

        if not buffer:
            buffer = line
            continue

        prev_end_ok = bool(re.search(r"[。！？!?；;：:]$|[)）】]$", buffer))
        curr_is_new_block = bool(is_header_like(line) or LIST_LINE_RE.match(line))

        if (not prev_end_ok) and (not curr_is_new_block):
            buffer += line
        else:
            merged.append(buffer.strip())
            buffer = line

    if buffer:
        merged.append(buffer.strip())

    return merged


def repair_table_like_blocks(lines: List[str]) -> List[str]:
    repaired = []
    candidate = []

    def flush_candidate():
        nonlocal candidate
        if not candidate:
            return
        if looks_like_table_candidate(candidate):
            repaired.append("\n".join(candidate))
        else:
            repaired.extend(candidate)
        candidate = []

    for line in lines:
        s = line.strip()
        if not s:
            flush_candidate()
            repaired.append("")
            continue

        if looks_like_table_row(s) and not is_header_like(s):
            candidate.append(s)
        else:
            flush_candidate()
            repaired.append(s)

    flush_candidate()
    return repaired


# =========================
# 6. Parse and classify
# =========================
def parse_markdown_sections(text: str, allow_implicit_headers: bool = True) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    sections = []
    header_stack = []
    current_content = []

    def flush_current():
        nonlocal current_content
        content = "\n".join(current_content).strip()
        if content:
            sections.append({"header_path": [h["text"] for h in header_stack], "content": content})
        current_content = []

    prev_was_blank = True
    for idx, line in enumerate(lines):
        stripped = line.strip()
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        match = HEADER_RE.match(stripped)
        implicit_match = allow_implicit_headers and not match and is_implicit_header_line(stripped, next_line, prev_was_blank)
        if match or implicit_match:
            flush_current()
            if match:
                level = len(match.group(1))
                title, inline_body = split_header_title_and_inline_body(match.group(2).strip())
            else:
                level = 2
                title, inline_body = split_header_title_and_inline_body(stripped)
            if not title:
                prev_was_blank = not bool(stripped)
                continue
            while header_stack and header_stack[-1]["level"] >= level:
                header_stack.pop()
            header_stack.append({"level": level, "text": title})
            if inline_body:
                current_content.append(inline_body)
        else:
            current_content.append(line)
        prev_was_blank = not bool(stripped)

    flush_current()
    if not sections and text.strip():
        sections.append({"header_path": [], "content": text.strip()})
    return sections


def split_paragraphs(text: str) -> List[str]:
    paragraphs = PARAGRAPH_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?])\s*|(?<=[.])\s+(?=[\"'(\[\u4e00-\u9fffA-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


@lru_cache(maxsize=4096)
def split_sentences_fast(text: str) -> Tuple[str, ...]:
    text = text.strip()
    if not text:
        return ()

    protected = text
    protected = protected.replace("...", "…")
    protected = re.sub(r"(?<=\d)\.(?=\d)", "§", protected)
    protected = MULTI_INITIAL_RE.sub(lambda m: m.group(0).replace(".", "§"), protected)
    protected = ABBREV_PERIOD_RE.sub(lambda m: m.group(0)[:-1] + "§", protected)
    protected = re.sub(r"\b(?:U\.S|U\.K|E\.U|P\.E|B\.V|C\.I\.P)\.", lambda m: m.group(0).replace(".", "§"), protected, flags=re.IGNORECASE)

    parts = re.split(r"(?<=[。！？!?])\s*|(?<=[.?!])\s+(?=[\"'(\[\u4e00-\u9fffA-Z0-9])", protected)
    restored = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        s = s.replace("§", ".").replace("…", "...")
        restored.append(s)
    return tuple(restored)


def strip_empty_visual_prefix(sentence: str) -> str:
    original = clean_text_fast(sentence or "")
    if not original:
        return ""
    stripped = EMPTY_PAGE_PREFIX_RE.sub("", original, count=1)
    stripped = re.sub(r"^(?:根据)?(?:图表|图中|图例|页面)(?:显示|展示|呈现)(?:的|，|:|：)?\s*", "", stripped)
    stripped = re.sub(r"[（(](?:蓝色|绿色|紫色|红色|黄色|橙色|黑色|灰色)[^）)]{0,18}[）)]", "", stripped)
    stripped = re.sub(r"(?:图例显示|图中图例|页面(?:左|右|上|下)?(?:侧)?(?:展示|显示))[^。！？；;]{0,120}[。！？；;]?", "", stripped)
    stripped = re.sub(r"(?:左侧|右侧)?纵轴[^。！？；;]{0,80}[。！？；;]?", "", stripped)
    stripped = re.sub(r"横轴[^。！？；;]{0,80}[。！？；;]?", "", stripped)
    stripped = re.sub(r"(?:页面)?(?:左下角|右下角|左上角|右上角|底部|顶部)[^。！？；;]{0,100}(?:页码|标语|字样|logo|Logo|标识|水印|二维码|显示)[^。！？；;]{0,80}[。！？；;]?", "", stripped)
    stripped = re.sub(r"(?:蓝色|绿色|紫色|红色|黄色|橙色|黑色|灰色)(?:线条|曲线|折线|柱状|色块)", "", stripped)
    stripped = MULTISPACE_RE.sub(" ", stripped).strip()
    stripped = stripped.lstrip("：:，,。；;、-— \t")
    return stripped if len(stripped) >= 4 else original


def sentence_information_score(sentence: str) -> float:
    s = clean_text_fast(sentence or "")
    if not s:
        return 0.0

    score = 0.06
    number_hits = len(NUMBER_VALUE_RE.findall(s))
    digit_hits = len(re.findall(r"\d", s))
    if number_hits:
        score += min(0.32, 0.12 + number_hits * 0.06)
    elif digit_hits:
        score += min(0.18, 0.04 + digit_hits * 0.01)
    if HIGH_INFO_RE.search(s):
        score += 0.18
    if DEFINITION_SIGNAL_RE.search(s):
        score += 0.16
    if CONCLUSION_SIGNAL_RE.search(s):
        score += 0.12
    if CAUSAL_SIGNAL_RE.search(s):
        score += 0.12
    if COMPARISON_SIGNAL_RE.search(s):
        score += 0.12
    if POLICY_SIGNAL_RE.search(s):
        score += 0.14
    if re.search(r"(?:M2\s*/\s*GDP|GDP|CAGR|同比|环比|百分点|%)", s, re.IGNORECASE):
        score += 0.12
    if 24 <= len(s) <= 220:
        score += 0.06
    if len(s) > 320:
        score -= 0.08

    visual_only = LAYOUT_VISUAL_RE.search(s) and not (number_hits or HIGH_INFO_RE.search(s))
    if visual_only:
        score -= 0.22
    if PROMOTIONAL_TEXT_RE.search(s):
        score -= 0.35
    if DECORATIVE_SENTENCE_RE.search(s) and not number_hits:
        score -= 0.18
    return clamp01(score)


def sentence_noise_score(sentence: str) -> float:
    s = clean_text_fast(sentence or "")
    if not s:
        return 1.0

    score = 0.0
    if DECORATIVE_SENTENCE_RE.search(s):
        score += 0.45
    if LAYOUT_VISUAL_RE.search(s):
        score += 0.25
    if PROMOTIONAL_TEXT_RE.search(s):
        score += 0.55
    if looks_like_toc_line(s):
        score += 0.30
    if is_page_noise(s):
        score += 0.45
    if looks_like_caption_noise(s) or looks_like_publication_metadata(s):
        score += 0.35
    if is_low_information_text(s):
        score += 0.20

    info_score = sentence_information_score(s)
    if info_score >= 0.45:
        score -= 0.25
    elif info_score >= 0.28:
        score -= 0.12
    return clamp01(score)


def is_decorative_sentence(sentence: str) -> bool:
    s = clean_text_fast(sentence or "")
    if not s:
        return True
    if PROMOTIONAL_TEXT_RE.search(s):
        return True
    info_score = sentence_information_score(s)
    if DECORATIVE_SENTENCE_RE.search(s) and info_score < 0.28:
        return True
    if LAYOUT_VISUAL_RE.search(s) and info_score < 0.22 and len(s) <= 180:
        return True
    if re.fullmatch(r"(?:服务大众，情系民生|页码(?:显示)?为?\d+|第\s*\d+\s*页|\d+)", s):
        return True
    return False


def classify_knowledge_unit_type(text: str, block_type: str = "paragraph") -> str:
    s = clean_text_fast(text or "")
    if not s:
        return "empty"
    if block_type == "table":
        return "table"
    if block_type == "list":
        return "list"
    if DEFINITION_SIGNAL_RE.search(s):
        return "definition"
    if POLICY_SIGNAL_RE.search(s):
        return "policy"
    if CAUSAL_SIGNAL_RE.search(s):
        return "causal"
    if COMPARISON_SIGNAL_RE.search(s) and NUMBER_VALUE_RE.search(s):
        return "comparison"
    if NUMBER_VALUE_RE.search(s):
        return "numeric_fact"
    if CONCLUSION_SIGNAL_RE.search(s):
        return "conclusion"
    if LAYOUT_VISUAL_RE.search(s) and sentence_information_score(s) < 0.25:
        return "visual_noise"
    if sentence_information_score(s) < 0.16:
        return "low_value"
    return "body"


def compute_info_density(text: str, block_type: str = "paragraph") -> float:
    clean = clean_text_fast(strip_html_tags(text or ""))
    if not clean:
        return 0.0
    if block_type == "table":
        base = 0.38
        if NUMBER_VALUE_RE.search(clean):
            base += 0.18
        if HIGH_INFO_RE.search(clean):
            base += 0.10
        if len(clean) >= 120:
            base += 0.08
        return clamp01(base - sentence_noise_score(clean) * 0.18)

    sentences = [strip_empty_visual_prefix(s) for s in split_sentences_fast(clean)]
    sentences = [s for s in sentences if s]
    if not sentences:
        sentences = [line.strip() for line in clean.splitlines() if line.strip()]
    if not sentences:
        return 0.0

    scores = [sentence_information_score(s) for s in sentences]
    high_value_count = sum(1 for score in scores if score >= 0.35)
    numeric_count = sum(1 for s in sentences if NUMBER_VALUE_RE.search(s))
    avg_score = sum(scores) / max(len(scores), 1)
    density = avg_score * 0.72
    density += min(0.18, high_value_count / max(len(sentences), 1) * 0.22)
    density += min(0.14, numeric_count / max(len(sentences), 1) * 0.18)
    if 70 <= len(clean) <= 650:
        density += 0.08
    if not numeric_count and not HIGH_INFO_RE.search(clean):
        density -= 0.10
    return clamp01(density)


def compute_noise_score(text: str, section_kind: str = "", block_type: str = "paragraph") -> float:
    clean = clean_text_fast(strip_html_tags(text or ""))
    if not clean:
        return 1.0
    units = list(split_sentences_fast(clean)) or [line.strip() for line in clean.splitlines() if line.strip()]
    units = [unit for unit in units if unit]
    if not units:
        return 1.0

    avg_noise = sum(sentence_noise_score(unit) for unit in units) / max(len(units), 1)
    visual_ratio = sum(1 for unit in units if LAYOUT_VISUAL_RE.search(unit)) / max(len(units), 1)
    decorative_ratio = sum(1 for unit in units if is_decorative_sentence(unit)) / max(len(units), 1)
    score = avg_noise * 0.70 + visual_ratio * 0.18 + decorative_ratio * 0.22
    if section_kind in {"cover", "catalog", "toc", "member_list", "promo_page"}:
        score += 0.18
    if block_type == "table":
        score -= 0.12
    return clamp01(score)


def _keyword_signature(text: str) -> set[str]:
    normalized = MULTISPACE_RE.sub(" ", clean_text_fast(text or "")).lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9./%+-]{1,}", normalized, re.IGNORECASE))
    numbers = set(re.findall(r"\d+(?:\.\d+)?%?", normalized))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_bigrams = {"".join(cjk_chars[i:i + 2]) for i in range(max(0, len(cjk_chars) - 1))}
    return {token for token in words | numbers | cjk_bigrams if token.strip()}


def validate_summary_consistency(summary: str, text: str) -> float:
    summary_clean = MULTISPACE_RE.sub(" ", clean_text_fast(summary or "")).strip()
    text_clean = MULTISPACE_RE.sub(" ", clean_text_fast(text or "")).strip()
    if not summary_clean or not text_clean:
        return 0.0
    if summary_clean in text_clean or text_clean.startswith(summary_clean):
        return 1.0

    summary_terms = _keyword_signature(summary_clean)
    text_terms = _keyword_signature(text_clean)
    if not summary_terms:
        return 0.0
    overlap = len(summary_terms & text_terms) / max(len(summary_terms), 1)
    summary_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", summary_clean))
    text_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", text_clean))
    if summary_numbers and not (summary_numbers & text_numbers):
        overlap -= 0.25
    return clamp01(overlap)


def collect_page_consistency_flags(item: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    header_path = item.get("header_path", []) or []
    section_title = str(item.get("section_title") or "")
    candidates = [str(value).strip() for value in list(header_path) + [section_title] if str(value).strip()]
    page_values: List[int] = []
    for candidate in candidates:
        match = re.fullmatch(r"(?:第\s*([0-9一二三四五六七八九十百零〇两]+)\s*页|page\s*([0-9]+))", candidate, re.IGNORECASE)
        if match:
            page_no = parse_simple_page_number(match.group(1) or match.group(2) or "")
            if page_no is not None:
                page_values.append(page_no)
    if len(set(page_values)) > 1:
        flags.append("page_metadata_mismatch")

    current_page = item.get("page_no")
    if isinstance(current_page, int) and page_values and current_page not in set(page_values):
        flags.append("page_metadata_mismatch")
    if is_clean_text_item(item) and str(item.get("chunk_level", "child")) == "child" and not page_values and not item.get("page_label"):
        flags.append("page_missing")
    return sorted(set(flags))


def clean_knowledge_text(text: str, source_profile: str = "clean_text") -> str:
    clean = clean_text_fast(text or "")
    if source_profile != "clean_text" or not clean:
        return clean

    kept: List[str] = []
    for paragraph in split_paragraphs(clean) or [clean]:
        sentences = list(split_sentences_fast(paragraph))
        if not sentences:
            sentences = [line.strip() for line in paragraph.splitlines() if line.strip()]
        paragraph_parts: List[str] = []
        for sentence in sentences:
            compact = strip_empty_visual_prefix(sentence)
            if not compact or is_decorative_sentence(compact):
                continue
            paragraph_parts.append(compact)
        if paragraph_parts:
            kept.append(" ".join(paragraph_parts).strip())
    return "\n\n".join(part for part in kept if part).strip()


def split_knowledge_units(text: str, target_chars: int, max_chars: int) -> List[Dict[str, str]]:
    cleaned = clean_knowledge_text(text, source_profile="clean_text")
    if not cleaned:
        return []

    sentences = list(split_sentences_fast(cleaned))
    if len(sentences) <= 1:
        sentences = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not sentences:
        return []

    high_value_types = {"definition", "policy", "causal", "comparison", "numeric_fact", "conclusion"}
    units: List[Dict[str, str]] = []
    current_parts: List[str] = []
    current_type = "body"

    def current_text() -> str:
        return " ".join(current_parts).strip()

    def flush() -> None:
        nonlocal current_parts, current_type
        unit_text = current_text()
        if unit_text:
            unit_type = classify_knowledge_unit_type(unit_text, "paragraph")
            units.append({
                "block_type": "paragraph",
                "text": unit_text,
                "knowledge_unit_type": unit_type if unit_type != "low_value" else current_type,
            })
        current_parts = []
        current_type = "body"

    for sentence in sentences:
        sentence = strip_empty_visual_prefix(sentence)
        if not sentence or is_decorative_sentence(sentence):
            continue
        unit_type = classify_knowledge_unit_type(sentence, "paragraph")
        high_value = unit_type in high_value_types or sentence_information_score(sentence) >= 0.42

        if len(sentence) > max_chars:
            flush()
            for piece in split_long_text_by_sentence(sentence, max_chars):
                if piece:
                    units.append({
                        "block_type": "paragraph",
                        "text": piece,
                        "knowledge_unit_type": classify_knowledge_unit_type(piece, "paragraph"),
                    })
            continue

        candidate = (current_text() + " " + sentence).strip() if current_parts else sentence
        type_shift = current_parts and current_type not in {"body", unit_type} and unit_type in high_value_types
        if current_parts and (len(candidate) > max_chars or (high_value and len(current_text()) >= 120) or (type_shift and len(current_text()) >= 80)):
            flush()

        current_parts.append(sentence)
        if current_type == "body" or unit_type in high_value_types:
            current_type = unit_type

        if high_value and len(current_text()) >= min(target_chars, 260):
            flush()
        elif len(current_text()) >= target_chars:
            flush()

    flush()
    return [unit for unit in units if unit["text"].strip()]


def classify_block(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"

    if is_reference_noise_line(stripped):
        return "reference"
    if contains_ocr_placeholder(stripped):
        return "reference"
    if looks_like_publication_metadata(stripped):
        return "reference"

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return "empty"

    if sum(1 for line in lines if is_reference_noise_line(line)) / len(lines) >= 0.5:
        return "reference"

    if len(lines) >= 2 and FAQ_QUESTION_RE.match(lines[0]):
        return "faq"

    if all(TABLE_LINE_RE.match(line) for line in lines) and len(lines) >= 2:
        return "table"

    if sum(1 for line in lines if looks_like_pipe_table_line(line)) >= max(2, int(len(lines) * 0.5)):
        return "table"

    if looks_like_table_candidate(lines):
        return "table"

    list_hits = sum(1 for line in lines if LIST_LINE_RE.match(line))
    if list_hits >= max(2, len(lines) - 1):
        return "list"

    if len(lines) >= 3 and list_hits >= max(2, int(len(lines) * 0.6)):
        return "list"

    return "paragraph"


def split_mixed_paragraph_block(text: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    groups: List[Dict[str, str]] = []
    current_lines: List[str] = []
    current_type = None

    def flush():
        nonlocal current_lines, current_type
        if current_lines:
            groups.append({"block_type": current_type or "paragraph", "text": "\n".join(current_lines).strip()})
        current_lines = []
        current_type = None

    for line in lines:
        if HEADING_ONLY_RE.match(line) or is_header_like(line):
            line_type = "heading"
        elif is_reference_noise_line(line):
            line_type = "reference"
        elif looks_like_pipe_table_line(line) or TABLE_LINE_RE.match(line):
            line_type = "table"
        else:
            line_type = "paragraph"

        if current_type is None:
            current_type = line_type
            current_lines = [line]
            continue

        if line_type == current_type:
            current_lines.append(line)
        else:
            flush()
            current_type = line_type
            current_lines = [line]

    flush()
    return groups


def infer_semantic_role(item: Dict[str, Any], table_meta: Dict[str, Any] | None = None) -> str:
    chunk_type = item.get("chunk_type", "paragraph")
    raw_text = (item.get("raw_text") or item.get("text") or "").strip()
    section_title = (item.get("section_title") or "").strip()
    header_path = item.get("header_path", [])
    knowledge_unit_type = str(item.get("knowledge_unit_type") or "").strip()
    table_meta = table_meta or get_empty_table_meta()
    family = table_meta.get("family", "")

    if chunk_type == "table":
        if family in {"parameter_table", "metric_table", "scenario_matrix", "policy_matrix", "wide_stats_table", "timeline_table", "distribution_table"}:
            return family
        if table_meta.get("specs"):
            return "field_table"
        if len(table_meta.get("headers", [])) >= 4 and len(table_meta.get("rows", [])) >= 3:
            return "table_rows"
        return "table"

    unit_role_map = {
        "definition": "definition",
        "conclusion": "conclusion",
        "policy": "recommendation",
        "causal": "causal",
        "comparison": "comparison",
        "numeric_fact": "metric_fact",
        "visual_noise": "front_matter",
        "low_value": "front_matter",
    }
    if knowledge_unit_type in unit_role_map:
        return unit_role_map[knowledge_unit_type]

    lowered = raw_text.lower()
    if looks_like_front_matter_noise(raw_text):
        return "front_matter"
    if any(token in lowered for token in ["warning", "caution", "注意", "警告", "风险", "勿", "不要", "禁止"]):
        return "warning"
    if re.search(r"(步骤|step\s*\d+|首先|其次|然后|最后|一、|二、|1[.)、]|2[.)、]|3[.)、])", raw_text, re.IGNORECASE):
        return "steps"
    if re.search(r"(定义|指的是|意味着|means|refers to|i\.e\.)", raw_text, re.IGNORECASE):
        return "definition"
    if re.search(r"(例如|比如|举例|for example|e\.g\.)", raw_text, re.IGNORECASE):
        return "example"
    if re.search(r"(建议|recommend|should|must|因此|综上|结论|conclusion|takeaway)", raw_text, re.IGNORECASE):
        return "recommendation"
    if re.search(r"(总结|小结|结论|summary)", raw_text, re.IGNORECASE):
        return "conclusion"
    if CAUSAL_SIGNAL_RE.search(raw_text):
        return "causal"
    if COMPARISON_SIGNAL_RE.search(raw_text):
        return "comparison"
    if NUMBER_VALUE_RE.search(raw_text):
        return "metric_fact"
    if len(raw_text) <= 120 and (is_header_like(section_title) or any(is_header_like(part) for part in header_path)):
        return "intro"
    if sum(1 for line in raw_text.splitlines() if LIST_LINE_RE.match(line.strip())) >= 2:
        return "list"
    return "body"


def compute_answerability_score(item: Dict[str, Any], table_meta: Dict[str, Any] | None = None, semantic_role: str = "") -> float:
    chunk_type = item.get("chunk_type", "paragraph")
    chunk_level = str(item.get("chunk_level") or "child")
    raw_text = (item.get("raw_text") or "").strip()
    section_kind = str(item.get("section_kind") or "")
    info_density = float(item.get("info_density", 0.0) or 0.0)
    noise_score = float(item.get("noise_score", 0.0) or 0.0)
    table_meta = table_meta or get_empty_table_meta()
    family = table_meta.get("family", "")
    length = len(raw_text)
    score = 0.18

    if chunk_type == "table":
        row_count = len(table_meta.get("rows", []))
        header_count = len(table_meta.get("headers", []))
        spec_count = len(table_meta.get("specs", []))
        if family in {"parameter_table", "metric_table", "scenario_matrix", "policy_matrix", "distribution_table"}:
            score += 0.45
        elif family == "wide_stats_table":
            score += 0.32
        else:
            score += 0.15
        if header_count >= 3:
            score += 0.12
        if 2 <= row_count <= 8:
            score += 0.12
        if spec_count:
            score += 0.12
        if row_count <= 1:
            score -= 0.16
        if header_count <= 1:
            score -= 0.12
        if row_count <= 2 and header_count >= 4:
            score -= 0.08
        if family == "generic_table":
            score -= 0.08
        if any(str(h).startswith("col_") for h in table_meta.get("headers", [])):
            score -= 0.08
    else:
        sentences = list(split_sentences_fast(raw_text))
        punct_count = len(PUNCT_COUNT_RE.findall(raw_text))
        if chunk_level == "parent":
            score = 0.28
            if 220 <= length <= 1200:
                score += 0.24
            elif 120 <= length < 220:
                score += 0.14
            elif length > 1400:
                score -= 0.08
            if len(sentences) >= 2:
                score += 0.08
            if len(sentences) >= 4:
                score += 0.04
            if punct_count >= 2:
                score += 0.05
        else:
            if 180 <= length <= 800:
                score += 0.25
            elif 90 <= length < 180:
                score += 0.12
            elif length > 1000:
                score -= 0.14
            if len(sentences) >= 2:
                score += 0.1
            if len(sentences) >= 4:
                score += 0.04
            if punct_count >= 2:
                score += 0.05
        if semantic_role in {"definition", "steps", "warning", "example", "recommendation", "conclusion", "metric_fact", "comparison", "causal"}:
            score += 0.18 if chunk_level != "parent" else 0.12
        if semantic_role in {"metric_fact", "comparison", "causal"} and NUMBER_VALUE_RE.search(raw_text):
            score += 0.08 if chunk_level != "parent" else 0.05
        if semantic_role == "front_matter":
            score -= 0.2
        if chunk_level != "parent":
            if len(raw_text) > 700 and len(sentences) <= 2:
                score -= 0.16
            if len(raw_text) > 900:
                score -= 0.14
        else:
            if len(raw_text) > 1400 and len(sentences) <= 2:
                score -= 0.08
    if looks_like_caption_noise(raw_text) or looks_like_reference_chunk(raw_text):
        score -= 0.28
    if contains_ocr_placeholder(raw_text):
        score -= 0.35
    if looks_like_publication_metadata(raw_text):
        score -= 0.28
    if is_low_information_text(raw_text):
        score -= 0.25
    if "mixed_heading" in item.get("quality_flags", []):
        score -= 0.12
    if "fragmented_text" in item.get("quality_flags", []):
        score -= 0.16
    if "truncated_boundary" in item.get("quality_flags", []):
        score -= 0.18
    score += info_density * (0.24 if chunk_level != "parent" else 0.16)
    score -= noise_score * (0.26 if chunk_level != "parent" else 0.16)
    if section_kind in {"cover", "catalog", "toc", "member_list", "promo_page"}:
        score -= 0.18 if chunk_level != "parent" else 0.10
    return max(0.0, min(1.0, score))


# =========================
# 7. Split blocks
# =========================
def split_long_text_by_sentence(text: str, max_chars: int) -> List[str]:
    sentences = list(split_sentences_fast(text))
    if len(sentences) <= 1:
        return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i:i + max_chars].strip()]

    chunks = []
    current = ""
    for sent in sentences:
        candidate = sent if not current else current + " " + sent
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(sent) <= max_chars:
            current = sent
        else:
            hard_splits = [sent[i:i + max_chars].strip() for i in range(0, len(sent), max_chars) if sent[i:i + max_chars].strip()]
            chunks.extend(hard_splits)
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def split_long_text_by_soft_boundary(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    parts = [part.strip() for part in SOFT_BOUNDARY_SPLIT_RE.split(text) if part.strip()]
    if len(parts) <= 1:
        return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i:i + max_chars].strip()]

    chunks = []
    current = ""
    for part in parts:
        candidate = part if not current else current + " " + part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(part) <= max_chars:
            current = part
        else:
            chunks.extend([part[i:i + max_chars].strip() for i in range(0, len(part), max_chars) if part[i:i + max_chars].strip()])
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def split_front_matter_text(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return split_long_text_by_soft_boundary(text, max_chars)

    chunks: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append("\n".join(current).strip())
            current = []

    for line in lines:
        boundary_hit = bool(re.match(r"^(?:#{1,6}\s*)?[●•·☑☒■□]\s*", line))
        boundary_hit = boundary_hit or bool(re.match(r"^(?:推荐语|推荐|前言|序言|作者|主理人|创始人|董事长|CEO|总经理|合伙人)[:：]?", line))
        boundary_hit = boundary_hit or bool(re.match(r"^[#]{1,6}\s*[^。！？!?]{2,30}$", line))
        if boundary_hit and current:
            flush_current()
        current.append(line)
        if len("\n".join(current)) >= max_chars:
            flush_current()

    flush_current()
    if not chunks:
        return split_long_text_by_soft_boundary(text, max_chars)

    normalized: List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            normalized.append(chunk)
        else:
            normalized.extend(split_long_text_by_soft_boundary(chunk, max_chars))
    return [chunk for chunk in normalized if chunk.strip()]


def split_dense_text(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    front_matter_style = (
        looks_like_front_matter_noise(text)
        and (
            re.search(r"(?m)^(?:#{1,6}\s*)?[●•·☑☒■□]\s*", text)
            or re.search(r"(?m)^(?:推荐语|推荐|前言|序言|作者|主理人|创始人|董事长|CEO|总经理|合伙人)[:：]?", text)
            or text.count("##") >= 2
        )
    )
    if front_matter_style:
        primary = split_front_matter_text(text, max(180, min(max_chars, 260)))
    else:
        primary = split_long_text_by_sentence(text, max_chars)
    final_chunks = []

    for chunk in primary:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk.strip())
            continue

        parts = re.split(r"(?<=[；;：:])\s*|(?<=\))\s+|(?<=\])\s+|(?<=[|])\s*", chunk)
        current = ""
        for part in [p.strip() for p in parts if p.strip()]:
            candidate = part if not current else current + " " + part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    final_chunks.append(current.strip())
                if len(part) <= max_chars:
                    current = part
                else:
                    final_chunks.extend(
                        [part[i:i + max_chars].strip() for i in range(0, len(part), max_chars) if part[i:i + max_chars].strip()]
                    )
                    current = ""
        if current:
            final_chunks.append(current.strip())

    return [chunk for chunk in final_chunks if chunk.strip()]


def split_table_block(text: str, target_chars: int = TARGET_CHUNK_CHARS, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= TABLE_BLOCK_SOFT_LIMIT and len(text) <= max_chars:
        return [text.strip()]
    chunks = []
    header_lines = lines[:1]
    current = list(header_lines)
    for line in lines[1:]:
        candidate = "\n".join(current + [line]).strip()
        if len(current) > len(header_lines) and len(candidate) > target_chars:
            chunks.append("\n".join(current).strip())
            current = list(header_lines) + [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def split_list_block(text: str, target_chars: int = TARGET_CHUNK_CHARS) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= LIST_BLOCK_SOFT_LIMIT:
        return [text.strip()]
    chunks = []
    current = []
    for line in lines:
        candidate = "\n".join(current + [line]).strip()
        if current and len(candidate) > target_chars:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def split_faq_block(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        return split_long_text_by_sentence(lines[0], MAX_CHUNK_CHARS)

    question = lines[0]
    answer = "\n".join(lines[1:]).strip()
    if len(question) <= FAQ_QUESTION_SOFT_LIMIT and len(text) <= MAX_CHUNK_CHARS:
        return [text.strip()]

    answer_chunks = split_long_text_by_sentence(answer, max(MAX_CHUNK_CHARS - len(question), 320))
    return [f"{question}\n{chunk}".strip() for chunk in answer_chunks if chunk.strip()]


def split_block_by_type(text: str, block_type: str, target_chars: int = TARGET_CHUNK_CHARS, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    if block_type == "table":
        chunks = split_table_block(text, target_chars=target_chars, max_chars=max_chars)
    elif block_type == "list":
        chunks = split_list_block(text, target_chars=target_chars)
    elif block_type == "faq":
        chunks = split_faq_block(text)
    else:
        chunks = split_dense_text(text, max_chars)

    normalized = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            normalized.append(chunk.strip())
        else:
            normalized.extend(split_long_text_by_sentence(chunk, max_chars))
    return [chunk for chunk in normalized if chunk.strip()]


def get_document_chunk_profile(text_length: int, source_profile: str = "clean_text") -> Dict[str, int]:
    if source_profile == "clean_text":
        if text_length >= 90000:
            return {"target_chars": 420, "max_chars": 620, "min_raw_chars": 220}
        if text_length >= 50000:
            return {"target_chars": 460, "max_chars": 660, "min_raw_chars": 240}
        if text_length >= 20000:
            return {"target_chars": 500, "max_chars": 700, "min_raw_chars": 240}
        return {"target_chars": 540, "max_chars": 700, "min_raw_chars": 260}
    if text_length >= 90000:
        return {"target_chars": 360, "max_chars": 540, "min_raw_chars": 180}
    if text_length >= 50000:
        return {"target_chars": 400, "max_chars": 620, "min_raw_chars": 200}
    if text_length >= 20000:
        return {"target_chars": 440, "max_chars": 680, "min_raw_chars": 220}
    return {"target_chars": 480, "max_chars": MAX_CHUNK_CHARS, "min_raw_chars": 240}


def parse_table_chunk(text: str) -> Dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"headers": [], "rows": [], "specs": [], "family": "generic_table"}

    parsed_rows = []
    for line in lines:
        if not looks_like_pipe_table_line(line):
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) >= 2:
            expanded_rows = expand_compound_pipe_cells(cells)
            if expanded_rows and expanded_rows != [cells]:
                parsed_rows.extend(expanded_rows)
            else:
                parsed_rows.append(cells)

    if not parsed_rows:
        return {"headers": [], "rows": [], "specs": [], "family": "generic_table"}

    if text_looks_like_parameter_table(text):
        rows = [[normalize_table_cell_text(cell) for cell in row if normalize_table_cell_text(cell)] for row in parsed_rows]
        rows = [row for row in rows if len(row) >= 2]
        inferred_width = min(8, max(2, max(len(row) for row in rows)))
        headers = infer_parameter_headers(inferred_width)
        family = "parameter_table"
    else:
        headers, rows = resolve_table_headers_and_rows(parsed_rows)
        provisional_headers = repair_table_headers(headers, rows)
        family = detect_table_family(text, provisional_headers, rows)
        headers = normalize_headers_by_family(provisional_headers, family, rows)
    expected_cols = max(1, len(headers))
    repaired_rows = []

    for row in rows:
        if len(row) == expected_cols:
            repaired_rows.append(row)
            continue

        if len(row) > expected_cols:
            start = 0
            while start < len(row):
                chunk = row[start:start + expected_cols]
                if len(chunk) < expected_cols and repaired_rows:
                    repaired_rows[-1][-1] = (repaired_rows[-1][-1] + " " + " | ".join(chunk)).strip()
                else:
                    repaired_rows.append(chunk)
                start += expected_cols
            continue

        repaired_rows.append(row + [""] * (expected_cols - len(row)))

    repaired_rows = repair_parameter_table_rows(headers, repaired_rows)
    repaired_rows = repair_truncated_table_rows(headers, repaired_rows)
    if family == "metric_table":
        repaired_rows = repair_metric_table_rows(headers, repaired_rows)
    if family in {"wide_stats_table", "metric_table"}:
        repaired_rows = split_wide_stats_logical_rows(repaired_rows, headers)
        repaired_rows = repair_wide_stats_rows(headers, repaired_rows)
    normalized_rows = []

    for row in repaired_rows:
        row_dict = {}
        for idx, header in enumerate(headers):
            key = header if header else f"col_{idx + 1}"
            row_dict[key] = row[idx] if idx < len(row) else ""
        if len(row) > len(headers):
            row_dict["_extra"] = row[len(headers):]
        normalized_rows.append(row_dict)

    if family == "parameter_table" and looks_like_narrative_matrix(text):
        family = detect_table_family(text, headers, normalized_rows)
        headers = normalize_headers_by_family(headers, family, repaired_rows)
        normalized_rows = []
        for row in repaired_rows:
            row_dict = {}
            for idx, header in enumerate(headers):
                key = header if header else f"col_{idx + 1}"
                row_dict[key] = row[idx] if idx < len(row) else ""
            if len(row) > len(headers):
                row_dict["_extra"] = row[len(headers):]
            normalized_rows.append(row_dict)

    metric_specs = extract_metric_specs_from_rows(headers, normalized_rows) if family == "metric_table" else []
    specs = metric_specs if metric_specs else (extract_parameter_specs_from_text(text, headers) if family == "parameter_table" else [])
    return {"headers": headers, "rows": normalized_rows, "specs": specs, "family": family}


def normalize_table_cell_text(text: str) -> str:
    text = html.unescape(text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return re.sub(r"(?<=\d)\s+(?=\d)", "", text)


def looks_like_parameter_table(headers: List[str]) -> bool:
    joined = " ".join(headers).lower()
    keywords = ["hp", "pto", "变速", "液压", "价格", "发动机", "品牌", "型号", "lift", "flow"]
    hard_keywords = ["型号", "品牌", "model", "gearbox", "发动机", "price", "价格"]
    return sum(1 for kw in keywords if kw in joined) >= 2 and any(kw in joined for kw in hard_keywords)


def is_data_like_header_cell(text: str) -> bool:
    s = normalize_table_cell_text(text)
    if not s:
        return False
    if re.match(r"^(?:col|value)_\d+$", s, re.IGNORECASE):
        return False
    if len(s) >= 28:
        return True
    if HP_VALUE_RE.search(s) or GEARBOX_VALUE_RE.search(s):
        return True
    if PRICE_VALUE_RE.search(s) or FLOW_VALUE_RE.search(s) or LIFT_VALUE_RE.search(s):
        return True
    if any(ch.isalpha() for ch in s) and any(ch.isdigit() for ch in s) and len(s) >= 4:
        return True
    if LONG_NUMERIC_TOKEN_RE.search(s) and len(re.findall(r"\d", s)) >= 3:
        return True
    if "%" in s or "占 GDP" in s or "亿马币" in s:
        return True
    return False


def looks_like_header_row(cells: List[str]) -> bool:
    normalized = [normalize_table_cell_text(cell) for cell in cells if normalize_table_cell_text(cell)]
    if len(normalized) < 2:
        return False
    good = sum(1 for cell in normalized if not is_data_like_header_cell(cell))
    return good / len(normalized) >= 0.6


def infer_parameter_headers(width: int) -> List[str]:
    template = ["model", "engine_hp", "pto_hp", "gearbox", "hydraulic_flow", "lift_capacity", "price", "scenario"]
    if width <= len(template):
        return template[:width]
    return template + [f"attr_{idx + 1}" for idx in range(width - len(template))]


def text_looks_like_parameter_table(text: str) -> bool:
    lowered = text.lower()
    narrative_tokens = ["差异优势", "目标客户", "作业续航", "座舱差异", "项目内容说明", "产品定位", "适配场景"]
    if sum(1 for token in narrative_tokens if token in text) >= 2:
        return False
    hp_hits = len(HP_VALUE_RE.findall(text))
    gearbox_hits = len(GEARBOX_VALUE_RE.findall(text))
    price_hits = len(PRICE_VALUE_RE.findall(text))
    brand_hits = sum(1 for brand in ["kubota", "john deere", "new holland", "massey", "eurostar"] if brand in lowered)
    schema_hits = sum(1 for token in ["model", "型号", "品牌", "gearbox", "pto", "发动机"] if token in lowered)
    return (hp_hits >= 2 and (gearbox_hits >= 1 or brand_hits >= 1)) or (schema_hits >= 2 and hp_hits >= 1 and price_hits >= 1)


def looks_like_narrative_matrix(text: str) -> bool:
    tokens = ["差异优势", "目标客户", "作业续航", "座舱差异", "使用场景", "终端用户", "核心原因", "适配场景", "家庭农业户", "果园", "高尔夫球场"]
    return sum(1 for token in tokens if token in text) >= 2


def detect_table_family(text: str, headers: List[str], rows: List[Dict[str, Any]] | List[List[str]]) -> str:
    lowered = text.lower()
    header_joined = " ".join(str(h).lower() for h in headers)
    if looks_like_narrative_matrix(text):
        if any(token in header_joined for token in ["使用场景", "终端用户", "原因", "场景"]) or any(token in text for token in ["家庭农业户", "果园", "高尔夫球场", "终端用户"]):
            return "scenario_matrix"
        if any(token in header_joined for token in ["时间", "日期", "公司", "型号", "事件", "网址", "出处"]):
            return "timeline_table"
        if any(token in header_joined for token in ["项目", "内容", "成立", "资金", "补贴", "政策"]):
            return "policy_matrix"
        return "scenario_matrix"
    if text_looks_like_parameter_table(text) or looks_like_parameter_table(headers):
        return "parameter_table"
    if any(token in header_joined for token in ["使用场景", "target", "终端用户", "原因", "场景"]):
        return "scenario_matrix"
    if any(token in header_joined for token in ["时间", "日期", "公司", "型号", "事件", "网址", "出处"]):
        return "timeline_table"
    if any(token in lowered for token in ["官网/新闻稿", "年份日期/出处", "推出马来西亚首款", "区域市场报告", "本地主流媒体"]):
        return "timeline_table"
    if any(token in header_joined for token in ["指标", "占比", "gdp", "数据", "metric"]) or any(token in lowered for token in ["指标 |", "占 gdp", "占全国就业", "农业产值", "market share"]):
        return "metric_table"
    if any(token in header_joined for token in ["项目", "内容全称", "成立时间", "资金规模", "核心原因", "政策", "补贴"]):
        return "policy_matrix"
    if any(token in lowered for token in ["目标客户", "差异优势", "作业续航", "座舱差异", "使用场景", "终端用户", "果园", "高尔夫球场"]):
        return "scenario_matrix"
    if len(headers) >= 6 or len(rows) >= 8:
        return "wide_stats_table"
    if "农田类型" in text or "国家集中度" in text or "首选机械" in text:
        return "distribution_table"
    if "政策" in lowered or "incentive" in lowered or "tax relief" in lowered:
        return "policy_matrix"
    return "generic_table"


def normalize_distribution_headers(headers: List[str]) -> List[str]:
    normalized = []
    for idx, header in enumerate(headers):
        s = normalize_table_cell_text(header)
        if idx == 0 or s in {"☐", "序号", "编号"}:
            normalized.append("id")
        elif "农田类型" in s:
            normalized.append("crop_type")
        elif "国家集中度" in s or "区域" in s or "分布" in s:
            normalized.append("region")
        elif "首选机械" in s or "机械" in s:
            normalized.append("preferred_machinery")
        else:
            normalized.append(f"col_{idx + 1}")
    return normalized


def normalize_metric_headers(headers: List[str], width: int) -> List[str]:
    base = ["metric"] + [f"value_{idx}" for idx in range(1, max(2, width))]
    normalized = []
    for idx in range(width):
        if idx == 0:
            normalized.append("metric")
        elif idx < len(base):
            normalized.append(base[idx])
        else:
            normalized.append(f"value_{idx}")
    return normalized


def normalize_scenario_headers(headers: List[str]) -> List[str]:
    normalized = []
    for idx, header in enumerate(headers):
        s = normalize_table_cell_text(header)
        if idx == 0 or "使用场景" in s:
            normalized.append("scenario")
        elif "原因" in s:
            normalized.append("reason")
        elif "终端用户" in s or "客户" in s:
            normalized.append("target_user")
        elif "果园" in s or "示例" in s or "案例" in s:
            normalized.append("example")
        else:
            normalized.append("value_prop" if "value_prop" not in normalized else f"col_{idx + 1}")
    return normalized


def normalize_policy_headers(headers: List[str]) -> List[str]:
    normalized = []
    for idx, header in enumerate(headers):
        s = normalize_table_cell_text(header)
        if idx == 0 or "项目" in s:
            normalized.append("item")
        elif "内容" in s or "全称" in s:
            normalized.append("content")
        elif "时间" in s or "成立" in s:
            normalized.append("time_or_origin")
        elif "资金" in s or "规模" in s or "补贴" in s:
            normalized.append("support")
        elif "原因" in s or "目的" in s:
            normalized.append("reason")
        else:
            normalized.append(f"col_{idx + 1}")
    return normalized


def normalize_timeline_headers(headers: List[str]) -> List[str]:
    normalized = []
    for idx, header in enumerate(headers):
        s = normalize_table_cell_text(header)
        if idx == 0 or "时间" in s or "日期" in s:
            normalized.append("date")
        elif "公司" in s or "出处" in s:
            normalized.append("source")
        elif "型号" in s:
            normalized.append("model")
        elif "事件" in s or "标题" in s:
            normalized.append("event")
        elif "网址" in s or "链接" in s:
            normalized.append("link_or_note")
        else:
            normalized.append(f"col_{idx + 1}")
    return normalized


def normalize_wide_stats_headers(headers: List[str]) -> List[str]:
    normalized = []
    for idx, header in enumerate(headers):
        s = normalize_table_cell_text(header)
        if idx == 0 and ("metric" in s.lower() or "指标" in s or s.startswith("col_")):
            normalized.append("metric")
        elif re.match(r"^(?:value|col)_\d+$", s, re.IGNORECASE):
            normalized.append(s.lower())
        elif s and not s.startswith("col_") and len(s) <= 40:
            normalized.append(s)
        else:
            normalized.append(f"col_{idx + 1}")
    return normalized


def normalize_headers_by_family(headers: List[str], family: str, rows: List[List[str]]) -> List[str]:
    width = max(len(headers), max((len(row) for row in rows), default=0))
    if width <= 0:
        return []
    padded_headers = headers + [f"col_{idx + 1}" for idx in range(len(headers), width)]
    if family == "distribution_table":
        return normalize_distribution_headers(padded_headers[:width])
    if family == "metric_table":
        return normalize_metric_headers(padded_headers[:width], width)
    if family in {"scenario_matrix", "policy_matrix"}:
        if family == "policy_matrix":
            return normalize_policy_headers(padded_headers[:width])
        return normalize_scenario_headers(padded_headers[:width])
    if family == "timeline_table":
        return normalize_timeline_headers(padded_headers[:width])
    if family == "wide_stats_table":
        return normalize_wide_stats_headers(padded_headers[:width])
    return padded_headers[:width]


def looks_like_truncated_boundary(text: str) -> bool:
    s = normalize_table_cell_text(text)
    if not s:
        return False
    tail = s[-80:]
    if re.search(r"(https?://\S*$|www\.\S*$)", tail, re.IGNORECASE):
        return True
    if re.search(r"[\-\:/&+]\s*$", tail):
        return True
    if re.search(r"(?i)\b(?:and|or|of|to|in|for|with|the|a|an)$", tail):
        return True
    if tail.count("(") > tail.count(")") or tail.count("[") > tail.count("]"):
        return True
    if re.search(r"(\\frac|\\sum|\\int|\\begin|\\end|=|\\cdot|\\times)\s*$", tail):
        return True
    if re.search(r"\.\.\.\s*$", tail) or re.search(r"[A-Za-z0-9]\.$", tail):
        return True
    return False


def repair_truncated_table_rows(headers: List[str], rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    repaired: List[List[str]] = []
    for row in rows:
        current = list(row)
        if repaired:
            prev = repaired[-1]
            prev_tail = prev[-1].strip() if prev and prev[-1] else ""
            current_head = current[0].strip() if current else ""
            if prev_tail.endswith("-") and current_head and not re.match(r"^(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩])$", current_head):
                prev[-1] = (prev_tail + current_head).replace("  ", " ").strip()
                if len(current) > 1:
                    repaired.append([""] + current[1:])
                continue
            if "推荐25-" in prev_tail and current_head.lower().endswith("50hp)"):
                prev[-1] = prev_tail.rstrip("-") + "-50HP)"
                if len(current) > 1:
                    repaired.append([""] + current[1:])
                continue
        repaired.append(current)
    return repaired


def repair_wide_stats_rows(headers: List[str], rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    repaired = []
    for row in rows:
        current = [normalize_table_cell_text(cell) for cell in row]
        if current:
            first = current[0]
            digits_only = re.sub(r"\D", "", first)
            if 3 <= len(digits_only) <= 4:
                current[0] = digits_only
            elif len(digits_only) >= 7 and repaired:
                if len(current) <= 2:
                    repaired[-1][-1] = (repaired[-1][-1] + " " + first).strip()
                    current = current[1:]
                else:
                    current[0] = first
        if current:
            repaired.append(current)
    return repaired


def repair_metric_table_rows(headers: List[str], rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    repaired = []
    for row in rows:
        normalized = [normalize_metric_table_cell_text(cell) if idx > 0 else normalize_table_cell_text(cell) for idx, cell in enumerate(row)]
        expanded = expand_metric_row_recursive(normalized)
        for expanded_row in expanded:
            padded = expanded_row + [""] * max(0, len(row) - len(expanded_row))
            repaired.append(padded[:len(row)])
    return repaired


def is_wide_stats_label_cell(text: str) -> bool:
    s = normalize_table_cell_text(text)
    if not s or is_value_like_cell(s):
        return False
    if len(s) > 60:
        return False
    lowered = s.lower()
    keywords = [
        "metrics measured", "total", "reduction", "emissions", "consumption", "performance",
        "miles", "hours", "gallons", "kwh", "dge", "nox", "co2", "pm", "fuel"
    ]
    keyword_hits = sum(1 for kw in keywords if kw in lowered)
    if keyword_hits >= 1 and (len(s.split()) >= 2 or "(" in s or ")" in s):
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9\s,&'’\-/().%]{6,}$", s) and any(ch.isalpha() for ch in s):
        return True
    return False


def split_wide_stats_logical_rows(rows: List[List[str]], headers: List[str]) -> List[List[str]]:
    if not rows:
        return rows

    width = max(len(headers), max((len(row) for row in rows), default=0))
    if width < 4:
        return rows

    split_rows: List[List[str]] = []
    for row in rows:
        normalized = [normalize_table_cell_text(cell) for cell in row if normalize_table_cell_text(cell)]
        if len(normalized) <= width:
            split_rows.append(normalized)
            continue

        current: List[str] = []
        for idx, cell in enumerate(normalized):
            next_cell = normalized[idx + 1] if idx + 1 < len(normalized) else ""
            if (
                current
                and len(current) >= 2
                and is_wide_stats_label_cell(cell)
                and not is_value_like_cell(current[-1])
                and len(normalized) - idx >= 2
            ):
                split_rows.append(current)
                current = [cell]
                continue
            current.append(cell)
        if current:
            split_rows.append(current)

    return split_rows


def row_contains_schema_terms(cells: List[str]) -> bool:
    joined = " ".join(normalize_table_cell_text(cell).lower() for cell in cells if normalize_table_cell_text(cell))
    keywords = ["型号", "品牌", "model", "engine", "pto", "gearbox", "lift", "flow", "price", "价格", "液压", "变速"]
    return sum(1 for keyword in keywords if keyword in joined) >= 2


def resolve_table_headers_and_rows(parsed_rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    cleaned_rows = [[normalize_table_cell_text(cell) for cell in row if normalize_table_cell_text(cell)] for row in parsed_rows]
    cleaned_rows = [row for row in cleaned_rows if len(row) >= 2]
    if not cleaned_rows:
        return [], []

    first_row = cleaned_rows[0]
    second_row = cleaned_rows[1] if len(cleaned_rows) > 1 else []
    width = max(len(row) for row in cleaned_rows[:6])
    family_probe = (first_row + second_row)[:10]

    if looks_like_parameter_table(family_probe):
        headers = infer_parameter_headers(width)
        data_rows = cleaned_rows[1:] if row_contains_schema_terms(first_row) else cleaned_rows
        return headers, data_rows

    if looks_like_header_row(first_row):
        return first_row, cleaned_rows[1:]

    if second_row and looks_like_header_row(second_row) and not looks_like_header_row(first_row):
        return second_row, [first_row] + cleaned_rows[2:]

    return [f"col_{idx + 1}" for idx in range(width)], cleaned_rows


def repair_table_headers(headers: List[str], rows: List[List[str]]) -> List[str]:
    if not headers:
        return headers

    clean_headers = [normalize_table_cell_text(h) for h in headers]
    if looks_like_parameter_table(clean_headers):
        return infer_parameter_headers(len(clean_headers))

    return [header if not is_data_like_header_cell(header) else f"col_{idx + 1}" for idx, header in enumerate(clean_headers)]


def is_model_like_cell(text: str) -> bool:
    s = normalize_table_cell_text(text)
    if not s:
        return False
    if HP_VALUE_RE.search(s) or PRICE_VALUE_RE.search(s) or GEARBOX_VALUE_RE.search(s):
        return False
    if any(ch.isalpha() for ch in s) and any(ch.isdigit() for ch in s):
        return True
    brands = ["kubota", "john deere", "new holland", "eurostar", "massey", "deere"]
    return any(brand in s.lower() for brand in brands)


def repair_parameter_table_rows(headers: List[str], rows: List[List[str]]) -> List[List[str]]:
    if not rows or not looks_like_parameter_table(headers):
        return rows

    expected_cols = len(headers)
    flattened = []
    for row in rows:
        flattened.extend([normalize_table_cell_text(cell) for cell in row if normalize_table_cell_text(cell)])

    rebuilt = []
    current = []
    for cell in flattened:
        if expected_cols >= 5 and "|" in cell:
            current.extend([normalize_table_cell_text(part) for part in cell.split("|") if normalize_table_cell_text(part)])
            continue
        if current and len(current) >= max(3, expected_cols - 1) and is_model_like_cell(cell):
            if len(current) < expected_cols:
                current.extend([""] * (expected_cols - len(current)))
            rebuilt.append(current[:expected_cols])
            current = [cell]
            continue

        current.append(cell)
        if len(current) >= expected_cols:
            rebuilt.append(current)
            current = []

    if current:
        if rebuilt and len(current) < max(2, expected_cols // 2):
            rebuilt[-1][-1] = (rebuilt[-1][-1] + " " + " ".join(current)).strip()
        else:
            current.extend([""] * (expected_cols - len(current)))
            rebuilt.append(current[:expected_cols])

    return rebuilt


def cleanup_model_text(text: str) -> str:
    text = normalize_table_cell_text(text)
    text = re.sub(r"^(?:sia|malaysia)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Kubota)\s+([LM])\s+(\d{3,4})\b", r"\1 \2\3", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" |,.;")
    return text


def normalize_numeric_token(text: str) -> str:
    text = normalize_table_cell_text(text)
    return re.sub(r"(?<=\d)\s+(?=\d)", "", text)


def summarize_spec_record(spec: Dict[str, str]) -> str:
    if spec.get("metric"):
        parts = [f"metric: {spec.get('metric', '').strip()}"]
        value = spec.get("value", "").strip()
        note = spec.get("note", "").strip()
        if value:
            parts.append(f"value: {value}")
        if note:
            parts.append(f"note: {note}")
        return " | ".join(parts)

    ordered_keys = ["model", "engine_hp", "pto_hp", "gearbox", "hydraulic_flow", "lift_capacity", "price", "scenario"]
    parts = []
    for key in ordered_keys:
        value = spec.get(key, "").strip()
        if value:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


def normalize_metric_table_cell_text(text: str) -> str:
    s = normalize_table_cell_text(text)
    if not s:
        return ""
    s = re.sub(r"(?<=[%％\)])(?=[\u4e00-\u9fffA-Za-z])", " | ", s)
    s = re.sub(r"(?<=[\u4e00-\u9fff])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[\u4e00-\u9fff])", " ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    return MULTISPACE_RE.sub(" ", s).strip()


def split_metric_packed_cell(text: str) -> List[str]:
    s = normalize_metric_table_cell_text(text)
    if not s:
        return []
    parts = [part.strip() for part in re.split(r"\s*\|\s*", s) if part.strip()]
    if len(parts) > 2:
        return parts
    metric_tokens = ["农业用地占比", "农业出口总额", "农业出口占比", "GDP比重", "GDP 比重", "市场规模"]
    hits = [token for token in metric_tokens if token in s]
    if len(hits) >= 2 or (len(parts) == 2 and all(len(part) > 4 for part in parts) and any(token in s for token in metric_tokens)):
        pieces = []
        remaining = s
        for token in hits:
            idx = remaining.find(token)
            if idx > 0:
                prefix = remaining[:idx].strip(" ,;|")
                if prefix:
                    pieces.append(prefix)
                remaining = remaining[idx:]
        if remaining:
            pieces.append(remaining.strip(" ,;|"))
        if len(pieces) > 1:
            return pieces
    return [s]


def is_metric_label_fragment(text: str) -> bool:
    s = normalize_metric_table_cell_text(text)
    if not s or is_value_like_cell(s):
        return False
    if len(s) > 24:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    keywords = [
        "产值", "就业", "用地", "出口", "占比", "总额", "规模", "市场", "销量", "收入", "面积", "产量",
        "农业", "GDP", "出口额", "就业人数", "农业用地", "农业出口", "农业产值"
    ]
    if any(keyword in s for keyword in keywords):
        return True
    return len(re.findall(r"[\u4e00-\u9fff]", s)) >= 2


def expand_metric_row_recursive(row: List[str]) -> List[List[str]]:
    width = len(row)
    current = [normalize_metric_table_cell_text(cell) if idx > 0 else normalize_table_cell_text(cell) for idx, cell in enumerate(row)]
    if width < 2:
        return [current]

    for idx in range(1, width):
        cell = current[idx]
        fragments = [frag.strip() for frag in re.split(r"\s*\|\s*", cell) if frag.strip()]
        if len(fragments) != 2:
            continue
        left, right = fragments
        if not is_value_like_cell(left) or not is_metric_label_fragment(right):
            continue

        left_row = current[:]
        left_row[idx] = left
        for tail_idx in range(idx + 1, width):
            left_row[tail_idx] = ""

        right_row = ["" for _ in range(width)]
        right_row[0] = right
        for tail_idx in range(idx + 1, width):
            right_row[tail_idx - idx] = current[tail_idx]
        right_row = right_row[:width]

        expanded_left = expand_metric_row_recursive(left_row)
        expanded_right = expand_metric_row_recursive(right_row)
        return expanded_left + expanded_right

    return [current]


def extract_metric_specs_from_rows(headers: List[str], rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not headers or not rows:
        return []

    metric_specs: List[Dict[str, str]] = []
    first_header = normalize_table_cell_text(headers[0]) if headers else ""
    for row in rows:
        metric = ""
        value_parts: List[str] = []
        for idx, header in enumerate(headers):
            key = header if header else f"col_{idx + 1}"
            raw_value = str(row.get(key, "")).strip()
            if not raw_value:
                continue
            cleaned = normalize_metric_table_cell_text(raw_value) if idx > 0 else normalize_table_cell_text(raw_value)
            if not cleaned:
                continue
            if idx == 0:
                metric = cleaned
                continue
            if cleaned.lower() in {"metric", "指标", "项目", "value", "value_1", "value_2"}:
                continue
            value_parts.append(cleaned)

        metric = normalize_metric_table_cell_text(metric)
        if not metric or metric.lower() in {"metric", "指标", "项目"}:
            continue
        if first_header and metric == first_header:
            continue
        if not value_parts:
            continue

        metric_specs.append({
            "metric": metric,
            "value": " | ".join(value_parts[:4]),
        })

    return metric_specs


def extract_parameter_specs_from_text(text: str, headers: List[str]) -> List[Dict[str, str]]:
    if not looks_like_parameter_table(headers):
        return []

    lines = [normalize_table_cell_text(line) for line in text.splitlines() if line.strip()]
    specs = []
    current: Dict[str, str] | None = None

    def flush_current():
        nonlocal current
        if current and current.get("model"):
            specs.append(current)
        current = None

    for line in lines:
        if "|" not in line:
            continue

        cells = [cleanup_model_text(cell) for cell in line.split("|") if cleanup_model_text(cell)]
        if not cells:
            continue

        model_candidates = [cell for cell in cells if is_model_like_cell(cell)]
        if model_candidates:
            flush_current()
            current = {"model": cleanup_model_text(model_candidates[0])}

        if current is None:
            continue

        for cell in cells:
            normalized = normalize_numeric_token(cell)
            if not current.get("engine_hp") and HP_VALUE_RE.search(normalized):
                current["engine_hp"] = HP_VALUE_RE.search(normalized).group(0)
                continue
            hp_matches = HP_VALUE_RE.findall(normalized)
            if current.get("engine_hp") and not current.get("pto_hp") and hp_matches:
                for hp in hp_matches:
                    if hp != current["engine_hp"]:
                        current["pto_hp"] = hp
                        break
            if not current.get("gearbox") and GEARBOX_VALUE_RE.search(normalized):
                current["gearbox"] = GEARBOX_VALUE_RE.search(normalized).group(0)
            if not current.get("hydraulic_flow") and FLOW_VALUE_RE.search(normalized):
                current["hydraulic_flow"] = FLOW_VALUE_RE.search(normalized).group(0)
            if not current.get("lift_capacity") and LIFT_VALUE_RE.search(normalized):
                current["lift_capacity"] = LIFT_VALUE_RE.search(normalized).group(0)
            if not current.get("price") and PRICE_VALUE_RE.search(normalized):
                current["price"] = PRICE_VALUE_RE.search(normalized).group(0)

        if len(cells) >= 2:
            tail = cells[-1]
            if not any(regex.search(tail) for regex in [HP_VALUE_RE, PRICE_VALUE_RE, GEARBOX_VALUE_RE, FLOW_VALUE_RE, LIFT_VALUE_RE]):
                if len(tail) >= 12 and not current.get("scenario"):
                    current["scenario"] = tail

    flush_current()
    return specs[:12]


def build_table_row_summary(headers: List[str], row: Dict[str, Any], family: str = "") -> str:
    parts = []
    for header in headers:
        value = str(row.get(header, "")).strip()
        if value:
            if family == "metric_table":
                metric_parts = split_metric_packed_cell(value)
                value = " | ".join(metric_parts) if len(metric_parts) > 1 else normalize_metric_table_cell_text(value)
            if header.startswith("col_"):
                parts.append(value)
            else:
                parts.append(f"{header}={value}")
    return " | ".join(parts)


def build_table_embedding_body(table_meta: Dict[str, Any], fallback_text: str) -> str:
    headers = table_meta.get("headers", [])
    rows = table_meta.get("rows", [])
    specs = table_meta.get("specs", [])
    family = table_meta.get("family", "")
    if specs:
        parts = []
        for spec in specs[:6]:
            summary = summarize_spec_record(spec)
            if summary:
                parts.append(summary)
        return "\n".join(parts).strip()

    if not headers:
        return fallback_text.strip()

    parts = [" | ".join(headers)]
    for row in rows[:6]:
        summary = build_table_row_summary(headers, row, family=family)
        if summary:
            parts.append(summary)

    return "\n".join(parts).strip() if parts else fallback_text.strip()


def build_normalized_table_text(table_meta: Dict[str, Any], fallback_text: str) -> str:
    specs = table_meta.get("specs", [])
    if specs:
        lines = [summarize_spec_record(spec) for spec in specs[:8] if summarize_spec_record(spec)]
        return "\n".join(lines).strip()

    headers = table_meta.get("headers", [])
    rows = table_meta.get("rows", [])
    family = table_meta.get("family", "")
    if not headers:
        return fallback_text.strip()

    lines = [" | ".join(headers)]
    for row in rows[:6]:
        summary = build_table_row_summary(headers, row, family=family)
        if summary:
            lines.append(summary)
    return "\n".join(lines).strip()


def looks_like_glued_table_header(header: str) -> bool:
    s = normalize_table_cell_text(header)
    if not s:
        return False
    hits = [token for token in GLUED_HEADER_TOKENS if token in s]
    if len(hits) >= 2:
        return True
    if len(s) >= 18 and "说明" in s and any(token in s for token in ["东南亚", "马来西亚", "泰国", "越南", "菲律宾"]):
        return True
    if len(s) >= 16 and any(ch.isdigit() for ch in s) and any(token in s for token in ["说明", "市场", "报告", "预估"]):
        return True
    return False


def looks_like_bad_table_structure(headers: List[str], rows: List[Dict[str, Any]] | List[List[str]], raw_text: str) -> bool:
    if not headers:
        return False

    normalized_headers = [normalize_table_cell_text(h) for h in headers if normalize_table_cell_text(h)]
    if not normalized_headers:
        return False

    glued_hits = sum(1 for header in normalized_headers if looks_like_glued_table_header(header))
    col_like_hits = sum(1 for header in normalized_headers if str(header).startswith("col_"))
    long_mixed_hits = sum(1 for header in normalized_headers if len(header) >= 16 and any(token in header for token in ["说明", "预估", "市场", "报告"]))
    row_count = len(rows or [])

    if glued_hits >= 1:
        return True
    if len(normalized_headers) >= 6 and row_count <= 2 and (col_like_hits >= 2 or long_mixed_hits >= 1):
        return True
    if len(normalized_headers) >= 5 and row_count <= 2 and raw_text.count("|") >= 4 and any("说明" in h for h in normalized_headers):
        return True
    return False


def build_chunk_text(chunk_type: str, chunk_text: str, table_meta: Dict[str, Any] | None = None) -> str:
    if chunk_type == "table":
        return build_normalized_table_text(table_meta or {"headers": [], "rows": [], "specs": []}, chunk_text)
    return chunk_text.strip()


def repair_table_chunk_text(text: str) -> str:
    lines = [strip_html_tags(html.unescape(line)).strip() for line in text.splitlines()]
    repaired = []

    for line in lines:
        s = MULTISPACE_RE.sub(" ", line).strip()
        if not s:
            continue
        if is_reference_noise_line(s) or is_weak_text_line(s):
            continue
        if looks_like_caption_noise(s) or looks_like_front_matter_noise(s):
            continue
        if TABLE_LINE_RE.match(s) or "|" in s:
            if s.count("|") >= PIPE_TABLE_CELL_SOFT_LIMIT or len(s) >= PIPE_TABLE_MAX_LINE_LENGTH:
                for piece in split_pipe_table_line(s):
                    piece = MULTISPACE_RE.sub(" ", piece).strip()
                    if piece:
                        repaired.append(piece)
            else:
                repaired.append(s)
            continue
        s = re.sub(r"\s*([~～-])\s*(\d)", r"\1\2", s)
        s = re.sub(r"\$\s*\^\s*\{?\s*(-?\d+)\s*\}?\s*\$", r"^{\1}", s)
        repaired.append(s)

    return "\n".join(repaired).strip()


def strip_formula_markup(text: str) -> str:
    text = INLINE_FORMULA_RE.sub(" ", text)
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"\s*([=+\-*/^])\s*", r" \1 ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def dedupe_ordered_texts(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        cleaned = MULTISPACE_RE.sub(" ", str(value or "")).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def sanitize_doc_title(text: str, max_len: int = 96) -> str:
    title = sanitize_header_text(text, max_len=max_len)
    if not title:
        return ""
    if is_reference_section_title(title):
        return ""
    if looks_like_front_matter_noise(title):
        return ""
    return title


def build_retrieval_context(header_path: List[str]) -> str:
    clean_headers = []
    for part in header_path:
        cleaned = sanitize_header_text(part, 48)
        if not cleaned:
            continue
        if (
            contains_ocr_placeholder(cleaned)
            or looks_like_publication_metadata(cleaned)
            or is_low_information_text(cleaned)
            or looks_like_mojibake_text(cleaned)
        ):
            continue
        clean_headers.append(cleaned)
    clean_headers = dedupe_ordered_texts(clean_headers)
    if not clean_headers:
        return ""
    return " > ".join(clean_headers[-3:])


def build_retrieval_body(chunk_type: str, chunk_text: str, table_meta: Dict[str, Any] | None = None) -> str:
    if chunk_type == "table":
        return build_table_embedding_body(table_meta or parse_table_chunk(chunk_text), chunk_text)

    lines = []
    for line in chunk_text.splitlines():
        cleaned = strip_html_tags(line)
        cleaned = strip_ocr_placeholder_noise(cleaned)
        cleaned = STYLE_FRAGMENT_RE.sub(" ", cleaned)
        cleaned = URL_FRAGMENT_RE.sub(" ", cleaned)
        cleaned = TRACKING_PARAM_RE.sub("", cleaned)
        cleaned = strip_formula_markup(cleaned)
        cleaned = MULTISPACE_RE.sub(" ", cleaned).strip(" -|")
        if not cleaned or is_reference_noise_line(cleaned):
            continue
        if contains_ocr_placeholder(cleaned):
            continue
        if looks_like_publication_metadata(cleaned):
            continue
        if looks_like_toc_line(cleaned):
            continue
        if looks_like_mojibake_text(cleaned):
            continue
        if ENDORSEMENT_CREDENTIAL_RE.search(cleaned) and len(cleaned) <= 60:
            continue
        if is_low_information_text(cleaned):
            continue
        lines.append(cleaned)
    return "\n".join(lines).strip()


def build_summary_1line(text: str, max_len: int = 96) -> str:
    clean = MULTISPACE_RE.sub(" ", strip_html_tags(text or "")).strip()
    if not clean:
        return ""
    if looks_like_mojibake_text(clean):
        return ""
    for sentence in split_sentences_fast(clean):
        candidate = strip_empty_visual_prefix(sentence)
        if not candidate or is_decorative_sentence(candidate):
            continue
        if sentence_information_score(candidate) >= 0.28:
            if len(candidate) <= max_len:
                return candidate
            return candidate[:max_len].rstrip() + "..."
    for sep in ["。", "！", "？", "；", ".", "!", "?", ";"]:
        if sep in clean:
            first = clean.split(sep, 1)[0].strip()
            if 8 <= len(first) <= max_len:
                return first
    if len(clean) <= max_len:
        return clean
    return clean[:max_len].rstrip() + "..."


def estimate_chunk_noise_ratio(text: str) -> float:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return 1.0
    noisy = 0
    for line in lines:
        if (
            contains_ocr_placeholder(line)
            or looks_like_publication_metadata(line)
            or looks_like_toc_line(line)
            or looks_like_mojibake_text(line)
            or is_low_information_line(line)
            or is_decorative_sentence(line)
        ):
            noisy += 1
    return noisy / max(len(lines), 1)


def build_embedding_text_clean(
    doc_title: str,
    header_path: List[str],
    chunk_type: str,
    chunk_text: str,
    table_meta: Dict[str, Any] | None = None,
    semantic_role: str = "",
) -> str:
    title = sanitize_doc_title(doc_title)
    context = build_retrieval_context(header_path)
    body = build_retrieval_body(chunk_type, chunk_text, table_meta)
    if not body:
        return ""
    role = str(semantic_role or "").strip()
    family = str((table_meta or {}).get("family", "")).strip()

    context_values = dedupe_ordered_texts([title, context])
    body_lines = [line.strip() for line in body.splitlines() if line.strip()]
    while body_lines and any(body_lines[0].casefold() == value.casefold() for value in context_values):
        body_lines.pop(0)
    body = "\n".join(body_lines).strip()

    prefix_lines = []
    if title:
        prefix_lines.append(f"Document: {title}")
    if context and context.casefold() != title.casefold():
        prefix_lines.append(f"Section: {context}")
    if chunk_type == "table":
        table_label = "table"
        if family and family != "generic_table":
            table_label = f"table ({family})"
        prefix_lines.append(f"Content type: {table_label}")
    elif role and role not in {"body"}:
        prefix_lines.append(f"Content type: {role}")

    parts = prefix_lines[:]
    if body:
        parts.append(body)
    return "\n".join(parts).strip()


def build_embedding_text(
    doc_title: str,
    header_path: List[str],
    chunk_type: str,
    chunk_text: str,
    table_meta: Dict[str, Any] | None = None,
    semantic_role: str = "",
) -> str:
    return build_embedding_text_clean(doc_title, header_path, chunk_type, chunk_text, table_meta, semantic_role)


def get_empty_table_meta() -> Dict[str, Any]:
    return {"headers": [], "rows": [], "specs": [], "family": "generic_table"}


def collect_quality_flags(item: Dict[str, Any], table_meta: Dict[str, Any] | None = None) -> List[str]:
    flags: List[str] = []
    raw_text = item.get("raw_text", "").strip()
    section_title = item.get("section_title", "")
    header_path = item.get("header_path", [])
    chunk_type = item.get("chunk_type", "paragraph")
    section_kind = str(item.get("section_kind", "")).strip()
    info_density = float(item.get("info_density", 0.0) or 0.0)
    noise_score = float(item.get("noise_score", 0.0) or 0.0)
    summary_consistency = float(item.get("summary_consistency_score", 1.0) or 0.0)
    table_meta = table_meta or get_empty_table_meta()
    headers = table_meta.get("headers", [])
    rows = table_meta.get("rows", [])

    if is_reference_section_title(section_title) or any(is_reference_section_title(part) for part in header_path):
        flags.append("reference_section")
    if contains_ocr_placeholder(section_title) or any(contains_ocr_placeholder(part) for part in header_path):
        flags.append("ocr_noise")
    if looks_like_publication_metadata(section_title) or any(looks_like_publication_metadata(part) for part in header_path):
        flags.append("header_pollution")
    if any(looks_like_mojibake_text(part) for part in header_path if part):
        flags.append("header_pollution")
    if section_kind == "publication":
        flags.append("publication_metadata")
    if section_kind == "toc":
        flags.append("toc_section")
    if section_kind == "catalog":
        flags.append("catalog_page")
    if section_kind == "cover":
        flags.append("cover_page")
    if section_kind == "member_list":
        flags.append("member_list")
    if section_kind == "promo_page":
        flags.append("promotional_page")
    if section_kind == "endorsement":
        flags.append("endorsement_like")
    if section_kind == "preface":
        flags.append("preface_like")
    if section_kind == "garbled":
        flags.append("garbled_text")
    if looks_like_front_matter_noise(raw_text):
        if any(is_copyright_noise_line(line) for line in raw_text.splitlines()):
            flags.append("copyright_page")
        if any(is_acknowledgement_noise_line(line) for line in raw_text.splitlines()):
            flags.append("acknowledgement_page")
        if any(is_author_page_noise_line(line) for line in raw_text.splitlines()):
            flags.append("author_page")
        if any(CITATION_PAGE_RE.search(line) for line in raw_text.splitlines()):
            flags.append("citation_page")
    if looks_like_caption_noise(raw_text):
        flags.append("caption_noise")
    if looks_like_reference_chunk(raw_text):
        flags.append("source_list")
    if contains_ocr_placeholder(raw_text):
        flags.append("ocr_noise")
    if looks_like_mojibake_text(raw_text):
        flags.append("garbled_text")
    if looks_like_publication_metadata(raw_text):
        flags.append("publication_metadata")
    if is_low_information_text(raw_text):
        flags.append("low_information")
    if info_density < 0.18 and noise_score >= 0.45:
        flags.append("low_info_density")
    if noise_score >= 0.60:
        flags.append("visual_noise_heavy")
    if 0.0 < summary_consistency < 0.35:
        flags.append("summary_mismatch")
    for page_flag in item.get("page_consistency_flags", []) or []:
        if str(page_flag).strip():
            flags.append(str(page_flag).strip())
    if any(CITATION_LEAD_RE.match(line.strip()) for line in raw_text.splitlines() if line.strip()):
        flags.append("citation_heavy")
    if "### " in raw_text or HEADING_ONLY_RE.search(raw_text):
        flags.append("mixed_heading")
    if HTML_FRAGMENT_RE.search(raw_text):
        flags.append("html_noise")
    if FORMULA_NOISE_RE.search(raw_text):
        flags.append("formula_noise")
    if len(FORMULA_HEAVY_RE.findall(raw_text)) >= 3:
        flags.append("formula_heavy")
    caption_like_lines = 0
    for line in raw_text.splitlines():
        cleaned_line = strip_html_tags(line).strip()
        if CAPTION_INDEX_RE.match(cleaned_line):
            caption_like_lines += 1
    if chunk_type == "paragraph" and caption_like_lines >= 2:
        flags.append("caption_index")
    if chunk_type == "table" and headers and all(str(h).startswith("col_") for h in headers):
        flags.append("generic_headers")
    if chunk_type == "table" and not headers and not rows:
        flags.append("empty_table")
    if chunk_type == "table" and len(raw_text) >= 240 and raw_text.count("|") < 2:
        flags.append("table_misaligned")
    if chunk_type == "table" and TABLE_GLUE_LINE_RE.search(raw_text):
        flags.append("table_glued")
    if chunk_type == "table" and looks_like_bad_table_structure(headers, rows, raw_text):
        flags.append("bad_table")
    if chunk_type == "table" and any(looks_like_glued_table_header(h) for h in headers):
        flags.append("table_header_glued")
    if chunk_type == "table" and looks_like_truncated_boundary(raw_text):
        flags.append("truncated_boundary")
    if chunk_type == "table":
        row_count = len(rows)
        low_confidence_signals = 0
        header_cells = [normalize_table_cell_text(h) for h in headers]
        if any(str(h).startswith("col_") for h in headers):
            low_confidence_signals += 1
        if any(not str(h).strip() for h in headers):
            low_confidence_signals += 1
        if any(is_data_like_header_cell(h) for h in header_cells):
            low_confidence_signals += 1
        if looks_like_truncated_boundary(raw_text):
            low_confidence_signals += 1
        if row_count <= 1 and len(raw_text) >= 140:
            flags.append("table_sparse")
            low_confidence_signals += 1
        if row_count <= 2 and len(headers) >= 4:
            flags.append("table_sparse")
            low_confidence_signals += 1
        if any(isinstance(row, dict) and row.get("_extra") for row in rows):
            flags.append("table_overflow")
            low_confidence_signals += 1
        if headers and sum(1 for h in headers if is_data_like_header_cell(h)) >= max(1, len(headers) // 3):
            flags.append("weak_headers")
            low_confidence_signals += 1
        if row_count and len(headers) >= 4 and row_count <= 3:
            low_confidence_signals += 1
        if len(raw_text) >= 220 and raw_text.count("|") < max(2, len(headers) // 2):
            low_confidence_signals += 1
        if low_confidence_signals >= 2 or (low_confidence_signals >= 1 and (row_count <= 2 or any(str(h).startswith("col_") for h in headers))):
            flags.append("table_low_confidence")
    if chunk_type == "table" and {"table_misaligned", "table_glued", "truncated_tail", "formula_noise"} & set(flags):
        flags.append("bad_table")
    if len(item.get("text", "")) > MAX_CHUNK_CHARS:
        flags.append("oversized_chunk")
    if re.search(r"\b\d+-\s*$", raw_text) or "推荐25-" in raw_text:
        flags.append("truncated_tail")
    if not header_path and chunk_type == "paragraph" and len(raw_text) >= 180:
        flags.append("weak_context")
    if chunk_type == "paragraph":
        sentence_probe = list(split_sentences_fast(raw_text))
        punct_count = len(PUNCT_COUNT_RE.findall(raw_text))
        if len(raw_text) >= 220 and punct_count == 0:
            flags.append("fragmented_text")
        if len(raw_text) >= 260 and len(sentence_probe) <= 1:
            flags.append("fragmented_text")
    sentences = list(split_sentences_fast(raw_text))
    if len(sentences) >= 3 and len(set(sentences)) / max(len(sentences), 1) < 0.7:
        flags.append("repeated_text")
    return sorted(set(flags))


def is_retrieval_eligible(item: Dict[str, Any], quality_flags: List[str] | None = None) -> bool:
    flags = set(quality_flags or item.get("quality_flags", []))
    score = float(item.get("quality_score", 1.0))
    answerability = float(item.get("answerability_score", 1.0))
    retrieval_text = str(item.get("retrieval_text", "")).strip()
    noise_ratio = float(item.get("ocr_noise_ratio", 0.0) or 0.0)
    info_density = float(item.get("info_density", 0.0) or 0.0)
    noise_score = float(item.get("noise_score", 0.0) or 0.0)
    section_kind = str(item.get("section_kind", "") or "")
    is_parent = str(item.get("chunk_level", "child")) == "parent"
    clean_text_mode = is_clean_text_item(item)
    score_threshold = PARENT_QUALITY_SCORE_RETRIEVAL_THRESHOLD if is_parent else QUALITY_SCORE_RETRIEVAL_THRESHOLD
    if clean_text_mode:
        score_threshold = max(0.30 if is_parent else 0.42, score_threshold - 0.08)
    if score < score_threshold:
        return False
    if not retrieval_text:
        return False
    if noise_ratio >= (0.80 if clean_text_mode else 0.60):
        return False
    if noise_score >= 0.85 and info_density < 0.28:
        return False
    if section_kind in {"cover", "catalog", "toc", "member_list", "promo_page"} and info_density < 0.36 and not is_parent:
        return False
    if item.get("chunk_type") == "paragraph" and answerability < (0.14 if clean_text_mode else 0.20) and len(item.get("text", "")) < (80 if clean_text_mode else 100) and not is_parent:
        return False
    if item.get("chunk_type") == "table" and answerability < (0.14 if clean_text_mode else 0.20) and len(item.get("text", "")) < (60 if clean_text_mode else 80):
        return False
    severe_flags = {
        "empty_chunk",
        "empty_table",
        "bad_table",
        "table_glued",
        "table_misaligned",
        "table_header_glued",
        "table_sparse",
        "table_overflow",
        "table_low_confidence",
        "ocr_noise",
        "garbled_text",
    }
    if severe_flags & flags:
        return False
    if {"visual_noise_heavy", "low_info_density", "promotional_page"} & flags and info_density < 0.30:
        return False
    if "weak_context" in flags and len(retrieval_text) < (60 if clean_text_mode else 90) and not is_parent:
        return False
    if {"citation_heavy", "formula_heavy"} & flags and item.get("chunk_type") == "paragraph" and len(item.get("text", "")) > 1200:
        return False
    if item.get("chunk_type") == "paragraph" and len(item.get("text", "")) > 1400 and "oversized_chunk" in flags:
        return False
    if item.get("chunk_type") == "paragraph" and "mixed_heading" in flags and ("|" in item.get("raw_text", "") or len(item.get("text", "")) > 1100):
        return False
    return True


def enrich_chunk_record(item: Dict[str, Any]) -> Dict[str, Any]:
    new_item = dict(item)
    chunk_type = new_item.get("chunk_type", "paragraph")
    chunk_level = str(new_item.get("chunk_level") or ("parent" if chunk_type == "section" else "child"))
    new_item["chunk_level"] = chunk_level
    raw_text = clean_text_fast(new_item.get("raw_text", ""))
    if not raw_text:
        new_item["raw_text"] = ""
        new_item["text"] = ""
        new_item["display_text"] = ""
        new_item["retrieval_text"] = ""
        new_item["embedding_text"] = ""
        new_item["embedding_text_clean"] = ""
        new_item["summary_1line"] = ""
        new_item["summary_consistency_score"] = 0.0
        new_item["text_length"] = 0
        new_item["info_density"] = 0.0
        new_item["noise_score"] = 1.0
        new_item["knowledge_unit_type"] = "empty"
        new_item["ocr_noise_ratio"] = 1.0
        new_item["page_consistency_flags"] = collect_page_consistency_flags(new_item)
        new_item["quality_flags"] = ["empty_chunk"]
        new_item["is_retrieval_eligible"] = False
        return new_item

    new_item["raw_text"] = raw_text
    table_meta = get_empty_table_meta()
    if new_item.get("chunk_type") == "table":
        repaired_table_text = repair_table_chunk_text(raw_text)
        if repaired_table_text:
            raw_text = repaired_table_text
            new_item["raw_text"] = repaired_table_text
        table_meta = parse_table_chunk(raw_text)
        new_item["table_headers"] = table_meta["headers"]
        new_item["table_rows"] = table_meta["rows"]
        new_item["table_row_count"] = len(table_meta["rows"])
        new_item["table_row_texts"] = [
            build_table_row_summary(table_meta["headers"], row, family=table_meta.get("family", ""))
            for row in table_meta["rows"][:8]
            if build_table_row_summary(table_meta["headers"], row, family=table_meta.get("family", ""))
        ]
        new_item["table_specs"] = table_meta.get("specs", [])
        new_item["table_spec_texts"] = [
            summarize_spec_record(spec)
            for spec in table_meta.get("specs", [])[:8]
            if summarize_spec_record(spec)
        ]
        new_item["normalized_table_text"] = build_normalized_table_text(table_meta, raw_text)
    else:
        new_item["table_headers"] = []
        new_item["table_rows"] = []
        new_item["table_row_count"] = 0
        new_item["table_row_texts"] = []
        new_item["table_specs"] = []
        new_item["table_spec_texts"] = []
        new_item["normalized_table_text"] = ""
    new_item["table_family"] = table_meta.get("family", "generic_table") if new_item.get("chunk_type") == "table" else ""
    new_item["section_kind"] = str(new_item.get("section_kind") or "main")
    new_item["knowledge_unit_type"] = str(
        new_item.get("knowledge_unit_type")
        or classify_knowledge_unit_type(raw_text, new_item.get("chunk_type", "paragraph"))
    )
    new_item["info_density"] = compute_info_density(raw_text, new_item.get("chunk_type", "paragraph"))
    new_item["noise_score"] = compute_noise_score(raw_text, new_item.get("section_kind", ""), new_item.get("chunk_type", "paragraph"))
    new_item["page_consistency_flags"] = collect_page_consistency_flags(new_item)
    new_item["semantic_role"] = infer_semantic_role(new_item, table_meta)
    new_item["text"] = build_chunk_text(new_item.get("chunk_type", "paragraph"), raw_text, table_meta)
    new_item["display_text"] = new_item["text"]
    new_item["retrieval_text"] = build_retrieval_body(new_item.get("chunk_type", "paragraph"), raw_text, table_meta)
    new_item["embedding_text_clean"] = build_embedding_text_clean(
        new_item.get("doc_title", ""),
        new_item.get("header_path", []),
        new_item.get("chunk_type", "paragraph"),
        raw_text,
        table_meta,
        new_item.get("semantic_role", ""),
    )
    new_item["embedding_text"] = new_item["embedding_text_clean"]
    new_item["summary_1line"] = build_summary_1line(new_item["retrieval_text"] or new_item["text"])
    new_item["summary_consistency_score"] = validate_summary_consistency(
        new_item["summary_1line"],
        new_item["retrieval_text"] or new_item["text"],
    )
    new_item["text_length"] = len(new_item["text"])
    new_item["ocr_noise_ratio"] = estimate_chunk_noise_ratio(raw_text)
    new_item["quality_flags"] = collect_quality_flags(new_item, table_meta)
    new_item["answerability_score"] = compute_answerability_score(new_item, table_meta, new_item.get("semantic_role", ""))
    if new_item["answerability_score"] < 0.42:
        new_item["quality_flags"] = sorted(set(new_item["quality_flags"] + ["low_answerability"]))
    quality_penalties = {
        "reference_section": 0.95,
        "source_list": 0.9,
        "copyright_page": 0.9,
        "acknowledgement_page": 0.8,
        "author_page": 0.75,
        "citation_page": 0.85,
        "caption_index": 0.7,
        "caption_noise": 0.8,
        "html_noise": 0.45,
        "citation_heavy": 0.35,
        "formula_heavy": 0.3,
        "empty_table": 0.9,
        "table_misaligned": 0.65,
        "table_glued": 0.55,
        "table_header_glued": 0.7,
        "bad_table": 0.8,
        "generic_headers": 0.3,
        "mixed_heading": 0.35,
        "oversized_chunk": 0.35,
        "truncated_tail": 0.4,
        "formula_noise": 0.3,
        "repeated_text": 0.5,
        "table_sparse": 0.35,
        "table_overflow": 0.3,
        "weak_headers": 0.25,
        "fragmented_text": 0.25,
        "table_low_confidence": 0.35,
        "truncated_boundary": 0.65,
        "low_answerability": 0.35,
        "ocr_noise": 0.85,
        "publication_metadata": 0.8,
        "header_pollution": 0.45,
        "low_information": 0.45,
        "toc_section": 0.9,
        "catalog_page": 0.78,
        "cover_page": 0.72,
        "member_list": 0.72,
        "promotional_page": 0.95,
        "visual_noise_heavy": 0.55,
        "low_info_density": 0.35,
        "summary_mismatch": 0.28,
        "page_metadata_mismatch": 0.25,
        "page_missing": 0.12,
        "endorsement_like": 0.75,
        "preface_like": 0.55,
        "garbled_text": 0.9,
        "weak_context": 0.18,
    }
    score = 1.0
    sentences = list(split_sentences_fast(raw_text))
    punct_count = len(PUNCT_COUNT_RE.findall(raw_text))
    is_parent_chunk = chunk_level == "parent"
    clean_text_mode = is_clean_text_item(new_item)
    if chunk_type == "paragraph":
        if len(raw_text) < 90:
            score -= 0.1
        elif len(raw_text) < 150:
            score -= 0.05
        elif len(raw_text) <= 220:
            score -= 0.06
        if len(raw_text) >= 220 and punct_count == 0:
            score -= 0.12
        if len(raw_text) >= 260 and len(sentences) <= 1:
            score -= 0.08
        if 220 < len(raw_text) <= 360:
            score -= 0.05
        elif 360 < len(raw_text) <= 520:
            score -= 0.08
        elif 520 < len(raw_text) <= 800:
            score -= 0.12
        elif len(raw_text) > 800:
            score -= 0.18
        if len(sentences) >= 3 and len(set(sentences)) / max(len(sentences), 1) < 0.8:
            score -= 0.08
        if 2 <= len(sentences) <= 3 and len(raw_text) >= 180:
            score -= 0.03
    elif is_parent_chunk:
        if len(raw_text) >= 240:
            score += 0.06
        if len(raw_text) >= 520:
            score += 0.05
        if len(raw_text) >= 900:
            score += 0.03
        if len(sentences) >= 2:
            score += 0.05
        if len(sentences) >= 4:
            score += 0.03
        if punct_count >= 2:
            score += 0.03
    elif chunk_type == "table":
        row_count = len(table_meta.get("rows", []))
        header_count = len(table_meta.get("headers", []))
        if row_count <= 1 and len(raw_text) >= 140:
            score -= 0.12
        if row_count <= 2 and header_count >= 4:
            score -= 0.08
        if 3 <= row_count <= 5:
            score -= 0.05
        elif 6 <= row_count <= 8:
            score -= 0.08
        elif row_count >= 9:
            score -= 0.12
        if table_meta.get("family", "generic_table") == "generic_table":
            score -= 0.05
        if any(isinstance(row, dict) and row.get("_extra") for row in table_meta.get("rows", [])):
            score -= 0.08
    for flag in new_item["quality_flags"]:
        score -= quality_penalties.get(flag, 0.08)
    if chunk_type == "paragraph" and looks_like_front_matter_noise(raw_text):
        score -= 0.15
    if chunk_type == "paragraph" and looks_like_caption_noise(raw_text):
        score -= 0.2
    if chunk_type == "paragraph" and "mixed_heading" in new_item["quality_flags"]:
        score -= 0.15
    if chunk_type == "paragraph" and looks_like_front_matter_noise(raw_text):
        score -= 0.18
    if chunk_type == "paragraph" and len(raw_text) > 650 and len(sentences) <= 2:
        score -= 0.12
    if chunk_type == "paragraph" and len(raw_text) > 900:
        score -= 0.12
    if new_item["ocr_noise_ratio"] >= 0.2:
        score -= 0.12
    if new_item["ocr_noise_ratio"] >= 0.34:
        score -= 0.18
    if chunk_type == "table" and table_meta.get("family") in {"generic_table"} and len(raw_text) >= 300:
        score -= 0.08
    if "html_noise" in new_item["quality_flags"] and chunk_type == "paragraph":
        score -= 0.15
    if chunk_type == "paragraph" and ("citation_heavy" in new_item["quality_flags"] or "formula_heavy" in new_item["quality_flags"]):
        score -= 0.1
    if chunk_type == "paragraph" and ("mixed_heading" in new_item["quality_flags"] or "fragmented_text" in new_item["quality_flags"]):
        score -= 0.1
    if chunk_type == "table" and ("table_glued" in new_item["quality_flags"] or "table_misaligned" in new_item["quality_flags"]):
        score -= 0.12
    if chunk_type == "table" and ("bad_table" in new_item["quality_flags"] or "table_header_glued" in new_item["quality_flags"]):
        score -= 0.2
    if chunk_type == "table" and "table_sparse" in new_item["quality_flags"]:
        score -= 0.08
    if chunk_type == "table" and "table_overflow" in new_item["quality_flags"]:
        score -= 0.08
    if chunk_type == "paragraph" and "fragmented_text" in new_item["quality_flags"]:
        score -= 0.12
    score += float(new_item.get("info_density", 0.0) or 0.0) * (0.18 if not is_parent_chunk else 0.12)
    score -= float(new_item.get("noise_score", 0.0) or 0.0) * (0.22 if not is_parent_chunk else 0.14)
    if chunk_type == "table" and "table_low_confidence" in new_item["quality_flags"]:
        score -= 0.12
    if "truncated_boundary" in new_item["quality_flags"]:
        score -= 0.18
    if chunk_type == "table" and len(raw_text) > 550:
        score -= 0.08
    if new_item["answerability_score"] < 0.25:
        score -= 0.08 if is_parent_chunk else 0.18
    elif new_item["answerability_score"] < 0.42:
        score -= 0.05 if is_parent_chunk else 0.12
    elif new_item["answerability_score"] < 0.55:
        score -= 0.03 if is_parent_chunk else 0.06
    if chunk_type == "table" and new_item["answerability_score"] < 0.5:
        score -= 0.1
    if clean_text_mode:
        score += 0.05
        if chunk_type == "paragraph" and len(raw_text) >= 120:
            score += 0.03
        if punct_count >= 1 and new_item["ocr_noise_ratio"] <= 0.12:
            score += 0.02
    new_item["quality_score"] = max(0.0, min(1.0, score))
    new_item["is_retrieval_eligible"] = is_retrieval_eligible(new_item, new_item["quality_flags"])
    return new_item


def get_parent_chunk_body(child_chunk: Dict[str, Any]) -> str:
    body = str(
        child_chunk.get("retrieval_text")
        or child_chunk.get("display_text")
        or child_chunk.get("text")
        or child_chunk.get("raw_text")
        or ""
    ).strip()
    overlap_text = str(child_chunk.get("overlap_text") or "").strip()
    if overlap_text and body:
        stripped = remove_overlap_pollution(overlap_text, body)
        if stripped.strip():
            body = stripped.strip()
    return clean_text_fast(body)


def take_text_head(text: str, max_chars: int = 120) -> str:
    cleaned = clean_text_fast(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].strip()


def take_text_tail(text: str, max_chars: int = 120) -> str:
    cleaned = clean_text_fast(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:].strip()


def build_parent_chunks(child_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not child_chunks:
        return []

    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for item in child_chunks:
        key = (item.get("doc_id"), item.get("section_id"))
        grouped.setdefault(key, []).append(item)

    parent_chunks: List[Dict[str, Any]] = []
    group_keys = sorted(grouped.keys(), key=lambda key: (str(key[0]), str(key[1])))

    for key in group_keys:
        items = grouped[key]
        items.sort(key=lambda x: (int(x.get("chunk_in_section", 0)), str(x.get("chunk_uid", ""))))
        section_parents: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        pending_children: List[Dict[str, Any]] = []
        pending_chars = 0

        def flush_pending():
            nonlocal pending_children, pending_chars
            if not pending_children:
                return

            first_child = pending_children[0]
            child_bodies = [get_parent_chunk_body(child) for child in pending_children]
            child_bodies = [body for body in child_bodies if body]
            parent_raw_text = "\n\n".join(child_bodies).strip()
            child_uids = [str(child.get("chunk_uid") or build_chunk_uid(str(child.get("source", "")), child)) for child in pending_children]
            role_counts: Dict[str, int] = {}
            for child in pending_children:
                role = str(child.get("semantic_role") or "body").strip() or "body"
                role_counts[role] = role_counts.get(role, 0) + 1
            dominant_role = "body"
            if role_counts:
                dominant_role = sorted(
                    role_counts.items(),
                    key=lambda kv: (kv[1], kv[0] != "body", len(kv[0])),
                )[-1][0]
            parent_index = len(section_parents)
            parent_page_meta = summarize_child_page_metadata(pending_children)
            parent_item = {
                "doc_id": first_child.get("doc_id"),
                "doc_title": first_child.get("doc_title", ""),
                "source": first_child.get("source", ""),
                "doc_total_chars": first_child.get("doc_total_chars", 0),
                "doc_chunk_target_chars": first_child.get("doc_chunk_target_chars", 0),
                "doc_chunk_max_chars": first_child.get("doc_chunk_max_chars", 0),
                "doc_chunk_min_raw_chars": first_child.get("doc_chunk_min_raw_chars", 0),
                "section_id": first_child.get("section_id"),
                "chunk_in_section": parent_index,
                "header_path": list(first_child.get("header_path", [])),
                "section_title": first_child.get("section_title", ""),
                "breadcrumb": first_child.get("breadcrumb", ""),
                "section_kind": first_child.get("section_kind", "main"),
                "source_text_profile": first_child.get("source_text_profile", ""),
                "chunk_type": "section",
                "chunk_level": "parent",
                "semantic_role": dominant_role,
                "raw_text": parent_raw_text,
                "child_chunk_uids": child_uids,
                "child_chunk_count": len(child_uids),
                "child_chunk_start_index": int(pending_children[0].get("chunk_in_section", 0)),
                "child_chunk_end_index": int(pending_children[-1].get("chunk_in_section", 0)),
            } | parent_page_meta
            parent_item = enrich_chunk_record(parent_item)
            parent_item["chunk_uid"] = build_parent_chunk_uid(str(parent_item.get("source", "")), parent_item)
            section_parents.append((parent_item, list(pending_children)))
            pending_children = []
            pending_chars = 0

        for child in items:
            child_body = get_parent_chunk_body(child)
            child_chars = len(child_body)
            if pending_children and (
                pending_chars + child_chars > PARENT_CHUNK_MAX_CHARS
                or (pending_chars >= PARENT_CHUNK_TARGET_CHARS and len(pending_children) >= 2)
            ):
                flush_pending()
            pending_children.append(child)
            pending_chars += child_chars

        flush_pending()

        parent_total = len(section_parents)
        for parent_index, (parent_item, _) in enumerate(section_parents):
            boundary_parts = []
            if parent_index > 0:
                prev_children = section_parents[parent_index - 1][1]
                prev_tail = take_text_tail(get_parent_chunk_body(prev_children[-1]), 120) if prev_children else ""
                if prev_tail:
                    boundary_parts.append(prev_tail)
            boundary_parts.append(parent_item.get("raw_text", ""))
            if parent_index + 1 < parent_total:
                next_children = section_parents[parent_index + 1][1]
                next_head = take_text_head(get_parent_chunk_body(next_children[0]), 120) if next_children else ""
                if next_head:
                    boundary_parts.append(next_head)
            enriched_parent = dict(parent_item)
            enriched_parent["raw_text"] = "\n\n".join(part for part in boundary_parts if part).strip()
            enriched_parent = enrich_chunk_record(enriched_parent)
            enriched_parent["chunk_uid"] = build_parent_chunk_uid(str(enriched_parent.get("source", "")), enriched_parent)
            section_parents[parent_index] = (enriched_parent, section_parents[parent_index][1])

        for parent_index, (parent_item, children) in enumerate(section_parents):
            parent_item["chunk_level"] = "parent"
            parent_item["chunk_in_section"] = parent_index
            parent_item["parent_chunk_index"] = parent_index
            parent_item["parent_chunk_count"] = parent_total
            parent_item["child_chunk_count"] = len(parent_item.get("child_chunk_uids", []))
            for child in children:
                child["chunk_level"] = "child"
                child["parent_chunk_uid"] = parent_item["chunk_uid"]
                child["parent_chunk_index"] = parent_index
                child["parent_chunk_count"] = parent_total
            parent_chunks.append(parent_item)

    return reindex_chunks(parent_chunks)


def max_raw_chars_for_item(item: Dict[str, Any]) -> int:
    doc_max_chars = int(item.get("doc_chunk_max_chars", MAX_CHUNK_CHARS))
    doc_min_raw_chars = int(item.get("doc_chunk_min_raw_chars", 240))
    context_only = build_embedding_text(
        item.get("doc_title", ""),
        item.get("header_path", []),
        item.get("chunk_type", "paragraph"),
        "",
        {
            "headers": item.get("table_headers", []),
            "rows": item.get("table_rows", []),
            "specs": item.get("table_specs", []),
            "family": item.get("table_family", ""),
        },
        item.get("semantic_role", ""),
    )
    reserved = len(context_only) + (2 if context_only else 0)
    base_limit = max(doc_min_raw_chars, doc_max_chars - reserved)
    if item.get("chunk_type") == "paragraph" and item.get("doc_total_chars", 0) >= 50000:
        base_limit = min(base_limit, max(180, doc_max_chars - 160))
    elif item.get("chunk_type") == "paragraph" and item.get("doc_total_chars", 0) >= 20000:
        base_limit = min(base_limit, max(200, doc_max_chars - 120))
    return max(doc_min_raw_chars, base_limit)


def get_min_expected_chunk_count(doc_total_chars: int) -> int:
    if doc_total_chars >= 120000:
        return 24
    if doc_total_chars >= 80000:
        return 18
    if doc_total_chars >= 50000:
        return 14
    if doc_total_chars >= 20000:
        return 10
    return 5


def split_chunk_record(item: Dict[str, Any], aggressive: bool = False) -> List[Dict[str, Any]]:
    chunk_text = item["raw_text"].strip()
    if not chunk_text:
        return []

    max_chars = max_raw_chars_for_item(item)
    if aggressive:
        max_chars = max(120, int(max_chars * 0.65))
    block_type = item.get("chunk_type", "paragraph")
    if block_type == "paragraph":
        pieces = split_dense_text(chunk_text, max_chars)
    else:
        target_chars = TARGET_CHUNK_CHARS if not aggressive else max(140, int(TARGET_CHUNK_CHARS * 0.7))
        pieces = []
        for piece in split_block_by_type(chunk_text, block_type, target_chars=target_chars, max_chars=max_chars):
            if len(piece) <= max_chars:
                pieces.append(piece)
            else:
                pieces.extend(split_dense_text(piece, max_chars))
    if len(pieces) <= 1:
        return [item]

    split_items = []
    for idx, piece in enumerate(pieces):
        new_item = dict(item)
        new_item["chunk_in_section"] = item.get("chunk_in_section", 0) + idx
        new_item["raw_text"] = piece.strip()
        new_item["knowledge_unit_type"] = classify_knowledge_unit_type(piece, block_type)
        split_items.append(enrich_chunk_record(new_item))
    return split_items


def second_pass_chunk_cleanup(chunks: List[Dict[str, Any]], aggressive: bool = False) -> List[Dict[str, Any]]:
    cleaned = []
    for item in chunks:
        new_item = enrich_chunk_record(item)
        raw_text = new_item["raw_text"]
        if not raw_text:
            continue

        split_limit = max_raw_chars_for_item(new_item)
        if aggressive:
            split_limit = max(120, int(split_limit * 0.7))
        if len(raw_text) > split_limit or len(new_item["embedding_text"]) > MAX_CHUNK_CHARS:
            cleaned.extend(split_chunk_record(new_item, aggressive=aggressive))
        else:
            cleaned.append(new_item)

    return reindex_chunks(cleaned)


def group_blocks_to_chunks(blocks: List[Dict[str, str]], target_chars: int = TARGET_CHUNK_CHARS, max_chars: int = MAX_CHUNK_CHARS) -> List[Dict[str, str]]:
    grouped = []
    current_parts: List[str] = []
    current_type = None
    current_unit_type = ""

    def flush():
        nonlocal current_parts, current_type, current_unit_type
        if current_parts:
            item = {"block_type": current_type or "paragraph", "text": "\n\n".join(current_parts).strip()}
            if current_unit_type:
                item["knowledge_unit_type"] = current_unit_type
            grouped.append(item)
        current_parts = []
        current_type = None
        current_unit_type = ""

    for block in blocks:
        block_text = block["text"].strip()
        block_type = block["block_type"]
        block_unit_type = str(block.get("knowledge_unit_type") or "")
        if not block_text:
            continue
        if block_type == "reference":
            continue

        if block_type == "heading":
            flush()
            current_parts = [block_text]
            current_type = "heading"
            current_unit_type = block_unit_type
            continue

        if len(block_text) > max_chars:
            flush()
            for piece in split_block_by_type(block_text, block_type, target_chars=target_chars, max_chars=max_chars):
                item = {"block_type": block_type, "text": piece}
                if block_unit_type:
                    item["knowledge_unit_type"] = block_unit_type
                grouped.append(item)
            continue

        if current_type == "heading":
            candidate = "\n\n".join(current_parts + [block_text]).strip()
            if len(candidate) <= target_chars + 80:
                item = {"block_type": block_type, "text": candidate}
                if block_unit_type:
                    item["knowledge_unit_type"] = block_unit_type
                grouped.append(item)
                current_parts = []
                current_type = None
                current_unit_type = ""
                continue
            flush()

        if current_type is None:
            current_parts = [block_text]
            current_type = block_type
            current_unit_type = block_unit_type
            continue

        same_type = block_type == current_type
        same_unit = not current_unit_type or not block_unit_type or current_unit_type == block_unit_type
        high_value_boundary = bool(block_unit_type and block_unit_type not in {"body", "low_value"} and current_unit_type and current_unit_type != block_unit_type)
        candidate = "\n\n".join(current_parts + [block_text]).strip()
        can_merge = same_type and same_unit and not high_value_boundary and len(candidate) <= target_chars and current_type in {"paragraph", "list"}

        if can_merge:
            current_parts.append(block_text)
            if not current_unit_type:
                current_unit_type = block_unit_type
        else:
            flush()
            current_parts = [block_text]
            current_type = block_type
            current_unit_type = block_unit_type

    flush()
    return grouped


def section_to_semantic_blocks(
    content: str,
    target_chars: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    source_profile: str = "clean_text",
) -> List[Dict[str, str]]:
    paragraphs = split_paragraphs(content)
    blocks = []
    skip_meta_noise = source_profile != "clean_text"
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        if not para:
            i += 1
            continue

        if looks_like_reference_chunk(para):
            i += 1
            continue
        if skip_meta_noise and looks_like_front_matter_noise(para):
            i += 1
            continue
        if skip_meta_noise and looks_like_caption_noise(para):
            i += 1
            continue
        if contains_ocr_placeholder(para):
            i += 1
            continue
        if skip_meta_noise and looks_like_publication_metadata(para):
            i += 1
            continue
        if source_profile == "clean_text":
            para = clean_knowledge_text(para, source_profile=source_profile)
            if not para:
                i += 1
                continue

        block_type = classify_block(para)
        if block_type == "reference":
            i += 1
            continue

        if block_type == "paragraph":
            mixed_blocks = split_mixed_paragraph_block(para)
            if len(mixed_blocks) > 1:
                cleaned_mixed_blocks = []
                for mixed_block in mixed_blocks:
                    mixed_text = clean_knowledge_text(mixed_block["text"], source_profile=source_profile) if source_profile == "clean_text" else mixed_block["text"]
                    if mixed_block["block_type"] != "reference" and mixed_text.strip():
                        cleaned_mixed_blocks.append({
                            "block_type": mixed_block["block_type"],
                            "text": mixed_text.strip(),
                            "knowledge_unit_type": classify_knowledge_unit_type(mixed_text, mixed_block["block_type"]),
                        })
                blocks.extend(cleaned_mixed_blocks)
                i += 1
                continue
            if source_profile == "clean_text":
                knowledge_blocks = split_knowledge_units(para, target_chars=target_chars, max_chars=max_chars)
                if knowledge_blocks:
                    blocks.extend(knowledge_blocks)
                    i += 1
                    continue

        if block_type == "faq" and i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            if next_para and classify_block(next_para) == "paragraph":
                para = para + "\n\n" + next_para
                i += 1

        blocks.append({
            "block_type": block_type,
            "text": para,
            "knowledge_unit_type": classify_knowledge_unit_type(para, block_type),
        })
        i += 1

    return group_blocks_to_chunks(blocks, target_chars=target_chars, max_chars=max_chars)


def build_initial_chunks_for_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_profile = str(doc.get("source_text_profile") or "clean_text")
    cleaned_text = prepare_text_for_chunking(doc["text"])
    doc_profile = get_document_chunk_profile(len(cleaned_text), source_profile=source_profile)
    sections = parse_markdown_sections(cleaned_text, allow_implicit_headers=source_profile != "clean_text")
    results = []

    for sec_idx, section in enumerate(sections):
        header_path = []
        for header in section["header_path"]:
            cleaned_header = sanitize_header_text(header)
            if cleaned_header:
                header_path.append(cleaned_header)
        content = prepare_text_for_chunking(section["content"])
        section_kind = classify_section_kind(header_path, content)
        page_meta = extract_page_metadata(header_path, header_path[-1] if header_path else "")
        semantic_blocks = section_to_semantic_blocks(
            content,
            target_chars=doc_profile["target_chars"],
            max_chars=doc_profile["max_chars"],
            source_profile=source_profile,
        )

        for chunk_idx, block in enumerate(semantic_blocks):
            chunk_text = block["text"].strip()
            if not chunk_text:
                continue

            doc_title = doc.get("title", f"doc_{doc['doc_id']}")
            breadcrumb = " > ".join([doc_title] + header_path).strip(" >")
            chunk_type = block["block_type"]
            results.append(enrich_chunk_record({
                "doc_id": doc["doc_id"],
                "doc_title": doc_title,
                "source": doc.get("source", ""),
                "doc_total_chars": len(cleaned_text),
                "doc_chunk_target_chars": doc_profile["target_chars"],
                "doc_chunk_max_chars": doc_profile["max_chars"],
                "doc_chunk_min_raw_chars": doc_profile["min_raw_chars"],
                "section_id": sec_idx,
                "chunk_in_section": chunk_idx,
                "header_path": header_path,
                "section_title": header_path[-1] if header_path else "",
                "breadcrumb": breadcrumb,
                "section_kind": section_kind,
                "source_text_profile": source_profile,
                "chunk_type": chunk_type,
                "knowledge_unit_type": str(block.get("knowledge_unit_type") or ""),
                "raw_text": chunk_text,
            } | page_meta))

    return results


# =========================
# 8. Semantic merge / overlap
# =========================
def char_ngrams(text: str, n: int = MINHASH_SHINGLE_N) -> set:
    text = MULTISPACE_RE.sub("", normalize_for_hash(text))
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def semantic_merge_adjacent_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    # Warm the embedding cache in batches before pairwise comparisons.
    prefetch_texts: List[str] = []
    seen_texts: set[str] = set()
    for item in chunks:
        text = str(item.get("embedding_text") or "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            prefetch_texts.append(text)
    if prefetch_texts:
        get_embeddings(prefetch_texts)

    merged = []
    i = 0
    while i < len(chunks):
        current = dict(chunks[i])
        current_shingles = char_ngrams(current["raw_text"])
        j = i + 1

        while j < len(chunks):
            same_doc = chunks[j]["doc_id"] == current["doc_id"]
            same_section = chunks[j]["section_id"] == current["section_id"]
            same_type = chunks[j].get("chunk_type") == current.get("chunk_type")
            current_role = str(current.get("semantic_role") or "")
            next_role = str(chunks[j].get("semantic_role") or "")

            if not same_doc:
                break
            if MERGE_ONLY_SAME_SECTION and not same_section:
                break
            if not same_type:
                break
            if current.get("chunk_type") == "table" and current_role != next_role:
                break
            if current.get("chunk_type") == "paragraph" and {current_role, next_role} & {"front_matter", "warning", "steps"} and current_role != next_role:
                break

            candidate_raw = current["raw_text"] + "\n\n" + chunks[j]["raw_text"]
            candidate_text = f"{current['breadcrumb']}\n\n{candidate_raw}" if current["breadcrumb"] else candidate_raw
            if len(candidate_text) > MAX_CHUNK_CHARS + 120:
                break

            next_shingles = char_ngrams(chunks[j]["raw_text"])
            lexical_sim = jaccard_similarity(current_shingles, next_shingles)
            if lexical_sim < LEXICAL_MERGE_THRESHOLD:
                break

            current_emb, next_emb = get_embeddings([current["embedding_text"], chunks[j]["embedding_text"]])
            sim = cosine_similarity(current_emb, next_emb)

            if sim >= SEMANTIC_MERGE_THRESHOLD:
                current["raw_text"] = candidate_raw
                current = enrich_chunk_record(current)
                current_shingles = char_ngrams(current["raw_text"])
                j += 1
            else:
                break

        merged.append(current)
        i = j

    counter = {}
    for item in merged:
        key = (item["doc_id"], item["section_id"])
        counter.setdefault(key, 0)
        item["chunk_in_section"] = counter[key]
        counter[key] += 1

    return merged


def get_last_n_sentences(text: str, n: int) -> str:
    if n <= 0:
        return ""
    sents = list(split_sentences_fast(text))
    if not sents:
        return ""
    return " ".join(sents[-n:]).strip()


def remove_overlap_pollution(prefix: str, current_raw_text: str) -> str:
    if not prefix:
        return current_raw_text

    if current_raw_text.startswith(prefix):
        return current_raw_text[len(prefix):].strip()

    current_sents = list(split_sentences_fast(current_raw_text))
    prefix_sents = list(split_sentences_fast(prefix))
    if current_sents and prefix_sents:
        k = min(len(prefix_sents), len(current_sents), 3)
        while k > 0:
            if current_sents[:k] == prefix_sents[-k:]:
                current_sents = current_sents[k:]
                break
            k -= 1
        return " ".join(current_sents).strip()

    return current_raw_text


def get_dynamic_overlap_sentences(item: Dict[str, Any]) -> int:
    chunk_type = item.get("chunk_type", "paragraph")
    role = str(item.get("semantic_role") or "")
    answerability = float(item.get("answerability_score", 0.0))
    length = len(item.get("raw_text", ""))

    if chunk_type == "table":
        if role in {"parameter_table", "metric_table", "scenario_matrix", "policy_matrix", "wide_stats_table", "timeline_table", "distribution_table"}:
            return 0
        return 0 if answerability >= 0.6 else 1

    if role in {"front_matter", "warning", "steps", "definition", "recommendation", "conclusion", "example"}:
        return 1
    if role == "list":
        return 0
    if answerability < 0.35 or length > 850:
        return 2
    if answerability < 0.55 or length > 520:
        return 1
    return 1


def add_sentence_overlap(chunks: List[Dict[str, Any]], overlap_sentences: int) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks

    results = []
    prev_doc_id = None
    prev_section_id = None
    prev_raw_text = ""

    for item in chunks:
        new_item = dict(item)
        overlap_text = ""
        dynamic_overlap = get_dynamic_overlap_sentences(item)

        if prev_doc_id == item["doc_id"] and prev_section_id == item["section_id"] and prev_raw_text and dynamic_overlap > 0:
            overlap_text = get_last_n_sentences(prev_raw_text, dynamic_overlap)
            cleaned_current_raw = remove_overlap_pollution(overlap_text, new_item["raw_text"])
            merged_raw = (overlap_text + "\n\n" + cleaned_current_raw).strip() if overlap_text else cleaned_current_raw
            new_item["raw_text"] = merged_raw
            new_item = enrich_chunk_record(new_item)
            new_item["overlap_sentences"] = dynamic_overlap
            new_item["overlap_text"] = overlap_text
        else:
            new_item["overlap_sentences"] = 0
            new_item["overlap_text"] = ""

        results.append(new_item)
        prev_doc_id = item["doc_id"]
        prev_section_id = item["section_id"]
        prev_raw_text = item["raw_text"]

    return results


# =========================
# 9. Filter / dedup
# =========================
def is_low_information_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if looks_like_toc_line(stripped):
        return True
    if is_page_noise(stripped):
        return True
    if contains_ocr_placeholder(stripped):
        return True
    if looks_like_publication_metadata(stripped):
        return True
    if looks_like_mojibake_text(stripped):
        return True
    if is_low_information_text(stripped):
        return True
    if len(stripped) <= 8 and not re.search(r"[\u4e00-\u9fffA-Za-z]", stripped):
        return True
    return False


def is_bad_chunk(item: Dict[str, Any]) -> bool:
    raw_text = item["raw_text"].strip()
    length = len(raw_text)
    chunk_type = item.get("chunk_type", "paragraph")
    quality_flags = set(item.get("quality_flags", []))
    quality_score = float(item.get("quality_score", 1.0))
    retrieval_text = str(item.get("retrieval_text", "")).strip()
    noise_ratio = float(item.get("ocr_noise_ratio", 0.0) or 0.0)
    info_density = float(item.get("info_density", 0.0) or 0.0)
    noise_score = float(item.get("noise_score", 0.0) or 0.0)
    section_kind = str(item.get("section_kind", "") or "")
    embedding_text_clean = (item.get("embedding_text_clean") or item.get("embedding_text") or "").strip()
    clean_text_mode = is_clean_text_item(item)

    quality_floor = 0.28 if clean_text_mode else max(0.35, QUALITY_SCORE_RETRIEVAL_THRESHOLD - 0.15)
    if quality_score < quality_floor:
        return True
    if not retrieval_text:
        return True
    if noise_ratio >= (0.82 if clean_text_mode else 0.70):
        return True
    if noise_score >= 0.88 and info_density < 0.28:
        return True
    if section_kind in {"cover", "catalog", "toc", "member_list", "promo_page"} and info_density < 0.32 and chunk_type == "paragraph":
        return True
    if not embedding_text_clean:
        return True

    if {"empty_chunk", "empty_table", "bad_table", "table_glued", "table_misaligned", "table_header_glued", "table_sparse", "table_overflow", "table_low_confidence", "ocr_noise", "garbled_text"} & quality_flags:
        return True
    if {"visual_noise_heavy", "low_info_density", "promotional_page"} & quality_flags and info_density < 0.30:
        return True

    if looks_like_caption_noise(raw_text) and length < (70 if clean_text_mode else 100):
        return True
    if contains_ocr_placeholder(raw_text) and length < (120 if not clean_text_mode else 80):
        return True
    if looks_like_publication_metadata(raw_text) and length < (60 if clean_text_mode else 100):
        return True
    if is_low_information_text(raw_text) and length < (50 if clean_text_mode else 80):
        return True

    if chunk_type == "paragraph" and length < (30 if clean_text_mode else 40):
        return True
    if chunk_type in {"table", "list", "faq"} and length < 20:
        return True

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if lines and sum(1 for line in lines if is_low_information_line(line)) / len(lines) > (0.75 if clean_text_mode else 0.6):
        return True

    punct_count = len(PUNCT_COUNT_RE.findall(raw_text))
    if chunk_type == "paragraph" and length > (260 if clean_text_mode else 180) and punct_count == 0:
        return True

    sents = list(split_sentences_fast(raw_text))
    if len(sents) >= 3:
        unique_ratio = len(set(sents)) / max(len(sents), 1)
        if unique_ratio < 0.6:
            return True

    if chunk_type == "paragraph" and length < MIN_CHUNK_CHARS and not SENTENCE_END_RE.search(raw_text):
        # Keep short-but-coherent explanatory paragraphs; only drop obvious fragments.
        if clean_text_mode:
            if len(sents) == 0 and punct_count == 0:
                return True
        elif len(sents) <= 1 and punct_count <= 1:
            return True

    if "mixed_heading" in quality_flags and (length > (900 if clean_text_mode else 700) or raw_text.count("|") >= (4 if clean_text_mode else 2)):
        return True

    if "formula_noise" in quality_flags and chunk_type == "paragraph" and length > 300:
        return True

    if raw_text.count("|") >= 6 and chunk_type != "table":
        return True

    if "oversized_chunk" in quality_flags and chunk_type == "paragraph" and length > 1100:
        return True
    if "weak_context" in quality_flags and len(retrieval_text) < (180 if clean_text_mode else 260):
        return True

    return False


def quality_filter(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [x for x in chunks if x.get("is_retrieval_eligible", True) and not is_bad_chunk(x)]
    counter = {}
    for item in filtered:
        key = (item["doc_id"], item["section_id"])
        counter.setdefault(key, 0)
        item["chunk_in_section"] = counter[key]
        counter[key] += 1
    return filtered


def normalize_for_hash(text: str) -> str:
    text = clean_text_fast(text)
    text = MULTISPACE_RE.sub(" ", text).strip().lower()
    return text


def exact_deduplicate(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    deduped = []
    for item in chunks:
        norm = normalize_for_hash(item["raw_text"])
        fp = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if fp in seen:
            continue
        seen[fp] = True
        item["exact_hash"] = fp
        deduped.append(item)
    return deduped


def choose_canonical_chunk(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    score_a = (-abs(len(a["raw_text"]) - TARGET_CHUNK_CHARS), len(a.get("header_path", [])), len(a["raw_text"]))
    score_b = (-abs(len(b["raw_text"]) - TARGET_CHUNK_CHARS), len(b.get("header_path", [])), len(b["raw_text"]))
    return a if score_a >= score_b else b


def near_deduplicate_fast(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    # Batch-populate embeddings once so later comparisons mostly hit cache.
    prefetch_texts: List[str] = []
    seen_texts: set[str] = set()
    for item in chunks:
        text = str(item.get("embedding_text") or "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            prefetch_texts.append(text)
    if prefetch_texts:
        get_embeddings(prefetch_texts)

    kept = []
    kept_shingles = []
    kept_compact_texts: List[str] = []
    bucket_to_indexes: Dict[Tuple[int, str], List[int]] = {}
    embedding_by_text: Dict[str, List[float]] = {}

    def get_embedding_for_text(text: str) -> List[float]:
        if text not in embedding_by_text:
            embedding_by_text[text] = get_embeddings([text])[0]
        return embedding_by_text[text]

    def compact_text(text: str) -> str:
        return MULTISPACE_RE.sub("", normalize_for_hash(text))

    def build_bucket_tokens(compact: str) -> List[str]:
        if not compact:
            return [""]
        tokens = {
            compact[:18],
            compact[-18:],
        }
        if len(compact) > 36:
            mid = len(compact) // 2
            tokens.add(compact[max(0, mid - 9):min(len(compact), mid + 9)])
        if len(compact) > 72:
            tokens.add(compact[:9] + compact[-9:])
        return [token for token in tokens if token]

    def build_bucket_keys(compact: str) -> List[Tuple[int, str]]:
        length_bucket = max(0, len(compact) // 80)
        return [(length_bucket, token) for token in build_bucket_tokens(compact)]

    def collect_candidate_indexes(compact: str) -> List[int]:
        length_bucket = max(0, len(compact) // 80)
        candidate_indexes: set[int] = set()
        for delta in (-1, 0, 1):
            probe_bucket = max(0, length_bucket + delta)
            for token in build_bucket_tokens(compact):
                candidate_indexes.update(bucket_to_indexes.get((probe_bucket, token), []))
        if not candidate_indexes:
            # Duplicates are usually nearby in the same OCR-heavy section; keep a small rolling fallback window.
            candidate_indexes.update(range(max(0, len(kept) - 48), len(kept)))
        return sorted(candidate_indexes)

    def register_kept_index(index: int, compact: str) -> None:
        for key in build_bucket_keys(compact):
            bucket_to_indexes.setdefault(key, []).append(index)

    for item in chunks:
        current_shingles = char_ngrams(item["raw_text"])
        current_compact = compact_text(item["raw_text"])
        duplicate_idx = None

        for idx in collect_candidate_indexes(current_compact):
            existing = kept[idx]
            existing_compact = kept_compact_texts[idx]
            if abs(len(current_compact) - len(existing_compact)) > 120:
                continue
            text_sim = jaccard_similarity(current_shingles, kept_shingles[idx])
            if text_sim < TEXT_SIM_THRESHOLD:
                continue

            emb_sim = cosine_similarity(
                get_embedding_for_text(item["embedding_text"]),
                get_embedding_for_text(existing["embedding_text"])
            )
            if emb_sim >= EMBED_SIM_THRESHOLD:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            item["duplicate_cluster_id"] = len(kept)
            kept.append(item)
            kept_shingles.append(current_shingles)
            kept_compact_texts.append(current_compact)
            register_kept_index(len(kept) - 1, current_compact)
        else:
            canonical = choose_canonical_chunk(kept[duplicate_idx], item)
            if canonical is item:
                item["duplicate_cluster_id"] = kept[duplicate_idx]["duplicate_cluster_id"]
                kept[duplicate_idx] = item
                kept_shingles[duplicate_idx] = current_shingles
                kept_compact_texts[duplicate_idx] = current_compact
                register_kept_index(duplicate_idx, current_compact)

    return kept


def reindex_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = {}
    for item in chunks:
        key = (item["doc_id"], item["section_id"])
        counter.setdefault(key, 0)
        item["chunk_in_section"] = counter[key]
        counter[key] += 1
    for item in chunks:
        item["chunk_uid"] = str(item.get("chunk_uid") or build_chunk_uid(item.get("source", ""), item))
    for idx, item in enumerate(chunks):
        prev_uid = chunks[idx - 1]["chunk_uid"] if idx > 0 and chunks[idx - 1].get("doc_id") == item.get("doc_id") else ""
        next_uid = chunks[idx + 1]["chunk_uid"] if idx + 1 < len(chunks) and chunks[idx + 1].get("doc_id") == item.get("doc_id") else ""
        item["prev_chunk_uid"] = prev_uid
        item["next_chunk_uid"] = next_uid
    return chunks


# =========================
# 10. Preview / process
# =========================
def safe_console_text(value: Any) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def print_preview(data: List[Dict[str, Any]], top_k: int = 5):
    print(f"Total chunks: {len(data)}")
    print("=" * 100)
    for item in data[:top_k]:
        print(f"doc_id: {safe_console_text(item['doc_id'])}")
        print(f"doc_title: {safe_console_text(item['doc_title'])}")
        print(f"section_id: {safe_console_text(item['section_id'])}")
        print(f"chunk_in_section: {safe_console_text(item['chunk_in_section'])}")
        print(f"chunk_type: {safe_console_text(item.get('chunk_type', 'paragraph'))}")
        header_path = ' > '.join(item['header_path']) if item['header_path'] else '(none)'
        print(f"header_path: {safe_console_text(header_path)}")
        print(f"text_length: {safe_console_text(item['text_length'])}")
        print(f"duplicate_cluster_id: {safe_console_text(item.get('duplicate_cluster_id', -1))}")
        print(f"overlap_text: {safe_console_text(item.get('overlap_text', '')[:80])}")
        print(f"text_preview: {safe_console_text(item['text'][:260])}")
        print(f"embedding_text_len: {safe_console_text(len(item.get('embedding_text', '')))}")
        print("-" * 100)


def process_one_input_file_with_hierarchy(input_file: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    documents, clean_stats = load_cleaned_documents(input_file)
    source_profiles = {str(doc.get("source_text_profile") or resolve_text_source_profile(input_file)) for doc in documents}
    source_profile_text = ", ".join(sorted(source_profiles)) if source_profiles else resolve_text_source_profile(input_file)
    use_embedding_refinement = any(should_use_embedding_refinement(profile) for profile in source_profiles) if source_profiles else should_use_embedding_refinement(resolve_text_source_profile(input_file))
    if clean_stats:
        print(
            f"[INFO] Pre-cleaned {input_file.name}: "
            f"chars {clean_stats.get('input_chars', 0)} -> {clean_stats.get('output_chars', 0)}, "
            f"removed metadata={clean_stats.get('removed_metadata_lines', 0)}, "
            f"ocr={clean_stats.get('removed_ocr_lines', 0)}, "
            f"low_value={clean_stats.get('removed_low_value_sections', 0)}"
        )

    print(f"[INFO] Loaded documents from {input_file.name}: {len(documents)}")
    print(f"[INFO] Source text profile for {input_file.name}: {source_profile_text}")
    if not use_embedding_refinement:
        print(f"[INFO] Clean-text mode: skip embedding-assisted semantic merge and near deduplicate for {input_file.name}")

    all_initial_chunks = []
    for doc in documents:
        text = doc.get("text", "").strip()
        if not text:
            continue
        all_initial_chunks.extend(build_initial_chunks_for_doc(doc))

    all_initial_chunks = second_pass_chunk_cleanup(all_initial_chunks)
    print(f"[INFO] Initial chunk count for {input_file.name}: {len(all_initial_chunks)}")

    pre_filter_chunks = list(all_initial_chunks)

    if QUALITY_FILTER_EARLY:
        all_initial_chunks = quality_filter(all_initial_chunks)
        print(f"[INFO] Early filtered chunk count for {input_file.name}: {len(all_initial_chunks)}")

    doc_total_chars = max((int(item.get("doc_total_chars", 0)) for item in all_initial_chunks), default=0)
    min_expected_chunks = get_min_expected_chunk_count(doc_total_chars)
    if len(all_initial_chunks) < min_expected_chunks:
        print(
            f"[INFO] Low chunk count detected for {input_file.name}: "
            f"{len(all_initial_chunks)} < {min_expected_chunks}, rerunning aggressive split"
        )
        aggressive_chunks = second_pass_chunk_cleanup(pre_filter_chunks, aggressive=True)
        if QUALITY_FILTER_EARLY:
            aggressive_chunks = quality_filter(aggressive_chunks)
            print(f"[INFO] Aggressive filtered chunk count for {input_file.name}: {len(aggressive_chunks)}")
        all_initial_chunks = aggressive_chunks

    if FAST_MODE:
        semantic_merged_chunks = all_initial_chunks
        print(f"[INFO] FAST_MODE=ON: skip semantic merge for {input_file.name}")
    elif not use_embedding_refinement:
        semantic_merged_chunks = all_initial_chunks
    else:
        semantic_merged_chunks = semantic_merge_adjacent_chunks(all_initial_chunks)

    overlap_chunks = add_sentence_overlap(semantic_merged_chunks, OVERLAP_SENTENCES)
    quality_chunks = quality_filter(overlap_chunks)
    exact_dedup_chunks = exact_deduplicate(quality_chunks)

    if FAST_MODE:
        near_dedup_chunks = exact_dedup_chunks
        print(f"[INFO] FAST_MODE=ON: skip near deduplicate for {input_file.name}")
    elif not use_embedding_refinement:
        near_dedup_chunks = exact_dedup_chunks
    else:
        near_dedup_chunks = near_deduplicate_fast(exact_dedup_chunks)

    final_chunks = reindex_chunks(near_dedup_chunks)
    parent_chunks = build_parent_chunks(final_chunks)

    print(f"[INFO] Final chunk count for {input_file.name}: {len(final_chunks)}")
    print(f"[INFO] Parent chunk count for {input_file.name}: {len(parent_chunks)}")
    return final_chunks, parent_chunks


def process_one_input_file(input_file: Path) -> List[Dict[str, Any]]:
    final_chunks, _ = process_one_input_file_with_hierarchy(input_file)
    return final_chunks


def main():
    # Main pipeline: discover inputs, validate runtime, process each file, and write outputs.
    input_files = iter_input_files(INPUT_PATH)
    if not input_files:
        raise FileNotFoundError(f"No supported input files found under: {INPUT_PATH}")

    preflight_runtime_checks(input_files)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    index = []

    for input_file in input_files:
        output_file = build_output_file_path(input_file, INPUT_PATH, OUTPUT_DIR)

        if output_file.exists() and not FORCE_REPROCESS:
            print(f"[SKIP] Already processed: {input_file}")
            skipped += 1
            index.append({
                "source_file": str(input_file),
                "output_file": str(output_file),
                "status": "skipped",
            })
            continue

        try:
            final_chunks, parent_chunks = process_one_input_file_with_hierarchy(input_file)
            write_chunks_file(output_file, input_file, final_chunks, parent_chunks)
            print_preview(final_chunks, top_k=2)
            processed += 1
            index.append({
                "source_file": str(input_file),
                "output_file": str(output_file),
                "status": "processed",
                "chunk_count": len(final_chunks),
                "parent_chunk_count": len(parent_chunks),
            })
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed to process {input_file}: {e}")
            if "CUDA device was requested" in str(e) or "Missing dependency" in str(e):
                raise
            index.append({
                "source_file": str(input_file),
                "output_file": str(output_file),
                "status": "failed",
                "error": str(e),
            })

    index_file = OUTPUT_DIR / "_index.json"
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")
    print(f"[INFO] Chunk files stored under: {OUTPUT_DIR}")
    print(f"[INFO] Index written to: {index_file}")
    print("[INFO] Final embedding and Qdrant sync now run separately via python -m rag_pipeline.ingest.embedding_qdrant")


if __name__ == "__main__":
    main()
