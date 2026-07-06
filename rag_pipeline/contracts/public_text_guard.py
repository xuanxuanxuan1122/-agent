from __future__ import annotations

import re
from typing import Any, Dict, List


MARKDOWN_IMAGE_RE = re.compile(r"!\s*\[[^\]]*](?:\([^)]+\))?|!\s*\([^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]{0,80}\]\((?:https?:)?//[^)]+|/[^)]+\)")
STATIC_ASSET_RE = re.compile(r"(?:^|/|\\)(?:_next/static|static/media|assets?/(?:img|image|logo))", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
WEB_CHROME_ASSET_RE = re.compile(
    r"(?:^|/|\\)(?:newstatic/images|assets?/(?:img|image|logo))"
    r"|(?:^|/|\\)logo\.(?:gif|png|jpg|jpeg|webp|svg)\b",
    re.I,
)
WEB_CHROME_NAV_RE = re.compile(
    r"homepage|data\s+center|market\s+center|quote\s+center|choice\s+data|"
    r"\u9996\u9875|\u767b\u5f55|\u6ce8\u518c|\u6570\u636e\u4e2d\u5fc3|"
    r"\u5168\u7403\u8d22\u7ecf\u5feb\u8baf|\u8d22\u7ecf\u5feb\u8baf|\u884c\u60c5\u4e2d\u5fc3|"
    r"Choice\s*\u6570\u636e|\u4e1c\u65b9\u8d22\u5bcc|\u81ea\u9009\u80a1|\u80a1\u5427|"
    r"\u5b66\u4e60\u8d85\u5e02|\u878d\u5408\u95e8\u6237|IT\s*\u878d\u5408\u95e8\u6237",
    re.I,
)
NAVIGATION_RE = re.compile(
    r"skip\s+to\s+content|product\s+documentation|cookie\s+policy|privacy\s+policy|"
    r"产品\s*!\s*产品|资源\s*!\s*资源|登录\s*注册|首页\s+产品\s+解决方案\s+资源",
    re.I,
)
MENU_DENSE_RE = re.compile(
    r"(?:产品|资源|文档|价格|登录|注册|联系我们|解决方案|客户案例|开发者|控制台|下载)"
    r"(?:\s*[!|/｜·>]\s*|\s+)"
    r"(?:产品|资源|文档|价格|登录|注册|联系我们|解决方案|客户案例|开发者|控制台|下载)"
)


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_public_text(text: Any) -> str:
    value = _text(text)
    value = MARKDOWN_IMAGE_RE.sub("", value)
    value = MARKDOWN_LINK_RE.sub("", value)
    value = URL_RE.sub("", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" |｜/!，,；;")
    return value


def public_text_quality(text: Any) -> Dict[str, Any]:
    value = _text(text)
    reasons: List[str] = []
    if not value:
        return {"ok": False, "severity": "reject", "reasons": ["empty"], "cleaned": ""}
    if MARKDOWN_IMAGE_RE.search(value):
        reasons.append("markdown_image")
    if len(MARKDOWN_LINK_RE.findall(value)) >= 2:
        reasons.append("markdown_link_cluster")
    if STATIC_ASSET_RE.search(value) or "_next/static" in value.lower() or "/static/media" in value.lower():
        reasons.append("next_static_asset")
    if WEB_CHROME_ASSET_RE.search(value):
        reasons.append("static_asset")
    if NAVIGATION_RE.search(value):
        reasons.append("navigation_chrome")
    if WEB_CHROME_NAV_RE.search(value):
        reasons.append("navigation_chrome")
    if MENU_DENSE_RE.search(value):
        reasons.append("menu_chrome")
    if len(URL_RE.findall(value)) >= 3:
        reasons.append("url_cluster")
    cleaned = clean_public_text(value)
    if reasons:
        return {"ok": False, "severity": "reject", "reasons": sorted(set(reasons)), "cleaned": cleaned}
    if not cleaned:
        return {"ok": False, "severity": "reject", "reasons": ["empty_after_cleaning"], "cleaned": cleaned}
    return {"ok": True, "severity": "ok", "reasons": [], "cleaned": cleaned}


def public_text_is_ok(text: Any) -> bool:
    return bool(public_text_quality(text).get("ok"))
