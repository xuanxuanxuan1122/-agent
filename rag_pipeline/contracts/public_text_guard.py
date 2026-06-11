from __future__ import annotations

import re
from typing import Any, Dict, List


MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)|!\([^)]+\)")
STATIC_ASSET_RE = re.compile(r"(?:^|/|\\)(?:_next/static|static/media|assets?/(?:img|image|logo))", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
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
    if STATIC_ASSET_RE.search(value) or "_next/static" in value.lower() or "/static/media" in value.lower():
        reasons.append("next_static_asset")
    if NAVIGATION_RE.search(value):
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
