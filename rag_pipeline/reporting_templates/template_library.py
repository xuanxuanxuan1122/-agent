from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


ASSET_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = ASSET_DIR / "layout_templates.json"
PROMPT_PATH = ASSET_DIR / "analysis_prompts.md"


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def load_template_library(path: Path | None = None) -> Dict[str, Any]:
    target = path or TEMPLATE_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def list_templates(path: Path | None = None) -> List[Dict[str, Any]]:
    data = load_template_library(path)
    return [item for item in _as_list(data.get("templates")) if isinstance(item, dict)]


def get_template(template_id: str, path: Path | None = None) -> Dict[str, Any]:
    target = str(template_id or "").strip()
    for template in list_templates(path):
        if str(template.get("id") or "") == target:
            return template
    raise KeyError(f"unknown report template id: {target}")


def validate_template_library(path: Path | None = None) -> List[str]:
    data = load_template_library(path)
    errors: List[str] = []
    if not data.get("version"):
        errors.append("missing version")
    templates = _as_list(data.get("templates"))
    if not templates:
        errors.append("templates is empty")
    seen = set()
    for index, template in enumerate(templates, start=1):
        if not isinstance(template, dict):
            errors.append(f"template #{index} is not an object")
            continue
        template_id = str(template.get("id") or "").strip()
        if not template_id:
            errors.append(f"template #{index} missing id")
        if template_id in seen:
            errors.append(f"duplicated template id: {template_id}")
        seen.add(template_id)
        if not template.get("name"):
            errors.append(f"{template_id or index}: missing name")
        if not _as_list(template.get("body_structure")):
            errors.append(f"{template_id or index}: missing body_structure")
        if not isinstance(template.get("quality_bar"), dict):
            errors.append(f"{template_id or index}: missing quality_bar")
    return errors


KEYWORD_RULES = [
    (
        "multi_sector_policy_impact",
        [
            "多行业",
            "多个行业",
            "半导体",
            "新能源",
            "消费品",
            "互联网",
            "关税",
            "出口管制",
            "市场准入",
            "制裁",
            "地缘",
        ],
    ),
    ("company_due_diligence", ["公司", "尽调", "财务", "客户结构", "估值", "投资判断"]),
    ("market_entry", ["进入", "切入", "立项", "市场进入", "渠道", "BD"]),
    ("competitive_analysis", ["竞品", "竞争对手", "对比", "竞争格局", "定位"]),
    ("technology_trend", ["技术路线", "技术趋势", "产品落地", "成熟度", "替代路径", "商业化"]),
    ("industry_deep_report", ["行业", "产业链", "市场空间", "增长", "机会", "风险"]),
]


def select_template(query: str, path: Path | None = None) -> Dict[str, Any]:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    if not text:
        return get_template("industry_deep_report", path)

    scored: List[tuple[int, int, str]] = []
    for priority, (template_id, keywords) in enumerate(KEYWORD_RULES):
        score = 0
        for keyword in keywords:
            if keyword and keyword.lower() in text.lower():
                score += 1
        if score:
            scored.append((score, -priority, template_id))
    if not scored:
        return get_template("industry_deep_report", path)
    scored.sort(reverse=True)
    return get_template(scored[0][2], path)


def prompt_path() -> Path:
    return PROMPT_PATH


def _print_template_summary(template: Dict[str, Any]) -> None:
    print(f"{template.get('id')}: {template.get('name')}")
    print(f"  reader_goal: {template.get('reader_goal')}")
    print("  body_structure:")
    for chapter in _as_list(template.get("body_structure")):
        if not isinstance(chapter, dict):
            continue
        print(f"    - {chapter.get('chapter_role')}: {chapter.get('title_pattern')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline report layout template library preview")
    parser.add_argument("--list", action="store_true", help="List available template ids.")
    parser.add_argument("--select", default="", help="Select a template for a query string.")
    parser.add_argument("--show", default="", help="Show one template by id.")
    parser.add_argument("--validate", action="store_true", help="Validate template JSON structure.")
    args = parser.parse_args()

    if args.validate:
        errors = validate_template_library()
        if errors:
            print("INVALID")
            for error in errors:
                print(f"- {error}")
            return 1
        print("OK")
        return 0

    if args.list:
        for template in list_templates():
            print(f"{template.get('id')}\t{template.get('name')}")
        return 0

    if args.show:
        _print_template_summary(get_template(args.show))
        return 0

    if args.select:
        _print_template_summary(select_template(args.select))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
