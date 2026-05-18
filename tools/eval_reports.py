from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_STRUCTURE_TERMS = [
    "一句话结论",
    "核心观点",
    "关键数据",
    "风险与反证",
    "行动建议",
]
DEFAULT_MECHANISM_TERMS = ["因为", "由于", "导致", "传导", "机制", "驱动", "所以", "转化"]
DEFAULT_COUNTER_TERMS = ["反证", "边界", "推翻", "风险", "不成立", "放弃条件"]
DEFAULT_DECISION_TERMS = ["进入", "观望", "放弃", "优先", "触发器", "验证指标", "行动建议"]
DEFAULT_DATA_PATTERN = re.compile(r"\d|%|亿元|万元|亿美元|万台|CAGR|同比|增速|毛利|渗透率", re.I)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text_hits(text: str, terms: Iterable[Any]) -> List[str]:
    hits: List[str] = []
    for term in terms:
        value = str(term or "").strip()
        if value and value in text:
            hits.append(value)
    return hits


def load_cases(path: Path) -> List[Dict[str, Any]]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cases: List[Dict[str, Any]] = []
    for item in paths:
        payload = json.loads(item.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            cases.extend(case for case in payload if isinstance(case, dict))
        elif isinstance(payload, dict):
            cases.extend(case for case in _as_list(payload.get("cases")) if isinstance(case, dict))
            if not payload.get("cases") and payload.get("case_id"):
                cases.append(payload)
    return cases


def find_report_for_case(case: Dict[str, Any], report_files: Sequence[Path], *, index: int = 0, pair_by_order: bool = False) -> Path | None:
    case_id = str(case.get("case_id") or "").strip().lower()
    aliases = [case_id, str(case.get("report_slug") or case.get("slug") or "").strip().lower()]
    if case_id and "_" in case_id:
        aliases.append(case_id.replace("_", "-"))
    for alias in [item for item in aliases if item]:
        for report in report_files:
            haystack = f"{report.parent.name.lower()} {report.stem.lower()}"
            if alias in haystack:
                return report
    query_terms = [str(item).strip().lower() for item in _as_list(case.get("must_cover")) if str(item).strip()]
    if query_terms:
        for report in report_files:
            haystack = f"{report.parent.name.lower()} {report.stem.lower()}"
            if sum(1 for term in query_terms if term in haystack) >= 2:
                return report
    if pair_by_order and index < len(report_files):
        return report_files[index]
    if len(report_files) == 1:
        return report_files[0]
    return None


def score_report_text(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    required = [str(item) for item in _as_list(case.get("must_cover")) if str(item).strip()]
    structure_terms = _as_list(case.get("ideal_structure")) or DEFAULT_STRUCTURE_TERMS
    forbidden = [str(item) for item in _as_list(case.get("forbidden_phrases")) if str(item).strip()]

    required_hits = _text_hits(text, required)
    structure_hits = _text_hits(text, structure_terms)
    forbidden_hits = _text_hits(text, forbidden)
    mechanism_hits = _text_hits(text, DEFAULT_MECHANISM_TERMS)
    counter_hits = _text_hits(text, DEFAULT_COUNTER_TERMS)
    decision_hits = _text_hits(text, DEFAULT_DECISION_TERMS)
    has_data = bool(DEFAULT_DATA_PATTERN.search(text))
    has_sources = bool(re.search(r"\[[SWE]-?\d+|来源|附录|http", text, re.I))
    has_coverage_matrix = "证据覆盖矩阵" in text or "璇佹嵁瑕嗙洊鐭╅樀" in text
    has_metric_table = "指标口径" in text or "鎸囨爣鍙ｅ緞" in text

    required_score = 30 * (len(required_hits) / max(len(required), 1))
    structure_score = 20 * (len(structure_hits) / max(len(structure_terms), 1))
    evidence_score = (8 if has_sources else 0) + (8 if has_data else 0) + (2 if has_coverage_matrix else 0) + (2 if has_metric_table else 0)
    analysis_score = min(20, len(mechanism_hits) * 4 + len(counter_hits) * 4 + len(decision_hits) * 3)
    penalty = min(30, len(forbidden_hits) * 10)
    score = round(max(0.0, min(100.0, required_score + structure_score + evidence_score + analysis_score - penalty)), 2)

    return {
        "score": score,
        "required_hits": required_hits,
        "missing_required": [item for item in required if item not in required_hits],
        "structure_hits": structure_hits,
        "missing_structure": [str(item) for item in structure_terms if str(item) not in structure_hits],
        "has_data": has_data,
        "has_sources": has_sources,
        "has_coverage_matrix": has_coverage_matrix,
        "has_metric_table": has_metric_table,
        "mechanism_hits": mechanism_hits,
        "counter_hits": counter_hits,
        "decision_hits": decision_hits,
        "forbidden_hits": forbidden_hits,
        "passed": score >= 75 and not forbidden_hits,
    }


def evaluate(cases: Sequence[Dict[str, Any]], reports_dir: Path | None, *, pair_by_order: bool = False) -> Dict[str, Any]:
    report_files = sorted(reports_dir.rglob("*.md")) if reports_dir and reports_dir.exists() else []
    results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        report = find_report_for_case(case, report_files, index=index, pair_by_order=pair_by_order)
        if report is None:
            results.append(
                {
                    "case_id": case.get("case_id"),
                    "query": case.get("query"),
                    "status": "no_report",
                    "passed": False,
                    "score": 0,
                }
            )
            continue
        text = report.read_text(encoding="utf-8", errors="ignore")
        score = score_report_text(text, case)
        results.append(
            {
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "status": "evaluated",
                "report_path": str(report),
                **score,
            }
        )

    evaluated = [item for item in results if item.get("status") == "evaluated"]
    average = round(sum(float(item.get("score") or 0) for item in evaluated) / max(len(evaluated), 1), 2)
    return {
        "cases_loaded": len(cases),
        "reports_found": len(report_files),
        "evaluated": len(evaluated),
        "average_score": average,
        "passed": bool(evaluated) and all(bool(item.get("passed")) for item in evaluated),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated reports against golden case expectations.")
    parser.add_argument("--case-set", default="golden_cases/minimal_cases.json", help="JSON file or directory containing golden cases.")
    parser.add_argument("--reports-dir", default="", help="Directory containing generated Markdown reports.")
    parser.add_argument("--output", default="eval_results.json", help="Where to write JSON evaluation results.")
    parser.add_argument("--pair-by-order", action="store_true", help="Pair cases with reports by sorted order when filenames do not match.")
    args = parser.parse_args()

    case_path = Path(args.case_set)
    cases = load_cases(case_path)
    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    result = evaluate(cases, reports_dir, pair_by_order=bool(args.pair_by_order))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("cases_loaded", "reports_found", "evaluated", "average_score", "passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
