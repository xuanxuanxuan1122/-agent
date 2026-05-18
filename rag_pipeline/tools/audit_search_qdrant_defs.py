"""Audit duplicate top-level definitions and constant assignments in the search engine module.

This script helps track technical debt in the monolithic retrieval file by
surfacing duplicate class/function names and duplicate top-level constant-like
assignments with their line locations.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def collect_definitions(target_file: Path) -> Dict[str, List[Tuple[int, int, str]]]:
    tree = ast.parse(target_file.read_text(encoding="utf-8"))
    defs: Dict[str, List[Tuple[int, int, str]]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            entries = defs.setdefault(node.name, [])
            entries.append((int(node.lineno), int(getattr(node, "end_lineno", node.lineno)), type(node).__name__))
    return defs


def is_auditable_assignment_name(name: str) -> bool:
    return bool(name) and (name.isupper() or name.startswith("_"))


def collect_assignments(target_file: Path) -> Dict[str, List[Tuple[int, int, str]]]:
    tree = ast.parse(target_file.read_text(encoding="utf-8"))
    assignments: Dict[str, List[Tuple[int, int, str]]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and is_auditable_assignment_name(target.id):
                    entries = assignments.setdefault(target.id, [])
                    entries.append((int(node.lineno), int(getattr(node, "end_lineno", node.lineno)), "Assign"))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if is_auditable_assignment_name(node.target.id):
                entries = assignments.setdefault(node.target.id, [])
                entries.append((int(node.lineno), int(getattr(node, "end_lineno", node.lineno)), "AnnAssign"))
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect duplicate top-level definitions and constant assignments in a Python file."
    )
    parser.add_argument(
        "--target",
        default=str(Path(__file__).resolve().parents[1] / "search" / "engine.py"),
        help="Target Python file to audit.",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Return non-zero when duplicate names are found.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"[ERROR] Target file not found: {target}")
        return 1

    defs = collect_definitions(target)
    assignments = collect_assignments(target)
    duplicate_defs = {name: entries for name, entries in defs.items() if len(entries) > 1}
    duplicate_assignments = {name: entries for name, entries in assignments.items() if len(entries) > 1}

    if not duplicate_defs and not duplicate_assignments:
        print(f"[OK] No duplicate top-level defs or assignments found in: {target}")
        return 0

    print(f"[WARN] Duplicate top-level items found in: {target}")
    if duplicate_defs:
        print("[defs]")
        for name in sorted(duplicate_defs):
            entries = duplicate_defs[name]
            locations = ", ".join([f"{kind}@{start}-{end}" for start, end, kind in entries])
            print(f" - {name}: {locations}")
    if duplicate_assignments:
        print("[assignments]")
        for name in sorted(duplicate_assignments):
            entries = duplicate_assignments[name]
            locations = ", ".join([f"{kind}@{start}-{end}" for start, end, kind in entries])
            print(f" - {name}: {locations}")

    return 2 if args.fail_on_duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
