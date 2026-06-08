from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "fast_contract: fast current-contract tests")
    config.addinivalue_line("markers", "pipeline_contract: stage pipeline contract tests")
    config.addinivalue_line("markers", "slow_integration: long-running integration tests")
    config.addinivalue_line("markers", "legacy_clean_contract: tests for the old clean-blocking delivery contract")


def pytest_sessionstart(session):
    os.environ["BRAIN_ENABLE_LLM_RESEARCH_PLANNER"] = "0"
    os.environ["BRAIN_ENABLE_LLM_PROBLEM_FRAMING"] = "0"
    os.environ["BRAIN_ENABLE_LLM_EVIDENCE_ANALYSIS"] = "false"
    os.environ["REPORT_ENABLE_LLM_CHAPTER_NARRATIVE"] = "false"
    os.environ["REPORT_ENABLE_LLM_BODY_REWRITE"] = "false"
    os.environ["REPORT_ENABLE_LLM_REWRITE"] = "false"
    os.environ["REPORT_ENABLE_FINAL_AUDIT"] = "0"
    os.environ["BRAIN_ENABLE_POST_QA_REPAIR"] = "false"
    os.environ.setdefault("REPORT_BLOCK_ON_QA_FAILURE", "false")
    os.environ.setdefault("REPORT_WRITE_CLEAN_REPORT", "false")


def pytest_collection_modifyitems(config, items):
    slow_files = {
        "test_enterprise_layout.py",
        "test_figure_table_contract.py",
    }
    legacy_files = {
        "test_report_quality_regressions.py",
    }
    legacy_tests = {
        "test_final_reference_analysis_uses_dynamic_report_logic_not_fixed_industry_chain",
    }
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in slow_files:
            item.add_marker(pytest.mark.slow_integration)
        if filename in legacy_files or item.name in legacy_tests:
            item.add_marker(pytest.mark.legacy_clean_contract)
