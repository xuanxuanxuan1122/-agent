"""Quality regression utilities for report production."""

from .conversion_summary import build_quality_conversion_summary
from .executor import build_quality_execution_plan, run_quality_regression_execution
from .regression import (
    build_run_quality_snapshot,
    load_quality_snapshots_from_paths,
    summarize_quality_regression_suite,
    summarize_repair_effectiveness,
    summarize_topic_regression,
    validate_golden_topic_suite,
)

__all__ = [
    "build_quality_execution_plan",
    "build_quality_conversion_summary",
    "build_run_quality_snapshot",
    "load_quality_snapshots_from_paths",
    "run_quality_regression_execution",
    "summarize_quality_regression_suite",
    "summarize_repair_effectiveness",
    "summarize_topic_regression",
    "validate_golden_topic_suite",
]
