"""Compatibility entry point for the full report generation flow."""

from rag_pipeline.flows.report.full_report import *  # noqa: F401,F403
from rag_pipeline.flows.report.full_report import main


if __name__ == "__main__":
    raise SystemExit(main())

