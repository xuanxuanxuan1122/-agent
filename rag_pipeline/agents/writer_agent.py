"""Compatibility entry point for the report writer agent.

The historical implementation in this module had accumulated multiple
overwritten renderer versions.  The active implementation now lives in
``writer_agent_clean.py`` so imports keep working while the writer has a
single source of truth.
"""

from __future__ import annotations

try:
    from .writer_agent_clean import (
        WriterAgentState,
        build_writer_report,
        create_writer_agent_tool,
        main,
        run_writer_agent,
        writer_agent_tool,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from writer_agent_clean import (  # type: ignore
        WriterAgentState,
        build_writer_report,
        create_writer_agent_tool,
        main,
        run_writer_agent,
        writer_agent_tool,
    )


__all__ = [
    "WriterAgentState",
    "build_writer_report",
    "create_writer_agent_tool",
    "main",
    "run_writer_agent",
    "writer_agent_tool",
]


if __name__ == "__main__":
    raise SystemExit(main())
