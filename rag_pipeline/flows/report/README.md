# Report Flow

This package contains the canonical implementation for the full report flow:

- `full_report.py`: end-to-end report orchestration.
- `evidence_extractor.py`: extracts clean evidence from writer packages.
- `reformatter_agent.py`: rewrites the final publishable report.
- `review_pipeline.py`, `review_agent.py`, `llm_review_agent.py`: final report cleanup and optional review.

The old top-level files in `current_rag_pipeline/` are kept as compatibility wrappers, so existing scripts and tests can continue to import them.

