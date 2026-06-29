from __future__ import annotations

from rag_pipeline.contracts.review_suggestion_contract import (
    make_review_suggestion,
    repair_action_for_review_suggestion,
)


def test_review_suggestion_is_diagnostic_only_and_not_renderable():
    suggestion = make_review_suggestion(
        issue_type="single_source",
        severity="warning",
        target={"chapter_id": "CH-market", "claim_id": "CL-1"},
        suggested_action="rewrite_with_caveat",
        detail={"source_count": 1},
    )

    assert suggestion["schema_version"] == "review_suggestion_v1"
    assert suggestion["diagnostic_only"] is True
    assert suggestion["must_not_render"] is True
    assert suggestion["public_text_allowed"] is False
    assert suggestion["target"]["claim_id"] == "CL-1"
    assert suggestion["detail"]["source_count"] == 1


def test_repair_action_dispatches_non_search_actions_before_search():
    assert repair_action_for_review_suggestion({"issue_type": "body_short", "bound_claim_count": 8}) == "rewrite_with_caveat"
    assert repair_action_for_review_suggestion({"issue_type": "low_claim_conversion"}) == "reanalyze_existing"
    assert repair_action_for_review_suggestion({"issue_type": "chapter_mismatch"}) == "recompose_outline"
    assert repair_action_for_review_suggestion({"issue_type": "section_ref_binding_invalid_only"}) == "reanalyze_existing"
    assert repair_action_for_review_suggestion({"issue_type": "section_ref_binding_unbound"}) == "recompose_outline"
    assert repair_action_for_review_suggestion({"issue_type": "missing_case"}) == "search_more"
    assert repair_action_for_review_suggestion({"issue_type": "single_source"}) == "rewrite_with_caveat"
