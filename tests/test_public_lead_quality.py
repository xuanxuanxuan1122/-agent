from rag_pipeline.agents.chapter_argument_agent import _is_snippet_like_public_text, run_chapter_argument_agent
from rag_pipeline.agents.public_report_sanitizer import sanitize_public_markdown


def test_search_snippet_lead_is_rewritten_or_removed_from_public_section():
    snippet = (
        "Over the weekend, the Futian district government in Shenzhen, "
        "Guangdong province, unveiled the AI Digital Employee 2.0..."
    )
    packages = run_chapter_argument_agent(
        report_blueprint={
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "AI Agent deployment",
                    "chapter_question": "Where is deployment becoming visible?",
                }
            ]
        },
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "Deployment signals",
                        "required_evidence_refs": ["E1"],
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_case",
                "block_type": "case_comparison",
                "claim": snippet,
                "reasoning": snippet,
                "mechanism": snippet,
                "counter_evidence": "",
                "supporting_facts": [snippet],
                "evidence_basis": [snippet],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "fact_card_to_block_match": True,
                "claim_strength": "directional",
            }
        ],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "case_evidence": [
                    {
                        "evidence_id": "E1",
                        "ref": "E1",
                        "source_ref": "S1",
                        "distilled_fact": snippet,
                        "public_fact_card": {"source_ref": "S1", "distilled_fact": snippet},
                    }
                ],
            }
        ],
    )

    public_text = "\n".join(
        str(value or "")
        for section in packages[0]["sections"]
        for value in (
            section.get("claim"),
            section.get("reasoning"),
            section.get("mechanism"),
            *(block.get("text") for block in section.get("render_blocks", []) if isinstance(block, dict)),
        )
    )
    assert "Over the weekend" not in public_text
    assert "Futian district government" not in public_text
    assert "..." not in public_text


def test_title_prefix_and_empty_parentheses_are_not_rendered_as_public_lead():
    snippet = (
        "\u76d8\u70b9\u4eba\u5f62\u673a\u5668\u4eba\u4ea7\u4e1a\u94fe2025\u5e74\u201c\u6210\u7ee9\u5355\u201d\uff08\uff09\uff1a"
        "\u4eba\u5f62\u673a\u5668\u4eba\u6b63\u4ece\u5b9e\u9a8c\u5ba4\u8d70\u8fdb\u73b0\u5b9e..."
    )
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "\u5546\u4e1a\u5316\u9a8c\u8bc1"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "ch_01_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "\u91cf\u4ea7\u4ea4\u4ed8\u662f\u5426\u52a0\u901f",
                        "required_evidence_refs": ["E1"],
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "ch_01_case",
                "block_type": "case_comparison",
                "claim": snippet,
                "reasoning": snippet,
                "mechanism": snippet,
                "supporting_facts": [snippet],
                "used_fact_refs": ["E1"],
                "evidence_refs": ["E1"],
                "fact_card_to_block_match": True,
            }
        ],
    )

    public_text = "\n".join(str(section.get("claim") or "") for section in packages[0]["sections"])
    assert "\uff08\uff09" not in public_text
    assert "\uff1a" not in public_text[:80]
    assert "..." not in public_text


def test_news_and_title_publisher_snippets_are_rejected():
    assert _is_snippet_like_public_text(
        "\u4ece\u5de5\u5177\u5230\u751f\u4ea7\u529b\u7684\u91cd\u5851-\u963f\u91cc\u4e91\u5f00\u53d1\u8005\u793e\u533a\uff1a"
        "\u5f53 AI \u4ece\u80fd\u804a\u5929\u7684\u52a9\u624b\u8fdb\u5316\u4e3a\u4f1a\u4e3b\u52a8\u5e72\u6d3b\u7684\u7cfb\u7edf"
    )
    assert _is_snippet_like_public_text(
        "\u4e00\u76c6\u51b7\u6c34\u7ec8\u4e8e\u6cfc\u5230\u4e86\u706b\u70ed\u7684\u5177\u8eab\u667a\u80fd\u8d5b\u9053\uff0c"
        "\u91d1\u6c99\u6c5f\u521b\u6295\u4e3b\u7ba1\u5408\u4f19\u4eba\u6731\u5578\u864e\u7684\u4e00\u53e5\u201c\u6211\u4eec\u6b63\u6279\u91cf\u9000\u51fa\u201d"
    )
    assert _is_snippet_like_public_text(
        "\u4eca\u5e743\u6708\u4efd\uff0c\u96c6\u5718\u767c\u4f48\u4e86\u300cPhancy\u300d\u6d88\u8cbb\u96fb\u5b50\u696d\u52d9"
    )


def test_public_sanitizer_removes_internal_public_terms_and_connector_title():
    markdown = "\n".join(
        [
            "# Report",
            "### \u4e3a\u6b64\u843d\u5730\u5230\u54ea\u4e00\u6b65",
            "The market metric and risk boundary should not leak into public prose.",
            "block_affinity: metric_reconciliation",
            "analysis_variable: market metric",
            "evidence_cards: EV-123",
        ]
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "market metric" not in cleaned
    assert "risk boundary" not in cleaned
    assert "block_affinity" not in cleaned
    assert "analysis_variable" not in cleaned
    assert "evidence_cards" not in cleaned
    assert "EV-123" not in cleaned
    assert "\u4e3a\u6b64\u843d\u5730\u5230\u54ea\u4e00\u6b65" not in cleaned
