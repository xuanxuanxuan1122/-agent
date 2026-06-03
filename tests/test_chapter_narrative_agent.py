import copy

from rag_pipeline.agents.chapter_narrative_agent import run_chapter_narrative


def _section(section_id: str = "s1", *, paragraph: str | None = None):
    text = paragraph or "Salesforce disclosed Agentforce workflow deployment for customer service [1]."
    return {
        "section_id": section_id,
        "chapter_id": "ch_01",
        "section_title": "Workflow deployment",
        "block_type": "case_comparison",
        "claim_strength": "moderate",
        "body_composition_status": "composed",
        "claim": "stale claim text that should not be used as the narrative source",
        "reasoning": text,
        "mechanism": text,
        "used_fact_refs": ["EV-1"],
        "evidence_refs": ["EV-1"],
        "citation_refs": ["[1]"],
        "supporting_facts": ["Salesforce disclosed Agentforce workflow deployment for customer service."],
        "render_blocks": [
            {"type": "paragraph", "label": "", "text": text},
            {"type": "table", "label": "deployment metrics", "rows": [["metric", "value"]]},
        ],
        "evidence_backed": True,
    }


def _chapter(*sections):
    return {
        "chapter_id": "ch_01",
        "chapter_title": "Demand validation",
        "chapter_question": "Can AI agents move from trials into workflow deployment?",
        "chapter_summary": {"key_takeaway": "Deployment signals exist."},
        "sections": [dict(item) for item in sections],
    }


def _valid_response():
    return {
        "payload": {
            "chapter_lead": "本章先判断企业智能体是否已经进入流程部署，再看样本能否支撑付费转化。",
            "sections": [
                {
                    "section_id": "s1",
                    "paragraph": "Salesforce 披露 Agentforce 已进入客服工作流部署，这说明企业智能体的需求不只停留在演示，而是开始嵌入可运行流程 [1]。",
                    "used_fact_refs": ["EV-1"],
                    "citation_refs": ["[1]"],
                },
                {
                    "section_id": "s2",
                    "paragraph": "Salesforce 披露 Agentforce 已进入客服工作流部署，第二个段落承接这一事实，强调复制性仍要看流程和权限集成 [1]。",
                    "used_fact_refs": ["EV-1"],
                    "citation_refs": ["[1]"],
                },
            ],
        }
    }


def test_chapter_narrative_disabled_leaves_packages_unchanged(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "false")
    chapters = [_chapter(_section())]
    original = copy.deepcopy(chapters)

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={},
        llm_config={},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert rewritten == original
    assert diagnostics["enabled"] is False
    assert diagnostics["skipped_reason"] == "disabled"


def test_chapter_narrative_rewrites_sections_and_preserves_non_paragraph_blocks(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", "1")
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json", lambda **kwargs: _valid_response())

    chapters = [_chapter(_section("s1"), _section("s2"))]
    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={"report_profile": "industry_research_report"},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert diagnostics["attempted_count"] == 1
    assert diagnostics["success_count"] == 1
    assert rewritten[0]["lead"].startswith("本章先判断")
    blocks = rewritten[0]["sections"][0]["render_blocks"]
    assert blocks[0]["type"] == "paragraph"
    assert "可运行流程" in blocks[0]["text"]
    assert any(block.get("type") == "table" for block in blocks[1:])
    assert rewritten[0]["sections"][0]["chapter_narrative_status"] == "rewritten"


def test_chapter_narrative_default_minimum_allows_two_evidence_sections(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.delenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", raising=False)
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_CACHE_ENABLED", "false")
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: True)
    calls = {"count": 0}

    def fake_call(**kwargs):
        calls["count"] += 1
        return _valid_response()

    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json", fake_call)

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=[_chapter(_section("s1"), _section("s2"))],
        report_blueprint={"report_profile": "industry_research_report"},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert calls["count"] == 1
    assert diagnostics["attempted_count"] == 1
    assert diagnostics["success_count"] == 1
    assert rewritten[0]["chapter_narrative_status"] == "rewritten"


def test_chapter_narrative_reads_current_render_block_paragraph(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", "1")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_CACHE_ENABLED", "false")
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: True)
    seen_payloads = []

    def fake_call(**kwargs):
        seen_payloads.append(kwargs["user_payload"])
        return _valid_response()

    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json", fake_call)
    render_text = "Render block paragraph is the public source text [1]."
    chapters = [_chapter(_section("s1", paragraph=render_text), _section("s2"))]

    run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert seen_payloads[0]["sections"][0]["composer_paragraph"] == render_text


def test_chapter_narrative_rolls_back_entire_chapter_on_missing_citation(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", "1")
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr(
        "rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "sections": [
                    {
                        "section_id": "s1",
                        "paragraph": "Salesforce deployments indicate workflow demand.",
                        "used_fact_refs": ["EV-1"],
                        "citation_refs": [],
                    }
                ]
            }
        },
    )
    chapters = [_chapter(_section("s1"))]
    original = copy.deepcopy(chapters)

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert rewritten == original
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["rejected_reasons"]["missing_required_citations"] == 1


def test_chapter_narrative_keeps_valid_sections_when_one_section_is_invalid(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", "1")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_CACHE_ENABLED", "false")
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: True)

    def fake_call(**kwargs):
        return {
            "payload": {
                "sections": [
                    {
                        "section_id": "s1",
                        "paragraph": "Salesforce 披露 Agentforce 已进入客服工作流部署，这说明企业智能体开始嵌入可运行流程 [1]。",
                        "used_fact_refs": ["EV-1"],
                        "citation_refs": ["[1]"],
                    },
                    {
                        "section_id": "s2",
                        "paragraph": "Salesforce 披露 Agentforce 已进入客服工作流部署，且新增了 999 个未给出的客户样本 [1]。",
                        "used_fact_refs": ["EV-1"],
                        "citation_refs": ["[1]"],
                    },
                ]
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json", fake_call)
    chapters = [_chapter(_section("s1"), _section("s2"))]

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert diagnostics["success_count"] == 1
    assert diagnostics["rejected_reasons"]["new_numeric_claim"] == 1
    assert "可运行流程" in rewritten[0]["sections"][0]["render_blocks"][0]["text"]
    assert rewritten[0]["sections"][1]["render_blocks"][0]["text"] == chapters[0]["sections"][1]["render_blocks"][0]["text"]


def test_chapter_narrative_tries_fallback_config_after_primary_error(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", "1")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_CACHE_ENABLED", "false")
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: bool(config.get("model")))
    models = []

    def fake_call(**kwargs):
        model = kwargs["config"]["model"]
        models.append(model)
        if model == "bad-primary":
            raise RuntimeError("primary key invalid")
        return _valid_response()

    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json", fake_call)

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=[_chapter(_section("s1"), _section("s2"))],
        report_blueprint={},
        llm_config={
            "url": "https://llm.test",
            "api_key": "bad",
            "model": "bad-primary",
            "fallback_config": {"url": "https://llm.test", "api_key": "ok", "model": "fallback-model"},
        },
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert diagnostics["success_count"] == 1
    assert diagnostics["fallback_model_used_count"] == 1
    assert models == ["bad-primary", "fallback-model"]
    assert rewritten[0]["chapter_narrative_status"] == "rewritten"


def test_chapter_narrative_skips_non_quality_paths(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    chapters = [_chapter(_section("s1"))]

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "deterministic_rebuild"},
    )

    assert rewritten == chapters
    assert diagnostics["enabled"] is True
    assert diagnostics["skipped_reason"] == "final_analysis_source_not_llm"


def test_chapter_narrative_skips_when_cited_sections_below_minimum(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_CHAPTER_NARRATIVE_MIN_EVIDENCE_SECTIONS", "6")

    def fail_call(**kwargs):
        raise AssertionError("chapter narrative LLM should not be called when cited section count is too low")

    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.llm_config_is_ready", lambda config: True)
    monkeypatch.setattr("rag_pipeline.agents.chapter_narrative_agent.call_openai_compatible_json", fail_call)

    chapters = [_chapter(_section("s1"), _section("s2"))]

    rewritten, diagnostics = run_chapter_narrative(
        chapter_packages=chapters,
        report_blueprint={},
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock"},
        quality_context={"final_analysis_source": "llm_partial_merged"},
    )

    assert rewritten == chapters
    assert diagnostics["enabled"] is True
    assert diagnostics["skipped_reason"] == "insufficient_cited_sections"
