from rag_pipeline.agents.section_body_rewrite_agent import (
    body_rewrite_enabled,
    body_rewrite_max_sections,
    rewrite_section_body,
    rewrite_sections_for_report,
)


def _section():
    return {
        "section_id": "s1",
        "chapter_id": "ch_01",
        "section_title": "Workflow deployment",
        "block_type": "case_comparison",
        "claim_strength": "moderate",
        "body_composition_status": "composed",
        "paragraph": "Salesforce disclosed customer-service workflow deployment [1]. This supports workflow demand.",
        "claim": "Salesforce disclosed customer-service workflow deployment [1].",
        "reasoning": "Salesforce disclosed customer-service workflow deployment [1]. This supports workflow demand.",
        "used_fact_refs": ["EV-1"],
        "evidence_refs": ["EV-1"],
        "citation_refs": ["[1]"],
        "supporting_facts": ["Salesforce disclosed customer-service workflow deployment."],
    }


def _facts():
    return [
        {
            "evidence_id": "EV-1",
            "distilled_fact": "Salesforce disclosed customer-service workflow deployment.",
            "source_ref": "[1]",
            "source_level": "A",
            "fact_type": "case",
        }
    ]


def test_body_rewrite_max_sections_has_quality_mode_floor(monkeypatch):
    monkeypatch.setenv("REPORT_BODY_REWRITE_MAX_SECTIONS", "12")
    monkeypatch.setenv("REPORT_REPLAY_EXECUTION_MODE", "quality_llm_replay")

    assert body_rewrite_max_sections() == 24


def test_rewrite_section_body_accepts_valid_llm_output(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "paragraph": "Salesforce disclosed customer-service workflow deployment, which shows enterprise agents are moving from trial use into an operating workflow [1].",
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            },
            "llm_call": {"model": "mock-model"},
        },
    )

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rewritten"
    assert "operating workflow" in result["paragraph"]
    assert result["input_ref_count"] == 1
    assert result["output_ref_count"] == 1


def test_rewrite_section_body_rejects_missing_refs(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "paragraph": "Salesforce deployments show operating workflow demand.",
                "used_fact_refs": [],
                "citation_refs": [],
            }
        },
    )

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rejected"
    assert result["paragraph"] == _section()["paragraph"]
    assert result["failure_reason"] == "missing_required_refs"


def test_rewrite_section_body_rejects_new_numbers(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "paragraph": "Salesforce deployments show workflow demand grew by 30% [1].",
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            }
        },
    )

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rejected"
    assert result["failure_reason"] == "new_numeric_claim"


def test_rewrite_section_body_rejects_internal_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "paragraph": "QA failed because evidence_cards still need repair [1]. This section should be fixed before publication.",
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            }
        },
    )

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rejected"
    assert result["failure_reason"] == "forbidden_public_text"


def test_rewrite_section_body_allows_high_quality_expansion_ratio(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("REPORT_BODY_REWRITE_MAX_EXPANSION_RATIO", "3.0")
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    original = _section()
    expanded = (
        "Salesforce disclosed customer-service workflow deployment [1]. "
        "This indicates enterprise agents are no longer only a trial interface, because a customer-service workflow "
        "requires integration with operating roles, permission boundaries, and repeatable service processes."
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "paragraph": expanded,
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            }
        },
    )

    result = rewrite_section_body(
        section=original,
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rewritten"
    assert len(result["paragraph"]) > len(original["paragraph"]) * 2


def test_rewrite_section_body_prefers_composer_paragraph_for_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    captured = {}

    def fake_call(**kwargs):
        captured["composer_paragraph"] = kwargs["user_payload"]["composer_paragraph"]
        return {
            "payload": {
                "paragraph": kwargs["user_payload"]["composer_paragraph"],
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            },
            "model": "mock-model",
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)
    short = _section()
    long_composed = (
        "Salesforce disclosed customer-service workflow deployment [1]. "
        "This composed paragraph explains how a workflow deployment signal affects enterprise demand, "
        "because operational adoption is stronger than a generic tool trial and requires process ownership."
    )
    short["paragraph"] = "Short claim [1]."
    short["claim"] = "Short claim [1]."
    short["composed_paragraph"] = long_composed
    short["render_blocks"] = [{"type": "paragraph", "text": long_composed}]

    result = rewrite_section_body(
        section=short,
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rewritten"
    assert captured["composer_paragraph"] == long_composed


def test_rewrite_section_body_sends_target_chars_to_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("REPORT_BODY_REWRITE_TARGET_SECTION_CHARS", "680")
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    captured = {}

    def fake_call(**kwargs):
        captured["target_chars"] = kwargs["user_payload"]["target_chars"]
        return {
            "payload": {
                "paragraph": kwargs["user_payload"]["composer_paragraph"],
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            },
            "model": "mock-model",
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "rewritten"
    assert captured["target_chars"] == 680


def test_rewrite_section_body_uses_cache_without_llm_call(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    calls = {"count": 0}

    def fake_call(**kwargs):
        calls["count"] += 1
        return {
            "payload": {
                "paragraph": "Salesforce disclosed customer-service workflow deployment, which shows enterprise agents are moving from trial use into an operating workflow [1].",
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)

    first = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )
    second = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert first["status"] == "rewritten"
    assert second["status"] == "cached"
    assert calls["count"] == 1


def test_rewrite_section_body_fail_open_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: False,
    )

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={},
    )

    assert result["status"] == "fallback"
    assert result["failure_reason"] == "llm_config_not_ready"
    assert result["paragraph"] == _section()["paragraph"]


def test_rewrite_section_body_marks_llm_error_as_called(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )

    def raise_error(**kwargs):
        raise RuntimeError("temporary outage")

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", raise_error)

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
    )

    assert result["status"] == "fallback"
    assert result["llm_called"] is True
    assert result["failure_reason"] == "llm_error:RuntimeError"


def test_rewrite_section_body_tries_fallback_config_after_primary_error(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: bool(config.get("model")),
    )
    models = []

    def fake_call(**kwargs):
        model = kwargs["config"]["model"]
        models.append(model)
        if model == "bad-primary":
            raise RuntimeError("primary key invalid")
        return {
            "payload": {
                "paragraph": "Salesforce disclosed customer-service workflow deployment, which shows enterprise agents are moving from trial use into operating workflow [1].",
                "used_fact_refs": ["EV-1"],
                "citation_refs": ["[1]"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)

    result = rewrite_section_body(
        section=_section(),
        facts=_facts(),
        chapter_question="Can AI agents convert into workflow deployment?",
        llm_config={
            "url": "https://llm.test",
            "api_key": "bad",
            "model": "bad-primary",
            "fallback_config": {"url": "https://llm.test", "api_key": "ok", "model": "fallback-model"},
        },
    )

    assert result["status"] == "rewritten"
    assert result["fallback_used"] is True
    assert models == ["bad-primary", "fallback-model"]


def test_body_rewrite_enabled_uses_env_flag(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    assert body_rewrite_enabled()
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")
    assert not body_rewrite_enabled()


def _chapter_package(*sections):
    return {
        "chapter_id": "ch_01",
        "chapter_question": "Can AI agents convert into workflow deployment?",
        "sections": [dict(section) for section in sections],
    }


def _section_with_ref(ref: str, citation: str, *, section_id: str | None = None):
    section = _section()
    section["section_id"] = section_id or f"s-{ref}"
    section["used_fact_refs"] = [ref]
    section["evidence_refs"] = [ref]
    section["citation_refs"] = [citation]
    section["supporting_facts"] = [f"Salesforce disclosed deployment for {ref}."]
    section["paragraph"] = f"Salesforce disclosed deployment for {ref} {citation}. This supports workflow demand."
    section["claim"] = section["paragraph"]
    section["reasoning"] = section["paragraph"]
    return section


def test_rewrite_sections_for_report_preserves_order_with_parallel_results(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("REPORT_BODY_REWRITE_CONCURRENCY", "3")
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )

    def fake_call(**kwargs):
        refs = kwargs["user_payload"]["used_fact_refs"]
        citation = kwargs["user_payload"]["citation_refs"][0]
        return {
            "payload": {
                "paragraph": f"Rewritten paragraph keeps the workflow demand signal {citation}.",
                "used_fact_refs": refs,
                "citation_refs": kwargs["user_payload"]["citation_refs"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)
    chapters = [
        _chapter_package(
            _section_with_ref("EV-A", "[1]", section_id="s-a"),
            _section_with_ref("EV-B", "[2]", section_id="s-b"),
            _section_with_ref("EV-C", "[3]", section_id="s-c"),
        )
    ]

    rewritten, diagnostics = rewrite_sections_for_report(
        chapter_packages=chapters,
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
        max_llm_calls=3,
        concurrency=3,
    )

    section_ids = [section["section_id"] for section in rewritten[0]["sections"]]
    assert section_ids == ["s-a", "s-b", "s-c"]
    assert [section["body_rewrite_status"] for section in rewritten[0]["sections"]] == ["rewritten", "rewritten", "rewritten"]
    assert diagnostics["called_count"] == 3
    assert diagnostics["submitted_count"] == 3
    assert diagnostics["concurrency"] == 3


def test_rewrite_sections_for_report_respects_concurrency_limit(monkeypatch, tmp_path):
    import threading
    import time

    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("REPORT_BODY_REWRITE_CONCURRENCY", "2")
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    lock = threading.Lock()
    active = {"current": 0, "max": 0}

    def fake_call(**kwargs):
        with lock:
            active["current"] += 1
            active["max"] = max(active["max"], active["current"])
        time.sleep(0.03)
        with lock:
            active["current"] -= 1
        return {
            "payload": {
                "paragraph": f"Rewritten paragraph keeps the workflow demand signal {kwargs['user_payload']['citation_refs'][0]}.",
                "used_fact_refs": kwargs["user_payload"]["used_fact_refs"],
                "citation_refs": kwargs["user_payload"]["citation_refs"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)
    chapters = [
        _chapter_package(
            _section_with_ref("EV-A", "[1]", section_id="s-a"),
            _section_with_ref("EV-B", "[2]", section_id="s-b"),
            _section_with_ref("EV-C", "[3]", section_id="s-c"),
            _section_with_ref("EV-D", "[4]", section_id="s-d"),
        )
    ]

    _, diagnostics = rewrite_sections_for_report(
        chapter_packages=chapters,
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
        max_llm_calls=4,
        concurrency=2,
    )

    assert active["max"] <= 2
    assert diagnostics["called_count"] == 4


def test_rewrite_sections_for_report_cache_hit_does_not_consume_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    calls = {"count": 0}

    def fake_call(**kwargs):
        calls["count"] += 1
        return {
            "payload": {
                "paragraph": f"Rewritten paragraph keeps the workflow demand signal {kwargs['user_payload']['citation_refs'][0]}.",
                "used_fact_refs": kwargs["user_payload"]["used_fact_refs"],
                "citation_refs": kwargs["user_payload"]["citation_refs"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)
    chapters = [_chapter_package(_section_with_ref("EV-A", "[1]", section_id="s-a"))]
    config = {"url": "https://llm.test", "api_key": "key", "model": "mock-model"}
    first, first_diag = rewrite_sections_for_report(chapter_packages=chapters, llm_config=config, max_llm_calls=1)
    second, second_diag = rewrite_sections_for_report(chapter_packages=chapters, llm_config=config, max_llm_calls=0)

    assert first[0]["sections"][0]["body_rewrite_status"] == "rewritten"
    assert first_diag["called_count"] == 1
    assert second[0]["sections"][0]["body_rewrite_status"] == "cached"
    assert second_diag["called_count"] == 0
    assert second_diag["cache_hit_count"] == 1
    assert calls["count"] == 1


def test_rewrite_sections_for_report_fail_open_per_section(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("REPORT_BODY_REWRITE_CONCURRENCY", "2")
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )

    def fake_call(**kwargs):
        refs = kwargs["user_payload"]["used_fact_refs"]
        if refs == ["EV-BAD"]:
            raise RuntimeError("temporary outage")
        return {
            "payload": {
                "paragraph": f"Rewritten paragraph keeps the workflow demand signal {kwargs['user_payload']['citation_refs'][0]}.",
                "used_fact_refs": refs,
                "citation_refs": kwargs["user_payload"]["citation_refs"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)
    bad = _section_with_ref("EV-BAD", "[2]", section_id="s-bad")
    chapters = [_chapter_package(_section_with_ref("EV-A", "[1]", section_id="s-a"), bad)]

    rewritten, diagnostics = rewrite_sections_for_report(
        chapter_packages=chapters,
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
        max_llm_calls=2,
        concurrency=2,
    )

    statuses = [section["body_rewrite_status"] for section in rewritten[0]["sections"]]
    assert statuses == ["rewritten", "fallback"]
    assert rewritten[0]["sections"][1]["claim"] == bad["claim"]
    assert diagnostics["called_count"] == 2
    assert diagnostics["fallback_count"] == 1


def test_rewrite_sections_for_report_inflight_deduplicates_same_cache_key(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("REPORT_BODY_REWRITE_CONCURRENCY", "2")
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    calls = {"count": 0}

    def fake_call(**kwargs):
        calls["count"] += 1
        return {
            "payload": {
                "paragraph": f"Rewritten paragraph keeps the workflow demand signal {kwargs['user_payload']['citation_refs'][0]}.",
                "used_fact_refs": kwargs["user_payload"]["used_fact_refs"],
                "citation_refs": kwargs["user_payload"]["citation_refs"],
            }
        }

    monkeypatch.setattr("rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json", fake_call)
    first = _section_with_ref("EV-A", "[1]", section_id="same-section")
    second = _section_with_ref("EV-A", "[1]", section_id="same-section")
    chapters = [_chapter_package(first, second)]

    rewritten, diagnostics = rewrite_sections_for_report(
        chapter_packages=chapters,
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
        max_llm_calls=2,
        concurrency=2,
    )

    assert [section["body_rewrite_status"] for section in rewritten[0]["sections"]] == ["rewritten", "rewritten"]
    assert calls["count"] == 1
    assert diagnostics["called_count"] == 1
    assert diagnostics["inflight_dedup_count"] == 1


def test_rewrite_sections_for_report_marks_budget_exhausted(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_ENABLED", "false")
    monkeypatch.setenv("REPORT_BODY_REWRITE_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.llm_config_is_ready",
        lambda config: True,
    )
    monkeypatch.setattr(
        "rag_pipeline.agents.section_body_rewrite_agent.call_openai_compatible_json",
        lambda **kwargs: {
            "payload": {
                "paragraph": f"Rewritten paragraph keeps the workflow demand signal {kwargs['user_payload']['citation_refs'][0]}.",
                "used_fact_refs": kwargs["user_payload"]["used_fact_refs"],
                "citation_refs": kwargs["user_payload"]["citation_refs"],
            }
        },
    )
    chapters = [
        _chapter_package(
            _section_with_ref("EV-A", "[1]", section_id="s-a"),
            _section_with_ref("EV-B", "[2]", section_id="s-b"),
        )
    ]

    rewritten, diagnostics = rewrite_sections_for_report(
        chapter_packages=chapters,
        llm_config={"url": "https://llm.test", "api_key": "key", "model": "mock-model"},
        max_llm_calls=1,
        concurrency=2,
    )

    assert [section["body_rewrite_status"] for section in rewritten[0]["sections"]] == ["rewritten", "skipped"]
    assert rewritten[0]["sections"][1]["body_rewrite"]["failure_reason"] == "budget_exhausted"
    assert diagnostics["budget_exhausted_count"] == 1
