from rag_pipeline.agents.report_contracts import (
    ChapterInsight,
    ClaimUnit,
    EvidenceFactCard,
    ReportSection,
    filter_resolvable_evidence_refs,
    resolve_evidence_source_ref,
)
from rag_pipeline.agents.chapter_argument_agent import run_chapter_argument_agent
from rag_pipeline.agents.section_composer import compose_section_paragraph
from rag_pipeline.agents.writer_agent_clean import _chapter_expandable, _expand_chapter_packages_for_body_target
from rag_pipeline.contracts.report_contract import build_report_contract_from_package


def test_evidence_fact_card_drops_internal_enum_tokens_from_variable():
    # Internal routing labels (analysis_role / block_type) must never become a
    # public "variable" subject, or they leak into prose like
    # "用于判断counter是否具备持续性".
    for token in ("counter", "claimable", "integrated_signal", "risk_trigger"):
        card = EvidenceFactCard.from_legacy_dict(
            {"evidence_id": "EV-x", "variable": token, "distilled_fact": "某事实陈述。"}
        )
        assert card.variable == "", f"internal token {token!r} leaked into variable"
    kept = EvidenceFactCard.from_legacy_dict(
        {"evidence_id": "EV-y", "variable": "渗透率", "distilled_fact": "某事实陈述。"}
    )
    assert kept.variable == "渗透率"


def test_evidence_fact_card_normalizes_legacy_shapes_and_preserves_zero_source_ref():
    item = {
        "evidence_id": "EV-1",
        "dimension_id": "ch_01",
        "source_id": 0,
        "source_level": "B",
        "source_verification_status": "readpage_verified",
        "public_fact_quality": {
            "eligible_for_report": True,
            "public_fact_card": {
                "subject": "Salesforce Agentforce",
                "action_or_signal": "announced customer-service workflow deployments",
                "variable": "customer deployment",
                "distilled_fact": "Salesforce Agentforce disclosed customer-service workflow deployments in 2025.",
                "fact_type": "case",
                "block_affinity": ["case_comparison"],
            },
        },
    }

    card = EvidenceFactCard.from_legacy_dict(item)

    assert card.evidence_id == "EV-1"
    assert card.chapter_id == "ch_01"
    assert card.source_ref == "0"
    assert card.block_affinity == ["case_comparison"]
    assert card.is_valid_for_report
    assert card.to_legacy_dict()["public_fact_card"]["distilled_fact"].startswith("Salesforce")


def test_claim_unit_collects_legacy_reference_aliases():
    unit = ClaimUnit.from_legacy_dict(
        {
            "chapter_id": "ch_01",
            "judgment": "Enterprise agents are moving from trials to workflow deployment.",
            "used_evidence_ids": ["EV-1"],
            "supporting_evidence_refs": ["EV-2"],
            "supporting_evidence": [{"evidence_id": "EV-3"}],
            "claim_strength": "directional",
        }
    )

    assert unit.claim.startswith("Enterprise agents")
    assert unit.evidence_refs == ["EV-1", "EV-2", "EV-3"]
    assert unit.to_legacy_dict()["used_fact_refs"] == ["EV-1", "EV-2", "EV-3"]


def test_fact_card_preserves_requirement_and_lineage_contract_fields():
    card = EvidenceFactCard.from_legacy_dict(
        {
            "evidence_id": "EV-REQ",
            "chapter_id": "ch_01",
            "hypothesis_id": "H1",
            "requirement_id": "H1_case",
            "search_task_id": "task_case_1",
            "source_id": "SRC-1",
            "analysis_role": "case",
            "analysis_eligible": True,
            "allowed_use": "directional_signal",
            "source_ref": "[3]",
            "public_fact_card": {
                "subject": "Salesforce Agentforce",
                "distilled_fact": "Salesforce disclosed workflow deployment cases.",
                "fact_type": "case",
                "block_affinity": ["case_comparison"],
            },
        }
    )

    legacy = card.to_legacy_dict()

    assert card.requirement_id == "H1_case"
    assert card.hypothesis_id == "H1"
    assert card.analysis_role == "case"
    assert card.analysis_eligible is True
    assert card.allowed_use == "directional_signal"
    assert card.lineage == {
        "chapter_id": "ch_01",
        "hypothesis_id": "H1",
        "requirement_id": "H1_case",
        "fact_id": "EV-REQ",
        "source_id": "SRC-1",
        "search_task_id": "task_case_1",
    }
    assert legacy["requirement_id"] == "H1_case"
    assert legacy["lineage"]["requirement_id"] == "H1_case"


def test_claim_unit_preserves_requirement_ids_strength_ceiling_and_lineage():
    unit = ClaimUnit.from_legacy_dict(
        {
            "claim_id": "CL-H1",
            "chapter_id": "ch_01",
            "hypothesis_id": "H1",
            "requirement_ids": ["H1_metric", "H1_case"],
            "claim": "Agent demand is moving toward workflow deployment.",
            "claim_strength": "moderate",
            "claim_strength_ceiling": "moderate",
            "evidence_use_level": "directional_signal",
            "writing_permission": "cautious_with_boundary",
            "metric_completeness_status": "incomplete",
            "metric_missing_fields": ["unit", "period"],
            "used_evidence_ids": ["EV-1"],
            "source_support_map": {"claim": ["EV-1"]},
            "lineage": {
                "requirement_ids": ["H1_metric", "H1_case"],
                "fact_ids": ["EV-1"],
                "source_ids": ["SRC-1"],
                "search_task_ids": ["task_metric_1"],
            },
        }
    )

    legacy = unit.to_legacy_dict()

    assert unit.hypothesis_id == "H1"
    assert unit.requirement_ids == ["H1_metric", "H1_case"]
    assert unit.claim_strength_ceiling == "moderate"
    assert unit.evidence_use_level == "directional_signal"
    assert unit.writing_permission == "cautious_with_boundary"
    assert unit.metric_completeness_status == "incomplete"
    assert unit.metric_missing_fields == ["unit", "period"]
    assert unit.lineage["source_ids"] == ["SRC-1"]
    assert legacy["requirement_ids"] == ["H1_metric", "H1_case"]
    assert legacy["claim_strength_ceiling"] == "moderate"
    assert legacy["evidence_use_level"] == "directional_signal"
    assert legacy["writing_permission"] == "cautious_with_boundary"
    assert legacy["metric_completeness_status"] == "incomplete"
    assert legacy["metric_missing_fields"] == ["unit", "period"]
    assert legacy["lineage"]["fact_ids"] == ["EV-1"]


def test_report_contract_emits_requirement_slots_with_stable_ids():
    package = {
        "query": "AI Agent workflow adoption",
        "evidence_package": {
            "metadata": {
                "research_plan": {
                    "query": "AI Agent workflow adoption",
                    "chapters": [
                        {
                            "chapter_id": "ch_01",
                            "chapter_title": "Workflow demand",
                            "core_question": "Is workflow demand real?",
                            "required_evidence_roles": ["metric", "case", "counter"],
                            "min_ab_sources": 1,
                        }
                    ],
                }
            }
        },
    }

    contract = build_report_contract_from_package(package)
    requirements = contract["evidence_requirements"]["requirements"]

    assert [item["requirement_id"] for item in requirements] == [
        "ch_01_metric",
        "ch_01_case",
        "ch_01_counter",
        "ch_01_source_check",
    ]
    assert requirements[0]["chapter_id"] == "ch_01"
    assert requirements[0]["proof_role"] == "metric"
    assert "value" in requirements[0]["required_fields"]
    assert requirements[1]["claim_strength_ceiling"] == "directional"


def test_report_contract_evaluates_requirement_status_from_fact_cards():
    package = {
        "query": "AI Agent workflow adoption",
        "evidence_package": {
            "metadata": {
                "research_plan": {
                    "query": "AI Agent workflow adoption",
                    "chapters": [
                        {
                            "chapter_id": "ch_01",
                            "chapter_title": "Workflow demand",
                            "core_question": "Is workflow demand real?",
                            "required_evidence_roles": ["metric", "case"],
                        }
                    ],
                }
            },
            "analysis_ready_evidence": [
                {
                    "evidence_id": "EV-CASE",
                    "chapter_id": "ch_01",
                    "requirement_id": "ch_01_case",
                    "source_level": "C",
                    "analysis_eligible": True,
                    "analysis_role": "case",
                    "source_ref": "[2]",
                    "fact": "A customer case disclosed workflow deployment.",
                }
            ],
        },
    }

    contract = build_report_contract_from_package(package)
    status_by_id = {
        item["requirement_id"]: item
        for item in contract["evidence_requirements"]["requirement_status"]
    }

    assert status_by_id["ch_01_case"]["status"] == "directional_ready"
    assert status_by_id["ch_01_case"]["matched_fact_refs"] == ["EV-CASE"]
    assert status_by_id["ch_01_metric"]["status"] == "needs_repair"
    assert "metric" in status_by_id["ch_01_metric"]["missing"]


def test_chapter_insight_and_report_section_round_trip_legacy_dicts():
    insight = ChapterInsight.from_legacy_dict(
        {
            "chapter_id": "ch_01",
            "chapter_question": "Can demand convert to paid workflow deployment?",
            "key_claims": [{"claim": "Demand is visible in support workflows.", "evidence_refs": ["EV-1"]}],
            "fact_chain": ["Salesforce disclosed deployments."],
        }
    )
    section = ReportSection.from_legacy_dict(
        {
            "section_id": "s1",
            "chapter_id": "ch_01",
            "block_type": "case_comparison",
            "section_title": "Customer-service deployment",
            "claim": "Demand is visible in support workflows.",
            "used_fact_refs": ["EV-1"],
            "supporting_facts": ["Salesforce disclosed deployments."],
            "evidence_backed": True,
        }
    )

    assert insight.claim_units[0].evidence_refs == ["EV-1"]
    assert section.evidence_refs == ["EV-1"]
    assert section.to_legacy_dict()["composition_status"] == "legacy"


def test_composer_turns_metric_fact_card_into_natural_paragraph():
    card = EvidenceFactCard.from_legacy_dict(
        {
            "evidence_id": "EV-M",
            "chapter_id": "ch_01",
            "public_fact_card": {
                "subject": "China humanoid robot market",
                "variable": "market size",
                "action_or_signal": "reached",
                "value": "8.2 billion yuan",
                "unit": "yuan",
                "time_or_scope": "2025 China",
                "distilled_fact": "China humanoid robot market reached 8.2 billion yuan in 2025.",
                "fact_type": "metric",
                "block_affinity": ["metric_reconciliation"],
            },
            "source_level": "A",
            "source_ref": "[1]",
            "source_verification_status": "document_verified",
        }
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=ClaimUnit(chapter_id="ch_01", claim_strength="moderate"),
        block_type="metric_reconciliation",
        chapter_question="Is market space real?",
    )

    assert result["composition_status"] == "composed"
    assert "market size: 8.2 billion yuan" not in result["paragraph"]
    assert "8.2 billion yuan" in result["paragraph"]
    assert result["used_fact_refs"] == ["EV-M"]
    assert result["variable_explanation"]
    assert result["composer_variable_explanation_count"] == 1


def test_internal_metric_unit_enum_does_not_leak_into_fact_card_or_paragraph():
    card = EvidenceFactCard.from_legacy_dict(
        {
            "evidence_id": "EV-USD",
            "chapter_id": "ch_01",
            "public_fact_card": {
                "subject": "AI Agent market",
                "variable": "market size",
                "value": "471亿美元",
                "unit": "currency_usd",
                "time_or_scope": "2025",
                "distilled_fact": "AI Agent market reached 471亿美元 in 2025.",
                "fact_type": "metric",
                "block_affinity": ["metric_reconciliation"],
            },
            "source_level": "B",
            "source_ref": "[2]",
            "source_verification_status": "document_verified",
        }
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=ClaimUnit(chapter_id="ch_01", claim_strength="moderate"),
        block_type="metric_reconciliation",
        chapter_question="Is market space real?",
    )

    assert card.unit == ""
    assert "currency_usd" not in str(result)


def test_ref_lineage_resolves_source_alias_and_filters_orphan_refs():
    source_registry = [
        {
            "ref": "[12]",
            "evidence_id": "EV-OK",
            "source_ref": "SRC-OK",
            "url": "https://example.org/source",
            "title": "Traceable source",
        }
    ]

    resolved = resolve_evidence_source_ref("EV-OK", source_registry)
    orphan = resolve_evidence_source_ref("EV-06-26", source_registry)
    filtered = filter_resolvable_evidence_refs(["EV-OK", "EV-06-26", "[202]"], source_registry)

    assert resolved["resolved"] is True
    assert resolved["source_ref"] == "[12]"
    assert orphan["resolved"] is False
    assert orphan["reason"] == "unresolved_ref"
    assert filtered["resolved_refs"] == ["[12]"]
    assert [item["ref"] for item in filtered["filtered_refs"]] == ["EV-06-26", "[202]"]


def test_composer_drops_search_snippet_like_fact_cards():
    card = EvidenceFactCard(
        evidence_id="EV-BAD",
        chapter_id="ch_01",
        subject="news page",
        distilled_fact="AI Agent commercialization-Alibaba Cloud Developer Community: this article introduces...",
        fact_type="case",
        block_affinity=["case_comparison"],
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=ClaimUnit(chapter_id="ch_01", claim_strength="directional"),
        block_type="case_comparison",
        chapter_question="Where are deployments happening?",
    )

    assert result["composition_status"] == "dropped"
    assert result["paragraph"] == ""
    assert result["omit_reason"] == "no_valid_fact_card"


def test_composer_drops_date_metadata_fact_cards():
    card = EvidenceFactCard(
        evidence_id="EV-DATE",
        chapter_id="ch_01",
        subject="Statistics page",
        variable="risk boundary",
        distilled_fact="Publication date: 02 Apr 2025",
        fact_type="risk",
        block_affinity=["risk_trigger"],
        source_ref="[3]",
        source_level="B",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=ClaimUnit(chapter_id="ch_01", claim_strength="directional"),
        block_type="risk_trigger",
        chapter_question="What can weaken the conclusion?",
    )

    assert result["composition_status"] == "dropped"
    assert result["paragraph"] == ""
    assert result["omit_reason"] == "no_valid_fact_card"


def test_composer_does_not_emit_generic_weak_boundary_sentences():
    card = EvidenceFactCard(
        evidence_id="EV-DIR",
        chapter_id="ch_01",
        subject="Salesforce Agentforce",
        variable="customer deployment",
        distilled_fact="Salesforce disclosed customer-service workflow deployments.",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[2]",
        source_level="C",
        claim_strength_hint="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=ClaimUnit(chapter_id="ch_01", claim_strength="directional"),
        block_type="case_comparison",
        chapter_question="Can workflow deployment convert into durable demand?",
    )

    combined = "".join(str(result.get(key) or "") for key in ("paragraph", "claim", "reasoning", "mechanism", "counter_evidence"))
    assert "不能外推为全行业确定结论" not in combined
    assert "继续观察同一变量" not in combined
    assert result["counter_evidence"]
    assert "customer deployment" in combined or "workflow" in combined


def test_composer_uses_claim_unit_analysis_fields_for_richer_paragraph():
    card = EvidenceFactCard(
        evidence_id="EV-CLAIM",
        chapter_id="ch_01",
        subject="Enterprise AI agent platform",
        variable="workflow deployment",
        distilled_fact="Enterprise AI agent platform deployments moved from pilot use into customer-service workflows.",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[2]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-1",
        chapter_id="ch_01",
        claim="Enterprise AI agents are moving from tool trials into workflow deployment.",
        evidence_refs=["EV-CLAIM"],
        evidence_basis=["Customer-service workflow deployment is visible in the cited fact card."],
        reasoning_chain="Workflow deployment matters because it requires integration, permission control, and process ownership.",
        limitation_boundary="The conclusion remains bounded by whether deployment depth converts into repeatable paid usage.",
        paragraph_seed="The evidence points to a shift from experimentation toward process-level adoption.",
        claim_strength="moderate",
        evidence_use_level="directional_signal",
        writing_permission="cautious_with_boundary",
        metric_completeness_status="incomplete",
        metric_missing_fields=["unit", "period"],
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="case_comparison",
        chapter_question="Can AI agents convert pilots into durable workflow demand?",
    )

    paragraph = result["paragraph"]
    assert result["composition_status"] == "composed"
    assert "Enterprise AI agents are moving from tool trials into workflow deployment." in paragraph
    assert "Customer-service workflow deployment is visible" in paragraph
    assert "requires integration, permission control, and process ownership" in paragraph
    assert "repeatable paid usage" in paragraph
    assert result["evidence_use_level"] == "directional_signal"
    assert result["writing_permission"] == "cautious_with_boundary"
    assert result["metric_completeness_status"] == "incomplete"
    assert result["metric_missing_fields"] == ["unit", "period"]
    assert len(paragraph) > len(card.distilled_fact) * 2


def test_composer_expands_valid_claim_to_research_paragraph(monkeypatch):
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "420")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "true")
    card = EvidenceFactCard(
        evidence_id="EV-LONG",
        chapter_id="ch_01",
        subject="Enterprise AI agent platform",
        variable="workflow deployment",
        action_or_signal="entered production workflow",
        distilled_fact="Enterprise AI agent platform deployments moved from pilot use into customer-service and software-engineering workflows.",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[2]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-LONG",
        chapter_id="ch_01",
        claim="Enterprise AI agents are beginning to convert from tool trials into workflow deployment.",
        evidence_refs=["EV-LONG"],
        evidence_basis=[
            "The cited deployment fact shows customer-service and software-engineering workflows rather than a generic demo scenario.",
            "Those workflows require ownership, permission control, and measurable operating outcomes.",
        ],
        reasoning_chain="The important mechanism is that workflow deployment forces the product to connect with existing systems, permissions, and accountability, which is a stronger signal than standalone usage.",
        limitation_boundary="The judgment should remain bounded by whether those deployments expand from a few workflows into repeatable paid usage across more customers.",
        paragraph_seed="The report should explain why workflow deployment is a stronger demand signal than isolated experimentation.",
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="case_comparison",
        chapter_question="Can AI agents convert pilots into durable workflow demand?",
    )

    paragraph = result["paragraph"]
    assert result["composition_status"] in {"composed", "composed_directional"}
    assert result["composer_expansion_status"] == "expanded"
    assert len(paragraph.replace(" ", "")) >= 350
    assert "tool trials" in paragraph
    assert "workflow deployment" in paragraph
    assert "repeatable paid usage" in paragraph


def test_composer_longform_does_not_repeat_same_fact_sentence(monkeypatch):
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "850")
    monkeypatch.setenv("REPORT_COMPOSER_EXPAND_TO_TARGET", "true")
    fact = "AI Agent commercialization landed, and the 3.3 trillion yuan track is accelerating."
    card = EvidenceFactCard(
        evidence_id="EV-REPEAT",
        chapter_id="ch_01",
        subject="AI Agent commercialization",
        variable="deployment depth",
        action_or_signal="landed",
        distilled_fact=fact,
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[2]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-REPEAT",
        chapter_id="ch_01",
        claim="AI Agent commercialization is moving from concept discussion into observable deployment signals.",
        evidence_refs=["EV-REPEAT"],
        evidence_basis=[fact],
        reasoning_chain="Deployment depth matters because it determines whether a signal can move from attention into workflow adoption.",
        limitation_boundary="The judgment remains bounded by whether deployment depth appears across more customers and paid workflows.",
        paragraph_seed="The section should explain why deployment depth is a stronger signal than attention alone.",
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="case_comparison",
        chapter_question="Can commercialization move into durable workflow demand?",
    )

    assert result["composition_status"] in {"composed", "composed_directional"}
    assert result["composer_expansion_status"] == "expanded"
    assert result["paragraph"].count(fact.rstrip(".")) <= 1


def test_composer_default_does_not_generate_generic_template_filler(monkeypatch):
    monkeypatch.delenv("REPORT_COMPOSER_EXPAND_TO_TARGET", raising=False)
    monkeypatch.setenv("REPORT_COMPOSER_TARGET_SECTION_CHARS", "850")
    card = EvidenceFactCard(
        evidence_id="EV-GENERIC",
        chapter_id="ch_01",
        subject="AI Agent deployment",
        variable="章节信号",
        action_or_signal="entered workflow",
        distilled_fact="AI Agent deployment entered customer workflow in the cited source.",
        fact_type="case",
        block_affinity=["integrated_signal"],
        source_ref="[2]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-GENERIC",
        chapter_id="ch_01",
        claim="AI Agent deployment shows a directional workflow signal.",
        evidence_refs=["EV-GENERIC"],
        evidence_basis=["The cited source says AI Agent deployment entered customer workflow."],
        claim_strength="directional",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="integrated_signal",
        chapter_question="How should enterprise AI Agent deployment be judged?",
    )

    paragraph = result["paragraph"]
    assert result["composer_expansion_status"] == "base_no_expand"
    assert "事实转成判断" not in paragraph
    assert "核心连接点" not in paragraph
    assert "这一事实用于判断" not in paragraph


def test_composer_does_not_promote_raw_evidence_basis_snippets():
    card = EvidenceFactCard(
        evidence_id="EV-CLEAN",
        chapter_id="ch_01",
        subject="Enterprise AI agent platform",
        variable="workflow deployment",
        distilled_fact="Enterprise AI agent platform deployments moved into customer-service workflows.",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[2]",
        source_level="B",
    )
    claim = ClaimUnit(
        claim_id="CL-SNIPPET",
        chapter_id="ch_01",
        claim="Enterprise AI agents are moving from demos into workflow deployment.",
        evidence_refs=["EV-CLEAN"],
        evidence_basis=[
            "字体： 大 中 小 近日，市经济和信息化局统计，国内垂直领域研报服务的标杆之一。",
            "AI 时代，唯一确定的是数据｜爱分析访谈 - 电子工程专辑（2026-05-21T00:00:00+08:00）：数据是穿越周期的壁垒。",
        ],
        reasoning_chain="Workflow deployment is a stronger signal than a generic demo because it requires process ownership.",
        paragraph_seed="The evidence points to production workflow adoption.",
        claim_strength="moderate",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=claim,
        block_type="case_comparison",
        chapter_question="Can AI agents move into durable workflow demand?",
    )

    combined = " ".join(
        [str(result.get("paragraph") or "")]
        + [str(item) for item in result.get("supporting_facts") or []]
    )
    assert "字体" not in combined
    assert "近日，市经济和信息化局统计" not in combined
    assert "国内垂直领域研报服务" not in combined
    assert "电子工程专辑" not in combined


def test_high_quality_mode_does_not_add_deterministic_expansion_sections(monkeypatch):
    monkeypatch.setenv("REPORT_QUALITY_MODE", "high")
    monkeypatch.setenv("REPORT_ENABLE_BODY_EXPANSION", "true")
    chapter = {
        "chapter_id": "ch_01",
        "chapter_title": "Workflow adoption",
        "chapter_fact_digest": [
            "Enterprise AI agent platform deployments moved into customer-service workflows.",
            "Deployment materials emphasize permissions and enterprise workflow controls.",
        ],
        "sections": [
            {
                "section_id": "s1",
                "section_title": "Workflow deployment",
                "claim": "Enterprise AI agent deployments are moving into workflow usage.",
                "evidence_refs": ["EV-1", "EV-2"],
                "supporting_facts": [
                    "Enterprise AI agent platform deployments moved into customer-service workflows.",
                    "Deployment materials emphasize permissions and enterprise workflow controls.",
                ],
                "evidence_backed": True,
            }
        ],
    }

    expanded = _expand_chapter_packages_for_body_target([chapter], target_chars=20000)
    titles = [section.get("section_title") for section in expanded[0].get("sections") or []]

    assert len(titles) == 1
    assert not any(section.get("expansion_generated") for section in expanded[0].get("sections") or [])
    assert "机制传导、约束条件与边界" not in titles


def test_composer_consumes_multiple_claim_referenced_fact_cards():
    first = EvidenceFactCard(
        evidence_id="EV-CASE",
        chapter_id="ch_01",
        subject="Salesforce Agentforce",
        variable="customer workflow",
        distilled_fact="Salesforce disclosed Agentforce customer workflow deployments.",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[1]",
        source_level="B",
    )
    second = EvidenceFactCard(
        evidence_id="EV-TECH",
        chapter_id="ch_01",
        subject="Salesforce Agentforce",
        variable="permission and integration",
        distilled_fact="Agentforce deployment materials emphasize permissions, integration, and enterprise workflow controls.",
        fact_type="technology",
        block_affinity=["case_comparison", "technology_maturity"],
        source_ref="[2]",
        source_level="B",
    )

    result = compose_section_paragraph(
        fact_cards=[first, second],
        claim_unit=ClaimUnit(
            chapter_id="ch_01",
            claim="Agent workflow adoption depends on both customer deployment and enterprise integration controls.",
            evidence_refs=["EV-CASE", "EV-TECH"],
            evidence_basis=[first.distilled_fact, second.distilled_fact],
            reasoning_chain="The two facts connect demand validation with operational feasibility.",
            claim_strength="moderate",
        ),
        block_type="case_comparison",
        chapter_question="Can workflow deployments scale?",
    )

    assert result["used_fact_refs"] == ["EV-CASE", "EV-TECH"]
    assert first.distilled_fact in result["supporting_facts"]
    assert second.distilled_fact in result["supporting_facts"]
    assert "operational feasibility" in result["paragraph"]


def test_composer_removes_empty_connector_subject_from_public_sentence():
    card = EvidenceFactCard(
        evidence_id="EV-BAD-SUBJECT",
        chapter_id="ch_01",
        subject="为此",
        variable="customer deployment",
        distilled_fact="A research center released a large-model application report.",
        fact_type="case",
        block_affinity=["case_comparison"],
        source_ref="[3]",
        source_level="B",
    )

    result = compose_section_paragraph(
        fact_cards=[card],
        claim_unit=ClaimUnit(chapter_id="ch_01", claim_strength="moderate"),
        block_type="case_comparison",
        chapter_question="Where are deployments happening?",
    )

    assert "为此的动作显示" not in result["paragraph"]
    assert "为此的动作显示" not in result["claim"]


def test_chapter_argument_uses_composer_for_layout_fallback_metric_section():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Market validation"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "block_type": "metric_reconciliation",
                        "required_evidence_refs": ["EV-M"],
                        "dynamic_section_title": "Market scale validation",
                    }
                ],
            }
        ],
        argument_units=[],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "metric_evidence": [
                        {
                            "evidence_id": "EV-M",
                            "source_ref": "[1]",
                            "source_level": "B",
                            "public_fact_card": {
                            "subject": "China humanoid robot market",
                            "variable": "market size",
                            "action_or_signal": "reached",
                            "value": "8.2 billion yuan",
                            "time_or_scope": "2025 China",
                            "distilled_fact": "China humanoid robot market reached 8.2 billion yuan in 2025.",
                            "fact_type": "metric",
                            "block_affinity": ["metric_reconciliation"],
                        },
                        "public_fact_quality": {"eligible_for_report": True},
                    }
                ],
            }
        ],
    )

    section = packages[0]["sections"][0]
    assert section["composition_status"] == "composed"
    assert "market size: 8.2 billion yuan" not in section["claim"]
    assert section["used_fact_refs"] == ["EV-M"]
    assert section["body_composition_status"] == "composed"
    assert section["composed_paragraph"] == section["render_blocks"][0]["text"]
    assert section["composer_paragraph_chars"] == len(section["composed_paragraph"].replace(" ", ""))


def test_chapter_argument_does_not_render_boundary_as_standalone_public_paragraph(monkeypatch):
    monkeypatch.delenv("REPORT_RENDER_BOUNDARY_PARAGRAPH", raising=False)
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Workflow adoption"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-1"],
                        "dynamic_section_title": "Workflow deployment signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_case",
                "block_type": "case_comparison",
                "claim": "Enterprise AI agents are moving from pilots into workflow deployment.",
                "reasoning": "Workflow deployment requires integration, permissions, and process ownership.",
                "counter_evidence": "This case is limited to one customer and should not represent the whole market.",
                "used_fact_refs": ["EV-1"],
                "supporting_facts": ["Enterprise AI agents entered customer-service workflow deployment."],
                "claim_strength": "directional",
            }
        ],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "case_evidence": [
                    {
                        "evidence_id": "EV-1",
                        "source_ref": "[1]",
                        "source_level": "B",
                        "public_fact_card": {
                            "subject": "Enterprise AI agents",
                            "variable": "workflow deployment",
                            "action_or_signal": "entered",
                            "distilled_fact": "Enterprise AI agents entered customer-service workflow deployment.",
                            "fact_type": "case",
                            "block_affinity": ["case_comparison"],
                        },
                        "public_fact_quality": {"eligible_for_report": True},
                    }
                ],
            }
        ],
    )

    section = packages[0]["sections"][0]
    public_blocks = [block for block in section["render_blocks"] if block.get("type") == "paragraph"]
    assert len(public_blocks) == 1
    assert "should not represent the whole market" not in public_blocks[0]["text"]
    assert section["counter_evidence"]


def test_chapter_argument_does_not_render_context_only_claim_as_public_section():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Context only"}]},
        micro_layouts=[],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "claim_id": "CL-context",
                "claim": "The statistics office publishes AI Agent statistics through several public channels.",
                "reasoning": "This is background context, not support for a report claim.",
                "used_fact_refs": ["EV-context"],
                "supporting_facts": ["The statistics office publishes statistics through public channels."],
                "claim_roles": ["context_claim"],
                "primary_claim_role": "context_claim",
                "public_render": True,
            }
        ],
        chapter_evidence_packages=[],
    )

    package = packages[0]
    assert package["omit_from_report"] is True
    assert package["sections"] == []
    assert package["dropped_sections"][0]["reason"] == "context_claim_not_public"


def test_chapter_argument_clean_public_text_preserves_long_analysis_after_clipping():
    from rag_pipeline.agents.chapter_argument_agent import _clean_public_text

    sentence = (
        "企业级智能体已经从演示能力进入流程部署，关键在于客户是否把权限、集成、责任边界和持续运维放进同一个业务闭环。"
        "这类事实说明需求判断不能只看概念热度，而要看部署动作是否进入真实岗位、真实流程和真实预算。"
    )
    paragraph = sentence * 6

    cleaned = _clean_public_text(paragraph, max_chars=900)

    assert len(cleaned.replace(" ", "")) > 500
    assert "企业级智能体" in cleaned


def test_chapter_argument_clean_public_text_removes_bad_sentence_not_whole_paragraph():
    from rag_pipeline.agents.chapter_argument_agent import _clean_public_text

    paragraph = (
        "企业级智能体的部署信号已经进入客服和运营流程，这说明需求正在从试用工具转向可复用工作流。"
        "Official statistics show AI agent adoption URL: https://example.invalid/report. "
        "后续判断应关注客户是否继续扩大使用范围、是否形成付费链路，以及权限治理是否能支撑规模化部署。"
    )

    cleaned = _clean_public_text(paragraph, max_chars=900)

    assert "Official statistics" not in cleaned
    assert "https://example.invalid" not in cleaned
    assert "企业级智能体" in cleaned
    assert "付费链路" in cleaned


def test_chapter_argument_composes_llm_claim_from_evidence_basis_when_cards_missing():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Workflow adoption"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-LLM"],
                        "dynamic_section_title": "Workflow deployment signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_case",
                "block_type": "case_comparison",
                "claim": "Enterprise AI Agent demand is moving from trials into workflow deployment.",
                "reasoning": "Workflow deployment matters because it requires repeatable operations, integrations, and permission controls.",
                "evidence_refs": ["EV-LLM"],
                "used_fact_refs": ["EV-LLM"],
                "public_render": True,
                "evidence_basis": [
                    "Salesforce Agentforce disclosed customer-service workflow deployments for enterprise users."
                ],
                "supporting_facts": [
                    "Salesforce Agentforce disclosed customer-service workflow deployments for enterprise users."
                ],
                "claim_strength": "moderate",
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]
    assert section["composition_status"] == "composed"
    assert section["body_composition_status"] == "composed"
    assert section["evidence_backed"] is True
    assert section["used_fact_refs"] == ["EV-LLM"]


def test_chapter_argument_inherits_llm_claim_evidence_refs_as_used_fact_refs():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Workflow adoption"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-LLM"],
                        "dynamic_section_title": "Workflow deployment signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_case",
                "block_type": "case_comparison",
                "claim": "Enterprise AI Agent demand is moving from trials into workflow deployment.",
                "reasoning": "Workflow deployment matters because it requires repeatable operations and permission controls.",
                "evidence_refs": ["EV-LLM"],
                "public_render": True,
                "evidence_basis": [
                    "Salesforce Agentforce disclosed customer-service workflow deployments for enterprise users."
                ],
                "claim_strength": "moderate",
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]
    assert section["composition_status"] == "composed"
    assert section["body_composition_status"] == "composed"
    assert section["evidence_backed"] is True
    assert section["evidence_refs"] == ["EV-LLM"]
    assert section["used_fact_refs"] == ["EV-LLM"]


def test_chapter_argument_does_not_expand_claim_refs_from_composer_output(monkeypatch):
    import rag_pipeline.agents.chapter_argument_agent as chapter_argument_agent

    def noisy_composer(**_kwargs):
        return {
            "composition_status": "composed",
            "body_composition_status": "composed",
            "paragraph": "Salesforce workflow deployments support the adoption claim.",
            "claim": "Enterprise AI Agent demand is moving from trials into workflow deployment.",
            "reasoning": "Workflow deployment indicates repeatable production use.",
            "supporting_facts": ["Salesforce disclosed workflow deployment pilots."],
            "used_fact_refs": ["EV-LLM", "EV-UNRELATED"],
        }

    monkeypatch.setattr(chapter_argument_agent, "compose_section_paragraph", noisy_composer)
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Workflow adoption"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-UNRELATED"],
                        "dynamic_section_title": "Workflow deployment signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_case",
                "block_type": "case_comparison",
                "claim": "Enterprise AI Agent demand is moving from trials into workflow deployment.",
                "reasoning": "Workflow deployment matters because it requires repeatable operations.",
                "evidence_refs": ["EV-LLM"],
                "used_fact_refs": ["EV-LLM"],
                "source_support_map": {"claim": ["EV-LLM"], "mechanism": ["EV-LLM"], "boundary": ["EV-LLM"]},
                "public_render": True,
                "evidence_basis": ["Salesforce disclosed workflow deployment pilots."],
                "claim_strength": "moderate",
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]
    assert section["used_fact_refs"] == ["EV-LLM"]
    assert section["evidence_refs"] == ["EV-LLM"]


def test_chapter_argument_inherits_llm_used_evidence_ids_as_used_fact_refs():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Workflow adoption"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "dynamic_section_title": "Workflow deployment signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_case",
                "block_type": "case_comparison",
                "claim": "Enterprise AI Agent demand is moving from trials into workflow deployment.",
                "reasoning": "Workflow deployment matters because it requires repeatable operations and permission controls.",
                "used_evidence_ids": ["EV-LLM"],
                "public_render": True,
                "evidence_basis": [
                    "Salesforce Agentforce disclosed customer-service workflow deployments for enterprise users."
                ],
                "claim_strength": "moderate",
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]
    assert section["composition_status"] == "composed"
    assert section["body_composition_status"] == "composed"
    assert section["evidence_backed"] is True
    assert section["evidence_refs"] == ["EV-LLM"]
    assert section["used_fact_refs"] == ["EV-LLM"]


def test_chapter_argument_demotes_metric_layout_when_llm_claim_is_case_signal():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Workflow adoption"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "block_type": "metric_reconciliation",
                        "required_evidence_refs": ["EV-CASE"],
                        "dynamic_section_title": "Market signal",
                    }
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_metric",
                "block_type": "case_comparison",
                "block_affinity": "case_comparison",
                "claim": "Enterprise AI Agent demand is moving from trials into workflow deployment.",
                "reasoning": "Workflow deployment matters because it requires repeatable operations and permission controls.",
                "evidence_refs": ["EV-CASE"],
                "used_fact_refs": ["EV-CASE"],
                "public_render": True,
                "evidence_basis": [
                    "Salesforce Agentforce disclosed customer-service workflow deployments for enterprise users."
                ],
                "supporting_facts": [
                    "Salesforce Agentforce disclosed customer-service workflow deployments for enterprise users."
                ],
                "claim_strength": "moderate",
            }
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    section = packages[0]["sections"][0]
    assert section["composition_status"] == "composed"
    assert section["block_type"] != "metric_reconciliation"
    assert section["block_type"] in {"case_comparison", "integrated_signal"}
    assert section["used_fact_refs"] == ["EV-CASE"]


def test_chapter_argument_fallback_variables_are_public_chinese_labels():
    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Market validation"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "block_type": "metric_reconciliation",
                        "required_evidence_refs": ["EV-M"],
                        "dynamic_section_title": "Market signal",
                    },
                    {
                        "section_id": "s_risk",
                        "block_type": "risk_trigger",
                        "required_evidence_refs": ["EV-R"],
                        "dynamic_section_title": "Risk signal",
                    },
                ],
            }
        ],
        argument_units=[
            {
                "chapter_id": "ch_01",
                "section_id": "s_metric",
                "block_type": "metric_reconciliation",
                "claim": "Market signal is visible.",
                "evidence_refs": ["EV-M"],
                "used_fact_refs": ["EV-M"],
                "evidence_basis": ["A verified source reported a market sizing signal."],
                "claim_strength": "moderate",
                "public_render": True,
            },
            {
                "chapter_id": "ch_01",
                "section_id": "s_risk",
                "block_type": "risk_trigger",
                "claim": "Risk signal is visible.",
                "evidence_refs": ["EV-R"],
                "used_fact_refs": ["EV-R"],
                "evidence_basis": ["A verified source reported an implementation risk signal."],
                "claim_strength": "directional",
                "public_render": True,
            },
        ],
        chapter_evidence_packages=[{"chapter_id": "ch_01"}],
    )

    public_text = "\n".join(
        str(value or "")
        for section in packages[0]["sections"]
        for value in [
            section.get("claim"),
            section.get("reasoning"),
            section.get("mechanism"),
            *[block.get("text") for block in section.get("render_blocks", []) if isinstance(block, dict)],
        ]
    )
    assert "market metric" not in public_text
    assert "risk boundary" not in public_text


def test_chapter_argument_applies_fail_open_body_rewrite(monkeypatch, tmp_path):
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
                "paragraph": (
                    "Salesforce Agentforce deployments show that enterprise agents are moving from isolated trials into customer-service workflows [1]. "
                    "This matters because workflow deployment requires integration, permission control, and repeatable customer operations rather than a single demo. "
                    "The stronger implication is that demand can be assessed through deployment depth: a workflow has to connect with roles, systems, service ownership, and operating accountability. "
                    "That makes this evidence more useful than a generic product launch, while the boundary remains whether the same deployment pattern can repeat across more customers and paid operating contexts."
                ),
                "used_fact_refs": ["EV-C"],
                "citation_refs": ["[1]"],
            }
        },
    )

    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-C"],
                        "dynamic_section_title": "Workflow deployment",
                    }
                ],
            }
        ],
        argument_units=[],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "case_evidence": [
                    {
                        "evidence_id": "EV-C",
                        "source_ref": "[1]",
                        "source_level": "A",
                        "public_fact_card": {
                            "subject": "Salesforce Agentforce",
                            "variable": "customer-service workflow deployment",
                            "action_or_signal": "disclosed",
                            "distilled_fact": "Salesforce Agentforce disclosed customer-service workflow deployments.",
                            "fact_type": "case",
                            "block_affinity": ["case_comparison"],
                        },
                        "public_fact_quality": {"eligible_for_report": True},
                    }
                ],
            }
        ],
    )

    section = packages[0]["sections"][0]
    assert section["body_rewrite_status"] == "rewritten"
    assert "isolated trials" in section["claim"]
    assert section["body_rewrite"]["status"] == "rewritten"


def test_chapter_argument_body_rewrite_is_independent_from_chapter_narrative(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")

    def rewrite_marker(**kwargs):
        packages = kwargs["chapter_packages"]
        packages[0]["sections"][0]["body_rewrite_status"] = "rewritten"
        packages[0]["sections"][0]["body_rewrite"] = {"status": "rewritten"}
        return packages, {"enabled": True, "success_count": 1}

    monkeypatch.setattr("rag_pipeline.agents.chapter_argument_agent.rewrite_sections_for_report", rewrite_marker)

    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-C"],
                        "dynamic_section_title": "Workflow deployment",
                    }
                ],
            }
        ],
        argument_units=[],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "case_evidence": [
                    {
                        "evidence_id": "EV-C",
                        "source_ref": "[1]",
                        "source_level": "A",
                        "public_fact_card": {
                            "subject": "Salesforce Agentforce",
                            "variable": "customer-service workflow deployment",
                            "action_or_signal": "disclosed",
                            "distilled_fact": "Salesforce Agentforce disclosed customer-service workflow deployments.",
                            "fact_type": "case",
                            "block_affinity": ["case_comparison"],
                        },
                        "public_fact_quality": {"eligible_for_report": True},
                    }
                ],
            }
        ],
    )

    section = packages[0]["sections"][0]
    assert section["body_composition_status"] == "composed"
    assert section["body_rewrite_status"] == "rewritten"


def test_chapter_argument_body_rewrite_off_even_when_chapter_narrative_enabled(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")

    packages = run_chapter_argument_agent(
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "Demand validation"}]},
        micro_layouts=[
            {
                "chapter_id": "ch_01",
                "sections": [
                    {
                        "section_id": "s_case",
                        "block_type": "case_comparison",
                        "required_evidence_refs": ["EV-C"],
                        "dynamic_section_title": "Workflow deployment",
                    }
                ],
            }
        ],
        argument_units=[],
        chapter_evidence_packages=[
            {
                "chapter_id": "ch_01",
                "case_evidence": [
                    {
                        "evidence_id": "EV-C",
                        "source_ref": "[1]",
                        "source_level": "A",
                        "public_fact_card": {
                            "subject": "Salesforce Agentforce",
                            "variable": "customer-service workflow deployment",
                            "action_or_signal": "disclosed",
                            "distilled_fact": "Salesforce Agentforce disclosed customer-service workflow deployments.",
                            "fact_type": "case",
                            "block_affinity": ["case_comparison"],
                        },
                        "public_fact_quality": {"eligible_for_report": True},
                    }
                ],
            }
        ],
    )

    section = packages[0]["sections"][0]
    assert section["body_composition_status"] == "composed"
    assert "body_rewrite_status" not in section


def test_chapter_expandable_allows_bounded_supporting_ab_evidence():
    chapter = {
        "chapter_id": "ch_supporting",
        "chapter_title": "Enterprise AI Agent adoption boundaries",
        "chapter_fact_digest": [
            "Official policy defines governance boundaries for enterprise AI Agent adoption.",
            "Research evidence records deployment friction and ROI uncertainty.",
        ],
        "sections": [
            {
                "evidence_refs": ["EV-POLICY", "EV-COUNTER"],
                "supporting_facts": [
                    "Official policy defines governance boundaries.",
                    "Research records ROI uncertainty.",
                ],
            }
        ],
        "evidence_quality_summary": {
            "core_evidence_count": 0,
            "core_ab_source_count": 0,
            "supporting_evidence_count": 3,
            "table_evidence_count": 0,
            "source_level_distribution": {"A": 1, "B": 2},
        },
        "evidence_gaps": [{"type": "insufficient_core_evidence"}],
    }

    assert _chapter_expandable(chapter) is True
