from rag_pipeline.agents.writer_agent_clean import select_report_layout


def test_report_plan_selects_product_layout():
    layout = select_report_layout({"report_type": "产品调研报告"})

    assert layout.report_type == "topic_report"
    assert layout.sections
    assert "章节论证" not in layout.sections


def test_report_plan_selects_competitor_layout():
    layout = select_report_layout({"report_name": "竞品分析报告"})

    assert layout.report_type == "topic_report"
    assert layout.sections
