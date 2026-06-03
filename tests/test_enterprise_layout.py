from rag_pipeline.agents.writer_agent_clean import build_writer_report

from tests.helpers import sample_evidence_package, sample_structured_analysis


def test_enterprise_report_has_required_blocks():
    result = build_writer_report(
        query="智能农业机器人",
        evidence_package=sample_evidence_package(),
        structured_analysis=sample_structured_analysis(),
        report_plan={"report_type": "industry_deep"},
    )
    text = result["report_markdown"]

    assert result["report_status"] in {"formal_scored", "final_clean"}
    assert text.strip()
    assert "# 智能农业机器人" in text
    assert "核心观点" in text or "核心观察" in text
    assert ("关键数据" in text or "关键事实" in text or "| 指标 |" in text)
    assert "内容目录" not in text
    assert "图表目录" not in text
    assert "章节任务地图" not in text
    assert "本章核心判断" not in text
    assert "本章小结" not in text
    assert result["layout_plan"]["chapters"]
    assert len({chapter["layout_type"] for chapter in result["layout_plan"]["chapters"]}) >= 2
    assert "风险" in text
    assert "研究口径" in text or "来源附录" in text
