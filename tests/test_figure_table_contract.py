from rag_pipeline.agents.writer_agent_clean import build_writer_report

from tests.helpers import sample_evidence_package, sample_structured_analysis


def test_enterprise_clean_tables_hide_figure_scaffolding():
    result = build_writer_report(
        query="智能农业机器人",
        evidence_package=sample_evidence_package(),
        structured_analysis=sample_structured_analysis(),
    )
    text = result["report_markdown"]
    packages = result["chapter_packages"]

    assert "图表解读" not in text
    assert "资料来源：" not in text
    body = text.split("\n## 附录", 1)[0]
    assert "| 引用 |" not in body
    assert "冲突/差异" not in text
    assert "报告使用方式" not in text
    assert packages
    assert result["validation"]["clean_format"]["table_count"] <= 12
