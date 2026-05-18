from rag_pipeline.agents.writer_agent_clean import build_writer_report

from tests.helpers import sample_evidence_package, sample_structured_analysis


def test_no_internal_analysis_labels_in_final_report():
    result = build_writer_report(
        query="智能农业机器人",
        evidence_package=sample_evidence_package(),
        structured_analysis=sample_structured_analysis(),
    )
    text = result["report_markdown"]

    forbidden = [
        "章节判断",
        "关键事实速览",
        "证据深读",
        "原文事实",
        "行业形势含义",
        "投资/产品判断",
        "与上下章节的联动",
        "战略含义与行动建议",
        "章节关系与参考分析",
        "进入综合决策章的变量",
        "后续判断的重点是确认该口径",
        "同类数据存在口径差异，后续测算应先统一",
    ]

    for phrase in forbidden:
        assert phrase not in text
