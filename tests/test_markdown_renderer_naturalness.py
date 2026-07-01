from rag_pipeline.agents.final_writer_agent import (
    _ensure_public_core_observation_block,
    _render_key_data_block,
    run_final_writer_agent,
)
from rag_pipeline.agents.markdown_renderer import (
    _compact_chapter_heading,
    render_appendix,
    render_chapter_deep_synthesis,
    render_chapter_package,
    render_executive_summary,
    render_final_reference_analysis,
    render_section,
    render_table_package,
)
from rag_pipeline.agents.public_report_sanitizer import (
    rewrite_internal_gap_language,
    sanitize_public_markdown,
)
from rag_pipeline.flows.report.full_report import finalize_formal_report


def test_dangling_cross_reference_parentheses_are_removed():
    # Referent stripped during cleaning leaves an empty cross-reference.
    assert rewrite_internal_gap_language("产业未来方向与科技生活碰撞的新火花（参见）。") == "产业未来方向与科技生活碰撞的新火花。"
    assert rewrite_internal_gap_language("结论清晰（）。") == "结论清晰。"
    assert "（见）" not in rewrite_internal_gap_language("行业景气（见）持续。")


def test_real_cross_reference_parentheses_are_preserved():
    # Parentheses with a real referent must not be stripped.
    assert "（参见图3）" in rewrite_internal_gap_language("市场规模持续扩大（参见图3）。")


def test_analysis_framework_narration_is_stripped_from_body():
    # Methodology-narration sentences make the report read like an internal tool;
    # they must be dropped while the real industry content is kept.
    text = (
        "2025年具身智能市场规模达53亿元，预计2030年突破千亿元。"
        "这一变化首先影响任务分工和能力配置，要求重新安排人员能力、系统接口和责任边界。"
        "深圳头部企业核心部件国产化率超90%。"
        "当主体行动持续、影响路径清楚时，相关变化才是可解释的分析材料。"
    )
    out = rewrite_internal_gap_language(text)
    assert "53亿元" in out
    assert "国产化率超90%" in out
    assert "任务分工" not in out
    assert "可解释的分析材料" not in out


def test_framework_strip_does_not_blank_an_all_framework_paragraph():
    # If a fragment is entirely framework narration, keep it rather than blank it.
    assert rewrite_internal_gap_language("这一判断可用于梳理岗位任务、能力要求的变化方向。").strip()


def test_analysis_value_narration_and_hedging_leadins_are_cleaned():
    # "这一判断的价值在于…" is internal methodology narration -> whole sentence dropped.
    out1 = rewrite_internal_gap_language(
        "2025年市场规模达53亿元。这一判断的价值在于把资料转化为资源配置的优先顺序。"
    )
    assert "53亿元" in out1
    assert "这一判断的价值在于" not in out1
    # Vague hedging lead-in is stripped but the fact it introduces is kept.
    out2 = rewrite_internal_gap_language("公开材料提到，2025年国内整机企业超140家。")
    assert "公开材料提到" not in out2
    assert "140家" in out2


def test_compact_chapter_heading_uses_complete_phrase_not_mid_clause_truncation():
    title = (
        "\u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c"
        "\u5e94\u4ece\u4eba\u624d\u9700\u6c42\u3001\u5c97\u4f4d\u7ed3\u6784"
        "\u548c\u7ec4\u7ec7\u8d22\u52a1\u6cbb\u7406\u9700\u6c42\u9a8c\u8bc1"
    )

    heading = _compact_chapter_heading(title, max_chars=28)

    assert heading == "\u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c"
    assert not heading.endswith("\u8d22")


def test_core_observation_ignores_bridge_sentences_and_keeps_concrete_facts():
    markdown = (
        "# \u4f1a\u8ba1\u5b66\u4e13\u4e1a\u5c31\u4e1a\u7814\u7a76\u62a5\u544a\n\n"
        "## 1. \u4f1a\u8ba1\u5c31\u4e1a\u7684\u73b0\u5b9e\u4ef7\u503c\n"
        "\u5982\u679c\u628a\u8fd9\u4e00\u4fe1\u53f7\u4e0e\u76f8\u90bb\u4e8b\u5b9e\u5bf9\u7167\uff0c\u5206\u6790\u91cd\u70b9\u4f1a\u4ece\u201c\u6709\u6ca1\u6709\u53d8\u5316\u201d\u8f6c\u5411\u201c\u53d8\u5316\u53d1\u751f\u5728\u54ea\u91cc\u3001\u8c01\u627f\u62c5\u53d8\u5316\u6210\u672c\u3001\u54ea\u4e9b\u73af\u8282\u5148\u53d7\u5f71\u54cd\u201d\u3002[1][2]\n"
        "\u8fd9\u4f1a\u628a\u5355\u70b9\u4e8b\u5b9e\u653e\u8fdb\u8fde\u7eed\u53d8\u5316\u4e2d\u89c2\u5bdf\uff0c\u4f7f\u5c97\u4f4d\u4efb\u52a1\u3001\u7ec4\u7ec7\u5b89\u6392\u548c\u80fd\u529b\u8981\u6c42\u4e4b\u95f4\u7684\u5173\u7cfb\u66f4\u6e05\u695a\u3002[1][2]\n"
        "\u65e0\u9521\u5e022023\u5e74\u4e09\u5b63\u5ea6\u4e0e2025\u5e74\u4e00\u5b63\u5ea6\u4eba\u529b\u8d44\u6e90\u5e02\u573a\u7edf\u8ba1\u663e\u793a\uff0c\u4f9b\u6c42\u603b\u91cf\u5747\u4fdd\u6301\u5e73\u7a33\u589e\u957f\uff0c\u4e14\u4e0e\u5de5\u4e1a\u751f\u4ea7\u6062\u590d\u3001\u6d88\u8d39\u590d\u82cf\u53ca\u670d\u52a1\u4e1a\u589e\u52bf\u76f8\u543b\u5408\u3002[1][2]\n"
        "\u65e0\u9521\u5e022023\u5e74\u4e09\u5b63\u5ea6\u4e0e2025\u5e74\u4e00\u5b63\u5ea6\u4eba\u529b\u8d44\u6e90\u5e02\u573a\u7edf\u8ba1\u663e\u793a\uff0c\u4f9b\u6c42\u603b\u91cf\u5747\u4fdd\u6301\u5e73\u7a33\u589e\u957f\uff0c\u4e14\u4e0e\u5de5\u4e1a\u751f\u4ea7\u6062\u590d\u3001\u6d88\u8d39\u590d\u82cf\u53ca\u670d\u52a1\u4e1a\u589e\u52bf\u76f8\u543b\u5408\u3002[1][2]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]

    assert "\u5982\u679c\u628a\u8fd9\u4e00\u4fe1\u53f7" not in summary
    assert "\u8fd9\u4f1a\u628a\u5355\u70b9\u4e8b\u5b9e" not in summary
    assert summary.count("\u65e0\u9521\u5e022023\u5e74\u4e09\u5b63\u5ea6") == 1


def test_core_observation_dedupes_boundary_already_embedded_in_claim():
    boundary = "\u9884\u6d4b\u6570\u636e\u57fa\u4e8e\u4e0d\u540c\u7814\u7a76\u673a\u6784\u7684\u6a21\u578b\u5047\u8bbe\uff0c\u7edf\u8ba1\u53e3\u5f84\u6db5\u76d6\u8303\u56f4\uff08\u662f\u5426\u5305\u542b\u4f20\u7edf\u5de5\u4e1a\u673a\u5668\u4eba\u6216\u6cdbAI\u786c\u4ef6\uff09\u672a\u5b8c\u5168\u7edf\u4e00\uff0c\u9700\u4ee5\u5b9e\u9645\u4ea4\u4ed8\u4e0e\u8425\u6536\u6570\u636e\u4e3a\u51c6\u3002[1]"
    markdown = (
        "# \u4e2d\u56fd\u5177\u8eab\u667a\u80fd\u673a\u5668\u4eba\u7814\u7a76\u62a5\u544a\n\n"
        "## 1. \u9700\u6c42\u4e0e\u5e02\u573a\u7a7a\u95f4\n"
        "\u653f\u7b56\u4e0e\u8d44\u672c\u53cc\u91cd\u9a71\u52a8\u53e0\u52a0\u6280\u672f\u8fed\u4ee3\uff0c\u63a8\u52a8\u4ea7\u4e1a\u89c4\u6a21\u8fdb\u5165\u9ad8\u901f\u589e\u957f\u901a\u9053\uff0c\u4f46\u5e7f\u4e49\u4e0e\u72ed\u4e49\u5177\u8eab\u667a\u80fd\u7684\u754c\u5b9a\u4e0d\u540c\u5bfc\u81f4\u9884\u6d4b\u6570\u636e\u5448\u73b0\u533a\u95f4\u5206\u5e03 "
        + boundary
        + "\n"
        + boundary
        + "\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]
    bullets = [line for line in summary.splitlines() if line.strip().startswith("- ")]

    assert summary.count("\u9884\u6d4b\u6570\u636e\u57fa\u4e8e\u4e0d\u540c\u7814\u7a76\u673a\u6784") == 1
    assert len(bullets) == 1


def test_core_observation_dedupes_metric_fact_and_evidence_context():
    markdown = (
        "# 中国具身智能机器人商业化研究报告\n\n"
        "## 1. 市场空间与商业化阶段\n"
        "2025年中国具身智能机器人市场规模预计达到52.95亿元，约占全球市场份额的27%。[1]\n"
        "公开材料提到，2025年中国具身智能机器人市场规模预计达52.95亿元，占全球约27%。[1]\n"
        "宏观市场规模数据直接反映产业商业化初期的体量与全球占比，表明中国在该领域已具备初步的产业基础与需求规模，脱离纯概念验证阶段。[1]\n"
        "\n"
        "## 2. 竞争格局\n"
        "头部企业正在围绕工业、物流和服务场景推进样机验证与客户试点，竞争重点从单点能力展示转向场景适配和交付能力。[2]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]
    bullets = [line for line in summary.splitlines() if line.strip().startswith("- ")]

    assert "公开材料提到" not in summary
    assert "2025年中国具身智能机器人市场规模预计达到52.95亿元" not in summary
    assert len(bullets) == 2
    assert "宏观市场规模数据直接反映" in bullets[0]
    assert "头部企业正在围绕工业、物流和服务场景" in bullets[1]


def test_core_observation_excludes_verification_caveats_from_opening():
    markdown = (
        "# 中国具身智能机器人商业化研究报告\n\n"
        "## 1. 市场空间与商业化阶段\n"
        "当前判断多基于定性信号，具体市场份额、技术代差及商业化渗透率等量化指标尚待进一步验证。[1][2]\n"
        "由于现阶段尚缺乏具体的全球市占率、技术专利横向对比或出口规模等量化指标支撑，相关结论的精确度仍需结合后续行业白皮书及权威统计数据进行交叉验证。[1][2]\n"
        "宏观市场规模数据直接反映产业商业化初期的体量与全球占比，表明中国在该领域已具备初步的产业基础与需求规模。[1]\n"
        "\n"
        "## 2. 竞争格局\n"
        "头部企业正在围绕工业、物流和服务场景推进样机验证与客户试点，竞争重点从单点能力展示转向场景适配和交付能力。[2]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]

    assert "当前判断多基于定性信号" not in summary
    assert "尚待进一步验证" not in summary
    assert "缺乏具体的全球市占率" not in summary
    assert "宏观市场规模数据直接反映" in summary
    assert "头部企业正在围绕工业、物流和服务场景" in summary


def test_core_observation_trims_trailing_caveat_from_judgment_line():
    markdown = (
        "# 中国具身智能机器人商业化研究报告\n\n"
        "## 1. 产业链协同\n"
        "政策与产业研究导向明确指向场景验证向产业链协同演进，表明核心环节已具备实际落地基础，产业重心正由早期样本测试转向规模化应用探索 缺乏具体技术环节成熟度分级数据，产业崛起的具体时间节点与各环节渗透率需以实际可复核材料与产能释放为准。[2]\n"
        "产业链上下游企业正在围绕整机、核心零部件和场景集成推进协同验证。[3]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]

    assert "政策与产业研究导向明确指向场景验证向产业链协同演进" in summary
    assert "缺乏具体技术环节成熟度分级数据" not in summary
    assert "渗透率需以实际可复核材料" not in summary


def test_core_observation_excludes_qualitative_label_lines():
    markdown = (
        "# 中国具身智能机器人商业化研究报告\n\n"
        "## 1. 市场空间\n"
        "“领跑全球”与“加速落地”为定性描述。[1][2][3]\n"
        "政策与产业研究导向明确指向场景验证向产业链协同演进，表明核心环节已具备实际落地基础。[3]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]

    assert "为定性描述" not in summary
    assert "政策与产业研究导向明确指向" in summary


def test_core_observation_excludes_source_attribution_lines():
    markdown = (
        "# 中国具身智能机器人商业化研究报告\n\n"
        "## 1. 产业趋势\n"
        "权威机构指出具身智能正从场景落地迈向产业崛起。[2]\n"
        "政策与产业研究导向明确指向场景验证向产业链协同演进，表明核心环节已具备实际落地基础。[2]\n"
        "头部企业正在围绕工业、物流和服务场景推进样机验证与客户试点，竞争重点转向场景适配和交付能力。[3]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]

    assert "权威机构指出" not in summary
    assert "政策与产业研究导向明确指向" in summary
    assert "头部企业正在围绕工业、物流和服务场景" in summary


def test_renderer_deep_synthesis_uses_public_prose_not_methodology_labels(monkeypatch):
    monkeypatch.setenv("REPORT_RENDER_CHAPTER_DEEP_SYNTHESIS", "true")
    lines = render_chapter_deep_synthesis(
        {
            "chapter_title": "\u4f1a\u8ba1\u5c31\u4e1a\u80fd\u529b\u53d8\u5316",
            "chapter_summary": {
                "key_takeaway": "\u8bfe\u7a0b\u548c\u5c97\u4f4d\u8981\u6c42\u90fd\u5f00\u59cb\u5f3a\u8c03\u6570\u636e\u5206\u6790\u80fd\u529b\u3002",
                "mechanisms": [
                    "\u8d22\u52a1\u5de5\u5177\u66f4\u65b0\u4f1a\u6539\u53d8\u8bfe\u7a0b\u8bad\u7ec3\u548c\u4f01\u4e1a\u7b5b\u9009\u6807\u51c6"
                ],
            },
        }
    )
    text = "\n".join(lines)

    assert "\u5f71\u54cd\u8def\u5f84" not in text
    assert "\u7ae0\u8282\u7ed3\u8bba" not in text
    assert "\u4e8b\u5b9e\u4e4b\u95f4" in text or "\u8d22\u52a1\u5de5\u5177" in text


def test_renderer_final_reference_analysis_uses_public_prose_not_path_label(monkeypatch):
    monkeypatch.setenv("REPORT_RENDER_FINAL_REFERENCE_ANALYSIS", "true")
    lines = render_final_reference_analysis(
        {
            "chapter_syntheses": [
                {
                    "chapter_title": "\u4f1a\u8ba1\u5c31\u4e1a\u80fd\u529b\u53d8\u5316",
                    "chapter_summary": {
                        "key_takeaway": "\u4f1a\u8ba1\u5c97\u4f4d\u5bf9\u6570\u636e\u5206\u6790\u80fd\u529b\u7684\u8981\u6c42\u63d0\u9ad8\u3002",
                        "mechanisms": [
                            "\u8bfe\u7a0b\u4f53\u7cfb\u5f00\u59cb\u52a0\u5165\u667a\u80fd\u8d22\u52a1\u548c\u6570\u636e\u5206\u6790\u8bad\u7ec3"
                        ],
                    },
                }
            ]
        }
    )
    text = "\n".join(lines)

    assert "\u5f71\u54cd\u8def\u5f84\u53ef\u4ee5\u6982\u62ec\u4e3a" not in text
    assert "\u5f3a\u5f31\u6392\u5e8f" not in text
    assert "\u8bfe\u7a0b\u4f53\u7cfb" in text


def test_render_section_drops_legacy_generic_action_advice_from_render_blocks():
    lines = render_section(
        {
            "section_title": "\u5546\u4e1a\u5316\u8fdb\u5c55",
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "\u4eba\u5f62\u673a\u5668\u4eba\u5728\u6625\u665a\u3001\u5e99\u4f1a\u548c\u5546\u573a\u7b49\u573a\u666f\u5df2\u6709\u516c\u5f00\u5e94\u7528\u6837\u672c\u3002[1]",
                },
                {
                    "type": "paragraph",
                    "text": "\u636e\u6b64\u53ef\u4f18\u5148\u6392\u5e03\u8d44\u6e90\u6295\u5165\u987a\u5e8f\uff0c\u5e76\u9501\u5b9a\u9700\u8981\u6301\u7eed\u8ddf\u8e2a\u7684\u5173\u952e\u53d8\u91cf\u3002[1]",
                },
                {
                    "type": "paragraph",
                    "text": "\u653e\u56de\u5177\u4f53\u95ee\u9898\u770b\uff0c\u91cd\u70b9\u4e0d\u662f\u53ea\u770b\u4e00\u4e2a\u6307\u6807\uff0c\u800c\u662f\u6307\u6807\u53e3\u5f84\u4e0e\u53ef\u6bd4\u6027\u7684\u4ef7\u503c\u4e0d\u5728\u4e8e\u51fa\u73b0\u4e00\u4e2a\u6848\u4f8b\u3002[2]",
                },
            ],
            "citation_refs": ["[1]", "[2]"],
        }
    )
    text = "\n".join(lines)

    assert "\u6625\u665a" in text
    assert "\u636e\u6b64\u53ef\u4f18\u5148\u6392\u5e03" not in text
    assert "\u653e\u56de\u5177\u4f53\u95ee\u9898\u770b" not in text


def test_core_observation_drops_roadmap_lines_and_leads_with_metric():
    from rag_pipeline.agents.final_writer_agent import _looks_like_core_observation_bridge_line

    # Roadmap/scope sentences describe the chapter, not a finding.
    assert _looks_like_core_observation_bridge_line("\u672c\u7ae0\u5c06\u91cd\u70b9\u5256\u6790\u9700\u6c42\u4fe1\u53f7\u7684\u8fb9\u754c\u3002[1]") is True
    assert _looks_like_core_observation_bridge_line("2023\u5e74\u4e2d\u56fd\u5e02\u573a\u89c4\u6a21\u8fbe4186\u4ebf\u5143\u3002[2]") is False

    markdown = (
        "# \u6d4b\u8bd5\u62a5\u544a\n\n"
        "## 1. \u9700\u6c42\u4e0e\u5e02\u573a\u7a7a\u95f4\n"
        "\u672c\u7ae0\u5c06\u91cd\u70b9\u5256\u6790\u9700\u6c42\u4fe1\u53f7\u7684\u8fb9\u754c\u4e0e\u542f\u793a\uff0c\u4e3a\u5224\u65ad\u5e02\u573a\u7a7a\u95f4\u63d0\u4f9b\u65b9\u5411\u6027\u89c6\u89d2\u3002[1]\n"
        "\u5149\u5927\u8bc1\u5238\u8fd1\u671f\u4e0a\u8c03\u67d0\u516c\u53f8\u8bc4\u7ea7\uff0c\u53cd\u6620\u8d44\u672c\u5e02\u573a\u7684\u65e9\u671f\u8ba4\u53ef\u3002[1]\n"
        "2023\u5e74\u4e2d\u56fd\u5177\u8eab\u667a\u80fd\u5e02\u573a\u89c4\u6a21\u5df2\u8fbe4186\u4ebf\u5143\uff0c\u6709\u671b\u57282027\u5e74\u589e\u957f\u81f36328\u4ebf\u5143\u3002[2]\n"
    )

    updated = _ensure_public_core_observation_block(markdown)
    summary = updated.split("## 1.", 1)[0]

    # The roadmap sentence must not be lifted into \u6838\u5fc3\u89c2\u5bdf.
    assert "\u672c\u7ae0\u5c06\u91cd\u70b9\u5256\u6790" not in summary
    # The metric-bearing observation should lead the summary, not the first body line.
    assert "4186\u4ebf\u5143" in summary
    first_bullet = next(line for line in summary.splitlines() if line.strip().startswith("- "))
    assert "4186\u4ebf\u5143" in first_bullet


def test_core_observation_drops_split_metric_fragments():
    from rag_pipeline.agents.final_writer_agent import _looks_like_core_observation_bridge_line

    # A dangling metric like "5%\uff1b\u4e0a\u6e38\u6db5\u76d6\u2026" is a split sentence, not an observation.
    assert _looks_like_core_observation_bridge_line("5%\uff1b\u4e0a\u6e38\u6db5\u76d6\u7a7a\u5fc3\u676f\u7535\u673a\u3001\u4f20\u611f\u5668\u3002[2]") is True
    assert _looks_like_core_observation_bridge_line("\u3001\u4f20\u611f\u5668\u4e0e\u63a7\u5236\u5668\u6784\u6210\u4e0a\u6e38\u3002[2]") is True
    # Real number-led observations must still be kept.
    assert _looks_like_core_observation_bridge_line("2025\u5e74\u5e02\u573a\u89c4\u6a21\u8fbe4186\u4ebf\u5143\u3002[1]") is False
    assert _looks_like_core_observation_bridge_line("\u4e2d\u56fd\u4f01\u4e1a\u4ee590%\u7684\u5168\u7403\u4efd\u989d\u5360\u636e\u4f18\u52bf\u3002[3]") is False


def test_compact_chapter_heading_returns_empty_instead_of_mid_clause_fragment():
    heading = _compact_chapter_heading(
        "会计与财务管理类专业的课程基础已从单一财会知识扩展为涵盖经济、金融、税务及管理的复合型理论体系",
        max_chars=28,
    )

    assert heading == ""


def test_compact_chapter_heading_rejects_already_truncated_ellipsis_title():
    heading = _compact_chapter_heading(
        "人工智能与大数据技术的深度融合正驱动会计专业人才培养体...",
        max_chars=28,
    )

    assert heading == ""


def test_render_chapter_package_does_not_repeat_generic_actionable_fallback():
    repeated = "\u540c\u53e3\u5f84\u6307\u6807\u3001\u7ed3\u679c\u9a8c\u8bc1\u548c\u53cd\u5411\u6837\u672c\u662f\u5224\u65ad\u5f3a\u5f31\u53d8\u5316\u7684\u6838\u5fc3\u53d8\u91cf\u3002"
    chapter = {
        "chapter_title": "\u4f1a\u8ba1\u5c31\u4e1a\u4e0e\u4eba\u624d\u9700\u6c42",
        "sections": [
            {
                "section_title": "\u5c97\u4f4d\u7ed3\u6784\u53d8\u5316",
                "claim": "\u533a\u57df\u4eba\u529b\u9700\u6c42\u53ef\u4ee5\u4f5c\u4e3a\u4f1a\u8ba1\u5c31\u4e1a\u7684\u80cc\u666f\u53c2\u7167\u3002",
                "reasoning": "\u8fd9\u4e00\u5224\u65ad\u9700\u8981\u56de\u5230\u5c97\u4f4d\u7c7b\u578b\u548c\u8bfe\u7a0b\u80fd\u529b\u5339\u914d\u4e0a\u7406\u89e3\u3002",
                "actionable": repeated,
                "citation_refs": ["[1]"],
                "evidence_refs": ["EV-1"],
                "evidence_backed": True,
            },
            {
                "section_title": "\u8bfe\u7a0b\u80fd\u529b\u8f6c\u5411",
                "claim": "\u667a\u80fd\u8d22\u52a1\u8bad\u7ec3\u6b63\u5728\u6539\u53d8\u4f1a\u8ba1\u4e13\u4e1a\u5c31\u4e1a\u7684\u80fd\u529b\u7ed3\u6784\u3002",
                "reasoning": "\u8fd9\u4e00\u53d8\u5316\u5e94\u8be5\u548c\u5b66\u6821\u8bfe\u7a0b\u3001\u5de5\u5177\u8bad\u7ec3\u548c\u5b9e\u4e60\u5c97\u4f4d\u653e\u5728\u4e00\u8d77\u5206\u6790\u3002",
                "actionable": repeated,
                "citation_refs": ["[2]"],
                "evidence_refs": ["EV-2"],
                "evidence_backed": True,
            },
        ],
    }

    rendered = render_chapter_package(chapter, 1)

    assert rendered.count(repeated) <= 1


def test_render_section_drops_current_generic_industry_fallback_sentences():
    lines = render_section(
        {
            "section_title": "具身智能商业化信号",
            "claim": "具身智能企业开始把样机验证推向具体客户场景。",
            "reasoning": "公开案例显示，部分企业围绕工业、物流和服务场景开展样机验证。",
            "actionable": (
                "这会影响相关主体的产品设计、资源投入和组织协作方式。"
                "产业含义在于资源会更集中流向已经出现真实需求和应用动作的环节。"
                "后续判断主要看资源投入是否持续、应用场景是否扩大、执行成本和风险约束是否同步改善。"
            ),
            "render_blocks": [
                {"type": "paragraph", "text": "具身智能企业开始把样机验证推向具体客户场景。"},
                {"type": "paragraph", "text": "公开案例显示，部分企业围绕工业、物流和服务场景开展样机验证。"},
                {
                    "type": "paragraph",
                    "text": (
                        "这会影响相关主体的产品设计、资源投入和组织协作方式。"
                        "产业含义在于资源会更集中流向已经出现真实需求和应用动作的环节。"
                        "后续判断主要看资源投入是否持续、应用场景是否扩大、执行成本和风险约束是否同步改善。"
                    ),
                },
            ],
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "evidence_backed": True,
        }
    )

    rendered = "\n".join(lines)
    assert "具身智能企业开始把样机验证推向具体客户场景" in rendered
    assert "这会影响相关主体" not in rendered
    assert "产业含义在于" not in rendered
    assert "后续判断主要看资源投入是否持续" not in rendered


def test_render_chapter_package_replaces_generic_dynamic_fallback_title_with_claim_context():
    chapter = {
        "chapter_title": "\u4f1a\u8ba1\u5c31\u4e1a\u4e0e\u533a\u57df\u9700\u6c42",
        "sections": [
            {
                "section_title": "\u5177\u4f53\u573a\u666f\u4f1a\u600e\u6837\u6539\u53d8\u7ed3\u8bba",
                "claim": "\u516c\u5171\u5c31\u4e1a\u670d\u52a1\u5e73\u53f0\u7684\u7edf\u8ba1\u6846\u67b6\u4ee5\u533a\u57df\u7efc\u5408\u4eba\u529b\u4f9b\u9700\u4e3a\u6838\u5fc3\u5355\u5143\u3002",
                "reasoning": "\u8fd9\u7c7b\u6570\u636e\u53ef\u4ee5\u89e3\u91ca\u533a\u57df\u5c31\u4e1a\u80cc\u666f\uff0c\u4f46\u4e0d\u5b9c\u76f4\u63a5\u5916\u63a8\u4e3a\u4f1a\u8ba1\u4e13\u4e1a\u5168\u56fd\u5c31\u4e1a\u7ed3\u8bba\u3002",
                "citation_refs": ["[1]"],
                "evidence_refs": ["EV-1"],
                "evidence_backed": True,
            }
        ],
    }

    rendered = render_chapter_package(chapter, 1)

    assert "\u5177\u4f53\u573a\u666f\u4f1a\u600e\u6837\u6539\u53d8\u7ed3\u8bba" not in rendered
    assert "### \u516c\u5171\u5c31\u4e1a\u670d\u52a1\u5e73\u53f0\u7684\u7edf\u8ba1\u6846\u67b6" in rendered


def test_render_section_removes_review_style_bridge_sentences_but_keeps_fact():
    lines = render_section(
        {
            "section_title": "公共就业服务平台的统计框架",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "evidence_backed": True,
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "公共就业服务机构汇总的供求数据印证区域人才市场具备韧性。"
                        "这种差异决定了相关判断能否从话题热度推进到更可复核的分析结论 "
                        "可复核内容适合先进入主线，再说明它会影响哪些对象、流程或约束条件。"
                        "暂时缺少覆盖的外推只适合作为边界或待验证问题处理。"
                    ),
                }
            ],
        }
    )

    rendered = "\n".join(lines)
    assert "公共就业服务机构汇总的供求数据" in rendered
    assert "可复核内容适合" not in rendered
    assert "待验证问题处理" not in rendered
    assert "话题热度推进" not in rendered


def test_render_chapter_package_replaces_duplicate_chapter_title_with_section_title():
    seen = {"会计学专业就业的现实价值"}
    markdown = render_chapter_package(
        {
            "chapter_title": "会计学专业就业的现实价值",
            "sections": [
                {
                    "section_title": "主要结论如何变化",
                    "claim": "组织财务治理与基础岗位需求受实体经济指标修复牵引。",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [{"type": "paragraph", "text": "组织财务治理与基础岗位需求受实体经济指标修复牵引。"}],
                },
                {
                    "section_title": "具体场景是否已经成立",
                    "dynamic_section_title": "就业市场验证应聚焦公共就业服务机构的实际招聘口径",
                    "claim": "公共就业数据可以校准岗位需求判断。",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [{"type": "paragraph", "text": "公共就业数据可以校准岗位需求判断。"}],
                }
            ],
        },
        2,
        seen_chapter_titles=seen,
    )

    assert "## 2. 会计学专业就业的现实价值" not in markdown
    assert "## 2. 主要结论如何变化" not in markdown
    assert "## 2. 组织财务治理与基础岗位需求受实体经济指标修复牵引" in markdown


def test_duplicate_chapter_title_can_fall_back_to_render_block_text():
    seen = {"会计学专业就业的现实价值"}
    markdown = render_chapter_package(
        {
            "chapter_title": "会计学专业就业的现实价值",
            "sections": [
                {
                    "section_title": "关键判断如何变化",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "组织财务治理与基础岗位需求受实体经济指标修复牵引，呈现结构性稳定特征。",
                        }
                    ],
                }
            ],
        },
        2,
        seen_chapter_titles=seen,
    )

    assert "## 2. 组织财务治理与基础岗位需求受实体经济指标修复牵引" in markdown
    assert "## 2. 会计学专业就业的现实价值" not in markdown


def test_duplicate_chapter_title_uses_angle_after_connector():
    markdown = render_chapter_package(
        {
            "chapter_title": "会计学专业就业的现实价值应从人才需求、岗位结构和组织财务治理需求验证，而不是套用产业赛道框架",
            "sections": [
                {
                    "section_title": "主要结论如何变化",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [{"type": "paragraph", "text": "组织财务治理与基础岗位需求受实体经济指标修复牵引。"}],
                }
            ],
        },
        2,
        seen_chapter_titles={"会计学专业就业的现实价值"},
    )

    assert "## 2. 人才需求、岗位结构和组织财务治理需求验证" in markdown
    assert "## 2. 会计学专业就业的现实价值" not in markdown


def test_metric_fact_is_rendered_as_sentence_not_bare_label():
    lines = render_section(
        {
            "section_title": "\u5e02\u573a\u89c4\u6a21\u80fd\u5426\u9a8c\u8bc1",
            "block_type": "metric_reconciliation",
            "evidence_refs": ["EV-METRIC"],
            "used_fact_refs": ["EV-METRIC"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143\u3002",
                    "fact_type": "metric",
                    "metric": "\u5e02\u573a\u89c4\u6a21",
                    "value": "\u8fbe8.2\u4ebf\u5143",
                    "period": "2025\u5e74",
                    "scope": "\u4e2d\u56fd\u4eba\u5f62\u673a\u5668\u4eba",
                }
            ],
        }
    )

    text = "\n".join(lines)
    assert "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143" not in text
    assert "\u5e02\u573a\u89c4\u6a21" in text
    assert "\u8fbe8.2\u4ebf\u5143" in text


def test_render_section_keeps_public_fact_chain_sentence():
    lines = render_section(
        {
            "section_title": "\u9700\u6c42\u4fe1\u53f7\u662f\u5426\u6210\u7acb",
            "block_type": "case_comparison",
            "evidence_refs": ["[1]"],
            "used_fact_refs": ["EV-1"],
            "citation_refs": ["[1]"],
            "evidence_backed": True,
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "\u5df2\u6709\u4e8b\u5b9e\u94fe\u663e\u793a\uff1a\u591a\u5730\u62ab\u9732\u4f4e\u7a7a\u822a\u7ebf\u8bd5\u70b9\u548c\u91c7\u8d2d\u52a8\u4f5c\u3002"
                        "\u8fd9\u4e9b\u6750\u6599\u53ef\u4ee5\u8bf4\u660e\u9700\u6c42\u4fe1\u53f7\u5df2\u7ecf\u8fdb\u5165\u53ef\u89c2\u5bdf\u9636\u6bb5\uff0c"
                        "\u4f46\u4ecd\u9700\u8981\u7528\u5ba2\u6237\u590d\u5236\u3001\u8fd0\u8425\u9891\u6b21\u548c\u4ed8\u8d39\u94fe\u8def\u7ee7\u7eed\u6821\u9a8c\u3002"
                    ),
                }
            ],
        }
    )

    text = "\n".join(lines)
    assert "\u5df2\u6709\u4e8b\u5b9e\u94fe\u663e\u793a" in text
    assert "\u9700\u6c42\u4fe1\u53f7\u5df2\u7ecf\u8fdb\u5165\u53ef\u89c2\u5bdf\u9636\u6bb5" in text
    assert "[1]" in text


def test_render_section_naturalizes_mechanical_transition_prefixes():
    lines = render_section(
        {
            "section_title": "产业链集聚",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "其中，广东集聚全国30%以上低空经济产业链企业。",
                },
                {
                    "type": "paragraph",
                    "text": "生产许可证已颁发。 其中，过去一年广州亿航智能获颁生产许可证。",
                },
                {
                    "type": "paragraph",
                    "text": "同时，来源为第三方分析，无具体量化数据。",
                },
                {
                    "type": "paragraph",
                    "text": "落到行业含义上，订单、运营频次和客户付费决定机会能否扩大。",
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "广东集聚全国30%以上低空经济产业链企业" in text
    assert "过去一年广州亿航智能获颁生产许可证" in text
    assert "可复核材料、运营频次和持续使用" in text
    assert "订单、运营频次和客户付费" not in text
    assert "其中，" not in text
    assert "同时，" not in text
    assert "落到行业含义上，" not in text


def test_render_section_inserts_sentence_boundary_before_bridge_phrases():
    lines = render_section(
        {
            "section_title": "产业信号是否可持续",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "具体案例支撑 已披露动作如果继续出现在更多主体、场景和时间窗口中，机会判断会更扎实",
                },
                {
                    "type": "paragraph",
                    "text": "真实产业机会 接下来更值得观察的是订单、运营频次、客户付费、政策执行和安全记录",
                },
                {
                    "type": "paragraph",
                    "text": "数据来源说明 来源为第三方分析，无具体量化数据",
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "支撑 已披露动作" not in text
    assert "机会 接下来" not in text
    assert "说明 来源为" not in text
    assert "具体案例支撑" in text
    assert "已披露动作如果继续" not in text
    assert "真实产业机会" in text
    assert "接下来更值得观察" not in text
    assert "订单" not in text
    assert "客户付费" not in text
    assert "数据来源说明" in text
    assert "来源为第三方分析" not in text


def test_render_section_strips_repeated_generic_bridge_sentences():
    lines = render_section(
        {
            "section_title": "产业信号是否可持续",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "政策、规模和技术形成正向循环，降低市场进入门槛并刺激需求。"
                        "已披露动作如果继续出现在更多主体、场景和时间窗口中，机会判断会更扎实；"
                        "如果缺少案例或数据承接，结论仍需要保留弹性。"
                        "这种差异决定了相关机会能否从话题热度推进到真实产业机会。"
                        "接下来更值得观察的是订单、运营频次、客户付费、政策执行和安全记录。"
                        "订单和运营频次决定需求是否真实扩散，客户付费决定商业模式是否成立，"
                        "政策执行和安全记录则决定风险边界是否可控；这些指标共同决定机会是扩大、收缩，还是停留在示范项目。"
                    ),
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "政策、规模和技术形成正向循环" in text
    assert "已披露动作如果继续" not in text
    assert "这种差异决定了相关机会能否" not in text
    assert "接下来更值得观察的是订单" not in text
    assert "这些指标共同决定机会" not in text


def test_render_section_rewrites_analysis_voice_into_report_voice():
    lines = render_section(
        {
            "section_title": "风险边界如何影响机会",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "本判断基于间接理论推导，缺乏低空经济具体市场数据，仅供方向参考。",
                },
                {
                    "type": "paragraph",
                    "text": "这说明风险边界是推翻或削弱本章判断的触发条件。",
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "本判断" not in text
    assert "本章判断" not in text
    assert "这一结论基于间接理论推导" in text
    assert "前述结论" in text


def test_render_section_skips_standalone_source_provenance_notes():
    lines = render_section(
        {
            "section_title": "产业信号是否可持续",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "低空政策体系不断完善，低空经济规模与日俱增，低空飞行技术助力应用。",
                },
                {
                    "type": "paragraph",
                    "text": "来源为赛文交通网文章，属于第三方分析，无具体量化数据。",
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "低空政策体系不断完善" in text
    assert "来源为赛文交通网文章" not in text
    assert "无具体量化数据" not in text


def test_render_section_strips_embedded_source_provenance_note():
    lines = render_section(
        {
            "section_title": "产业信号是否可持续",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "政策、规模和技术形成正向循环，降低市场进入门槛并刺激需求。"
                        "来源为赛文交通网文章，属于第三方分析，无具体量化数据。"
                        "商业化程度仍需具体案例支撑。"
                    ),
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "政策、规模和技术形成正向循环" in text
    assert "商业化程度仍需具体案例支撑" in text
    assert "来源为赛文交通网文章" not in text
    assert "无具体量化数据" not in text


def test_render_section_drops_empty_public_material_intro():
    lines = render_section(
        {
            "section_title": "补充观察",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {"type": "paragraph", "text": "广东已集聚全国30%以上的低空经济产业链企业。"},
                {"type": "paragraph", "text": "公开材料提到，"},
            ],
        }
    )

    text = "\n".join(lines)
    assert "广东已集聚全国30%以上" in text
    assert "公开材料提到" not in text


def test_render_section_inserts_boundary_before_limitation_sentences():
    lines = render_section(
        {
            "section_title": "外溢效应是否成立",
            "citation_refs": ["[1]"],
            "evidence_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "这些指标共同决定机会是扩大、收缩，还是停留在示范项目 原文未提及低空经济，外溢效应属于逻辑推断",
                },
                {
                    "type": "paragraph",
                    "text": "这些指标共同决定机会是扩大、收缩，还是停留在示范项目 该数据为宏观先行指标，低空经济订单变化需企业公开数据验证",
                },
                {
                    "type": "paragraph",
                    "text": "这些指标共同决定机会是扩大、收缩，还是停留在示范项目 本判断基于间接理论推导，缺乏低空经济具体市场数据",
                },
                {
                    "type": "paragraph",
                    "text": "这些指标共同决定机会是扩大、收缩，还是停留在示范项目 标准制定到实施有周期，对企业实际商业合同的影响短期内不明显",
                },
                {
                    "type": "paragraph",
                    "text": "这些指标共同决定机会是扩大、收缩，还是停留在示范项目 目前整治行动在总体产业层面，影响路径和时间不确定",
                },
            ],
        }
    )

    text = "\n".join(lines)
    assert "项目 原文未提及" not in text
    assert "项目 该数据为" not in text
    assert "项目 本判断基于" not in text
    assert "项目 标准制定到实施" not in text
    assert "项目 目前整治行动" not in text
    assert "原文未提及低空经济" in text
    assert "该数据为宏观先行指标" in text
    assert "这一结论基于间接理论推导" in text
    assert "标准制定到实施有周期" in text
    assert "目前整治行动在总体产业层面" in text


def test_render_section_keeps_public_paragraph_when_template_sentence_is_present():
    lines = render_section(
        {
            "section_title": "\u8ba2\u5355\u4e0e\u8bd5\u70b9\u662f\u5426\u5f62\u6210\u5546\u4e1a\u4fe1\u53f7",
            "block_type": "case_comparison",
            "evidence_refs": ["[1]"],
            "used_fact_refs": ["EV-1"],
            "citation_refs": ["[1]"],
            "evidence_backed": True,
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "2025\u5e74\u6c11\u7528\u65e0\u4eba\u673a\u5e02\u573a\u89c4\u6a21\u9884\u8ba1\u540c\u6bd4\u589e\u957f15%\uff0c"
                        "eVTOL\u5e74\u5ea6\u8ba2\u5355\u603b\u989d\u5df2\u8d85\u8fc7300\u4ebf\u5143\u3002"
                        "\u4ece\u884c\u4e1a\u5224\u65ad\u770b\uff0c\u8fd9\u7c7b\u8ba2\u5355\u548c\u8bd5\u70b9\u66f4\u9002\u5408\u4f5c\u4e3a\u89c2\u5bdf\u4ea7\u4e1a\u4f18\u5148\u7ea7\u3002"
                    ),
                }
            ],
        }
    )

    text = "\n".join(lines)
    assert "\u6c11\u7528\u65e0\u4eba\u673a\u5e02\u573a\u89c4\u6a21" in text
    assert "300\u4ebf\u5143" in text
    assert "\u4ece\u884c\u4e1a\u5224\u65ad\u770b" not in text
    assert "[1]" in text


def test_key_data_block_requires_public_citation_and_scans_past_first_row():
    block = _render_key_data_block(
        "\u5173\u952e\u6570\u636e",
        {},
        [
            {
                "should_render": True,
                "headers": ["\u6307\u6807", "\u6570\u503c"],
                "rows": [
                    {"metric": "\u65e0\u5f15\u7528\u6307\u6807", "value": "30\u4e2a"},
                    {"metric": "\u6709\u5f15\u7528\u6307\u6807", "value": "42%", "citation_ref": "[2]"},
                    {"metric": "\u5907\u9009\u6307\u6807", "value": "12%", "citation_refs": ["[3]"]},
                ],
            }
        ],
    )

    assert "\u65e0\u5f15\u7528\u6307\u6807" not in block
    assert "\u6709\u5f15\u7528\u6307\u6807" in block
    assert "[2]" in block


def test_short_cited_section_expands_for_longform_mode(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_RENDERER_TEMPLATE_EXPANSION", "true")
    monkeypatch.setenv("REPORT_RENDER_MIN_SECTION_CHARS", "520")
    lines = render_section(
        {
            "section_title": "\u4ed8\u8d39\u8f6c\u5316\u5728\u54ea\u91cc\u53d1\u751f",
            "block_type": "case_comparison",
            "body_composition_status": "composed",
            "evidence_backed": True,
            "citation_refs": ["[1]"],
            "used_fact_refs": ["EV-1"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "\u4f01\u4e1a\u7ea7 AI Agent \u5df2\u8fdb\u5165\u5ba2\u670d\u5de5\u4f5c\u6d41\u3002",
                }
            ],
        }
    )

    body = "\n".join(line for line in lines if not line.startswith("###"))
    assert len(body.replace(" ", "")) >= 400
    assert "[1]" in body
    assert "\u4ed8\u8d39\u8f6c\u5316" in body or "\u90e8\u7f72" in body


def test_chapter_render_augments_render_blocks_with_reasoning_boundary_and_actionable():
    markdown = render_chapter_package(
        {
            "chapter_title": "Demand validation",
            "sections": [
                {
                    "section_title": "Pilot signal",
                    "claim": "Enterprise pilots show a directional deployment signal.",
                    "reasoning": "The mechanism is that pilots require workflow integration and operating ownership.",
                    "counter_evidence": "The boundary is that one pilot does not prove market-wide adoption.",
                    "actionable": "Track repeat deployments and comparable customer disclosures.",
                    "evidence_refs": ["[1]"],
                    "citation_refs": ["[1]"],
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise pilots show a directional deployment signal.",
                        }
                    ],
                }
            ],
        },
        1,
    )

    assert "Enterprise pilots show a directional deployment signal" in markdown
    assert "workflow integration and operating ownership" in markdown
    assert "one pilot does not prove market-wide adoption" in markdown
    assert "Track repeat deployments" in markdown


def test_chapter_render_skips_fallback_actionable_text():
    markdown = render_chapter_package(
        {
            "chapter_title": "Demand validation",
            "sections": [
                {
                    "section_title": "Pilot signal",
                    "claim": "Enterprise pilots show a directional deployment signal.",
                    "reasoning": "The mechanism is that pilots require workflow integration and operating ownership.",
                    "counter_evidence": "The boundary is that one pilot does not prove market-wide adoption.",
                    "actionable": "Same-period indicators and renewal disclosures determine whether the signal strengthens.",
                    "actionable_is_fallback": True,
                    "evidence_refs": ["[1]"],
                    "citation_refs": ["[1]"],
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise pilots show a directional deployment signal.",
                        }
                    ],
                }
            ],
        },
        1,
    )

    assert "Enterprise pilots show a directional deployment signal" in markdown
    assert "workflow integration and operating ownership" in markdown
    assert "one pilot does not prove market-wide adoption" in markdown
    assert "Same-period indicators" not in markdown


def test_chapter_render_does_not_emit_internal_analysis_suggestions_from_fields():
    markdown = render_chapter_package(
        {
            "chapter_title": "Demand validation",
            "sections": [
                {
                    "section_title": "Pilot signal",
                    "claim": "Enterprise pilots show a directional deployment signal.",
                    "reasoning": (
                        "diagnostic_only score_gap missing_proof_standard "
                        "repair_task_seed search_more must_not_render"
                    ),
                    "counter_evidence": (
                        "review_suggestion public_text_allowed=false "
                        "reanalyze_existing rewrite_with_caveat"
                    ),
                    "actionable": (
                        "source_check semantic_judge executor_should_decide "
                        "\u8865\u8bc1\u5efa\u8bae \u5ba1\u67e5\u5efa\u8bae"
                    ),
                    "evidence_refs": ["[1]"],
                    "citation_refs": ["[1]"],
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise pilots show a directional deployment signal.",
                        }
                    ],
                }
            ],
        },
        1,
    )

    assert "Enterprise pilots show a directional deployment signal" in markdown
    for forbidden in (
        "diagnostic_only",
        "score_gap",
        "missing_proof_standard",
        "repair_task_seed",
        "search_more",
        "must_not_render",
        "review_suggestion",
        "public_text_allowed=false",
        "reanalyze_existing",
        "rewrite_with_caveat",
        "source_check",
        "semantic_judge",
        "executor_should_decide",
        "\u8865\u8bc1\u5efa\u8bae",
        "\u5ba1\u67e5\u5efa\u8bae",
    ):
        assert forbidden not in markdown


def test_chapter_heading_rewrites_internal_evidence_terms():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u54ea\u4e9b\u73af\u8282\u5df2\u6709\u5546\u4e1a\u5316\u8bc1\u636e\uff0c\u54ea\u4e9b\u4ecd\u5904\u4e8e\u6982\u5ff5\u6216\u8bd5\u70b9",
            "sections": [
                {
                    "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "\u4f01\u4e1a\u7ea7 AI Agent \u5df2\u8fdb\u5165\u5ba2\u6237\u6d41\u7a0b\u3002[1]",
                        }
                    ],
                    "citation_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "\u5546\u4e1a\u5316\u8bc1\u636e" not in markdown
    assert "\u5546\u4e1a\u5316\u4fe1\u53f7" in markdown


def test_executive_summary_filters_metric_fragments_and_keeps_public_judgment():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {"judgment": "\u6e17\u900f\u7387\uff1b2023\u5e74"},
                {"judgment": "adoption: 50%"},
                {
                    "judgment": (
                        "\u4f01\u4e1a\u7ea7 AI Agent \u7684\u9700\u6c42\u6b63\u5728\u4ece\u5de5\u5177\u8bd5\u7528"
                        "\u8f6c\u5411\u4e1a\u52a1\u90e8\u7f72\uff0c\u4f46\u4ed8\u8d39\u6df1\u5ea6\u4ecd\u53d6\u51b3\u4e8e ROI "
                        "\u4e0e\u6743\u9650\u6cbb\u7406\u3002"
                    )
                },
            ]
        },
        [],
    )

    assert "\u6e17\u900f\u7387\uff1b2023\u5e74" not in markdown
    assert "adoption: 50%" not in markdown
    assert "\u4f01\u4e1a\u7ea7 AI Agent" in markdown
    assert "\u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba" in markdown


def test_executive_summary_omits_block_when_all_judgments_are_fragments():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {"judgment": "\u6e17\u900f\u7387\uff1b2023\u5e74"},
                {"judgment": "\u5e02\u573a\u89c4\u6a21: 8.2\u4ebf\u5143"},
                {"judgment": "adoption: 50%"},
            ]
        },
        [],
    )

    assert "\u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba" not in markdown
    assert "\u6e17\u900f\u7387" not in markdown
    assert "adoption" not in markdown


def test_finalize_formal_report_removes_stale_invalid_executive_summary_block():
    markdown = finalize_formal_report(
        "# AI Agent\u7814\u7a76\u62a5\u544a\n\n"
        "## \u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba\n"
        "- \u6e17\u900f\u7387\uff1b2023\u5e74\n"
        "\n"
        "## 1. \u9700\u6c42\u9a8c\u8bc1\n"
        "### \u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670\n"
        "\u5df2\u51fa\u73b0\u53ef\u8ffd\u8e2a\u9700\u6c42\u4fe1\u53f7\u3002"
    )

    assert "\u6838\u5fc3\u89c2\u70b9\u4e0e\u4e3b\u8981\u7ed3\u8bba" not in markdown
    assert "\u6e17\u900f\u7387\uff1b2023\u5e74" not in markdown
    assert "\u9700\u6c42\u9a8c\u8bc1" in markdown


def test_metric_claim_without_render_blocks_is_rendered_as_sentence():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u5e02\u573a\u89c4\u6a21",
            "sections": [
                {
                    "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                    "block_type": "metric_reconciliation",
                    "claim": "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143\u3002",
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143" not in markdown
    assert "\u5e02\u573a\u89c4\u6a21" in markdown
    assert "\u8fbe8.2\u4ebf\u5143" in markdown


def test_manifest_citation_replaces_stale_trailing_render_block_citation():
    markdown = render_chapter_package(
        {
            "chapter_title": "Demand validation",
            "sections": [
                {
                    "section_title": "Deployment signal",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "The deployment signal is visible in customer workflow evidence [8].",
                        }
                    ],
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "[1]" in markdown
    assert "[8]" not in markdown


def test_hypothesis_id_section_title_is_not_rendered_as_public_h3():
    markdown = render_chapter_package(
        {
            "chapter_title": "Risk boundary",
            "sections": [
                {
                    "section_title": "H4",
                    "claim": "AI Agent risk boundaries depend on permissions, security, and integration cost.",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "AI Agent risk boundaries depend on permissions, security, and integration cost [1].",
                        }
                    ],
                    "citation_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "### H4" not in markdown
    assert "risk boundaries" in markdown


def test_metric_render_block_is_rewritten_even_when_section_block_is_commercial():
    lines = render_section(
        {
            "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
            "block_type": "unit_economics",
            "evidence_refs": ["EV-DEPLOY"],
            "used_fact_refs": ["EV-DEPLOY"],
            "render_blocks": [{"type": "paragraph", "text": "\u51fa\u8d27/\u90e8\u7f72: \u8d85140\u5bb6"}],
        }
    )

    text = "\n".join(lines)
    assert "\u51fa\u8d27/\u90e8\u7f72: \u8d85140\u5bb6" not in text
    assert "\u76f8\u5173\u8fdb\u5c55" in text
    assert "\u51fa\u8d27/\u90e8\u7f72" not in text
    assert "\u8d85140\u5bb6" in text


def test_final_writer_rewrites_residual_bare_metric_lines():
    output = run_final_writer_agent(
        query="\u4eba\u5f62\u673a\u5668\u4eba",
        report_blueprint={"chapters": [{"chapter_id": "ch_01", "chapter_title": "\u5e02\u573a\u9700\u6c42"}]},
        chapter_packages=[
            {
                "chapter_id": "ch_01",
                "chapter_title": "\u5e02\u573a\u9700\u6c42",
                "sections": [
                    {
                        "section_id": "s_metric",
                        "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                        "claim": "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143\u3002",
                        "evidence_refs": ["EV-METRIC"],
                        "used_fact_refs": ["EV-METRIC"],
                        "supporting_facts": [
                            {
                                "source_ref": "EV-METRIC",
                                "value": "\u8fbe8.2",
                                "unit": "\u4ebf\u5143",
                                "period": "2025\u5e74",
                            }
                        ],
                        "evidence_backed": True,
                    }
                ],
            }
        ],
        source_registry=[
            {
                "ref": "EV-METRIC",
                "evidence_id": "EV-METRIC",
                "title": "\u4eba\u5f62\u673a\u5668\u4eba\u5e02\u573a\u89c4\u6a21\u62a5\u544a",
                "url": "https://www.salesforce.com/news/humanoid-market-size",
                "source_level": "B",
            }
        ],
    )

    markdown = output["report_markdown"]
    assert "\u5e02\u573a\u89c4\u6a21: \u8fbe8.2\u4ebf\u5143" not in markdown
    assert "\u5e02\u573a\u89c4\u6a21\u8fbe8.2\u4ebf\u5143" in markdown


def test_news_snippet_inside_claim_is_removed_before_rendering():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u98ce\u9669\u8fb9\u754c",
            "sections": [
                {
                    "section_title": "\u53cd\u5411\u4fe1\u53f7\u5982\u4f55\u5f71\u54cd\u5224\u65ad",
                    "claim": (
                        "\u98ce\u9669\u4e8b\u5b9e\u7528\u4e8e\u6821\u51c6\u4e50\u89c2\u5224\u65ad\uff1b"
                        "\u4e00\u76c6\u51b7\u6c34\u7ec8\u4e8e\u6cfc\u5230\u4e86\u706b\u70ed\u7684\u5177\u8eab\u667a\u80fd\u8d5b\u9053"
                    ),
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "\u4e00\u76c6\u51b7\u6c34" not in markdown


def test_headline_like_claims_are_removed_before_rendering():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u5546\u4e1a\u5316\u8fdb\u5c55",
            "sections": [
                {
                    "section_title": "\u5546\u4e1a\u5316\u4fe1\u53f7\u662f\u5426\u6e05\u6670",
                    "claim": "AI Agent\u5546\u4e1a\u5316\u843d\u5730\uff0c3.3\u4e07\u4ebf\u8d5b\u9053\u52a0\u901f\u7206\u53d1\u3002",
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                },
                {
                    "section_title": "\u5ba2\u6237\u843d\u5730",
                    "claim": "\u6784\u5efa\u4f01\u4e1a\u7ea7 AI Agent \u8d22\u62a5\u5206\u6790\u6d41\u6c34\u7ebf",
                    "evidence_refs": ["E2"],
                    "evidence_backed": True,
                },
            ],
            "table_packages": [],
        },
        1,
    )

    assert "3.3\u4e07\u4ebf\u8d5b\u9053\u52a0\u901f\u7206\u53d1" not in markdown
    assert "\u6784\u5efa\u4f01\u4e1a\u7ea7 AI Agent \u8d22\u62a5\u5206\u6790\u6d41\u6c34\u7ebf" not in markdown


def test_empty_chapter_package_is_not_rendered_as_h2_shell():
    markdown = render_chapter_package(
        {
            "chapter_title": "\u6ca1\u6709\u8bc1\u636e\u7684\u7ae0\u8282",
            "sections": [],
            "table_packages": [],
            "chapter_omitted_no_evidence": True,
        },
        1,
    )

    assert markdown == ""


def test_render_chapter_package_avoids_cross_chapter_section_title_duplicates():
    seen_sections = set()
    first = render_chapter_package(
        {
            "chapter_title": "\u4f1a\u8ba1\u5c31\u4e1a\u7684\u533a\u57df\u4fe1\u53f7",
            "sections": [
                {
                    "section_title": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1",
                    "claim": "\u533a\u57df\u516c\u5171\u5c31\u4e1a\u6570\u636e\u53ef\u4ee5\u8bf4\u660e\u4eba\u529b\u9700\u6c42\u7684\u57fa\u672c\u80cc\u666f\u3002",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "\u533a\u57df\u516c\u5171\u5c31\u4e1a\u6570\u636e\u53ef\u4ee5\u8bf4\u660e\u4eba\u529b\u9700\u6c42\u7684\u57fa\u672c\u80cc\u666f\u3002",
                        }
                    ],
                }
            ],
        },
        1,
        seen_section_titles_global=seen_sections,
    )
    second = render_chapter_package(
        {
            "chapter_title": "\u4f1a\u8ba1\u5c97\u4f4d\u7684\u80fd\u529b\u7ed3\u6784",
            "sections": [
                {
                    "section_title": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1",
                    "claim": "\u8bfe\u7a0b\u8bad\u7ec3\u548c\u5c97\u4f4d\u8981\u6c42\u7684\u540c\u65f6\u53d8\u5316\u6b63\u5728\u63a8\u52a8\u4f1a\u8ba1\u80fd\u529b\u7ed3\u6784\u8c03\u6574\u3002",
                    "citation_refs": ["[2]"],
                    "evidence_refs": ["EV-2"],
                    "evidence_backed": True,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "\u8bfe\u7a0b\u8bad\u7ec3\u548c\u5c97\u4f4d\u8981\u6c42\u7684\u540c\u65f6\u53d8\u5316\u6b63\u5728\u63a8\u52a8\u4f1a\u8ba1\u80fd\u529b\u7ed3\u6784\u8c03\u6574\u3002",
                        }
                    ],
                }
            ],
        },
        2,
        seen_section_titles_global=seen_sections,
    )
    combined = first + "\n" + second

    assert combined.count("### \u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1") == 1
    assert "### \u8bfe\u7a0b\u8bad\u7ec3\u548c\u5c97\u4f4d\u8981\u6c42\u7684\u540c\u65f6\u53d8\u5316" in second


def test_duplicate_section_title_uses_claim_tail_instead_of_numeric_suffix():
    seen_sections = {"\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1"}
    markdown = render_chapter_package(
        {
            "chapter_title": "\u4eba\u624d\u9700\u6c42\u4e0e\u5c97\u4f4d\u7ed3\u6784",
            "sections": [
                {
                    "section_title": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1",
                    "dynamic_section_title": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1",
                    "claim": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1\u5e94\u805a\u7126\u516c\u5171\u5c31\u4e1a\u670d\u52a1\u673a\u6784\u7684\u5b9e\u9645\u62db\u8058\u53e3\u5f84\uff0c\u907f\u514d\u5c06\u5b8f\u89c2\u53d9\u4e8b\u76f4\u63a5\u7b49\u540c\u4e8e\u4e13\u4e1a\u5c97\u4f4d\u9700\u6c42\u3002",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1\u5e94\u805a\u7126\u516c\u5171\u5c31\u4e1a\u670d\u52a1\u673a\u6784\u7684\u5b9e\u9645\u62db\u8058\u53e3\u5f84\uff0c\u907f\u514d\u5c06\u5b8f\u89c2\u53d9\u4e8b\u76f4\u63a5\u7b49\u540c\u4e8e\u4e13\u4e1a\u5c97\u4f4d\u9700\u6c42\u3002",
                        }
                    ],
                }
            ],
        },
        2,
        seen_section_titles_global=seen_sections,
    )

    assert "### 2" not in markdown
    assert "### \u516c\u5171\u5c31\u4e1a\u670d\u52a1\u673a\u6784\u7684\u5b9e\u9645\u62db\u8058\u6570\u636e" in markdown


def test_duplicate_claim_derived_section_title_uses_claim_tail_when_public_title_is_empty():
    seen_sections = {"\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1"}
    markdown = render_chapter_package(
        {
            "chapter_title": "\u4eba\u624d\u9700\u6c42\u4e0e\u5c97\u4f4d\u7ed3\u6784",
            "sections": [
                {
                    "section_title": "\u5177\u4f53\u573a\u666f\u662f\u5426\u5df2\u7ecf\u6210\u7acb",
                    "claim": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1\u5e94\u805a\u7126\u516c\u5171\u5c31\u4e1a\u670d\u52a1\u673a\u6784\u7684\u5b9e\u9645\u62db\u8058\u53e3\u5f84\uff0c\u907f\u514d\u5c06\u5b8f\u89c2\u53d9\u4e8b\u76f4\u63a5\u7b49\u540c\u4e8e\u4e13\u4e1a\u5c97\u4f4d\u9700\u6c42\u3002",
                    "citation_refs": ["[1]"],
                    "evidence_refs": ["EV-1"],
                    "evidence_backed": True,
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "\u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1\u5e94\u805a\u7126\u516c\u5171\u5c31\u4e1a\u670d\u52a1\u673a\u6784\u7684\u5b9e\u9645\u62db\u8058\u53e3\u5f84\uff0c\u907f\u514d\u5c06\u5b8f\u89c2\u53d9\u4e8b\u76f4\u63a5\u7b49\u540c\u4e8e\u4e13\u4e1a\u5c97\u4f4d\u9700\u6c42\u3002",
                        }
                    ],
                }
            ],
        },
        2,
        seen_section_titles_global=seen_sections,
    )

    assert "### \u5c31\u4e1a\u5e02\u573a\u9a8c\u8bc1" not in markdown
    assert "### \u516c\u5171\u5c31\u4e1a\u670d\u52a1\u673a\u6784\u7684\u5b9e\u9645\u62db\u8058\u6570\u636e" in markdown


def test_chapter_heading_is_compacted_before_rendering():
    markdown = render_chapter_package(
        {
            "chapter_title": (
                "\u4eba\u5f62\u673a\u5668\u4eba\u4ece\u201c\u8868\u6f14\u578b\u79d1\u6280\u201d\u8fdb\u5165"
                "\u201c\u771f\u5b9e\u573a\u666f\u9a8c\u8bc1 + \u8d44\u672c\u5b9a\u4ef7\u201d\u9636\u6bb5\u4e86\u5417"
                "\u662f\u5426\u5b58\u5728\u771f\u5b9e\u9700\u6c42\u548c\u5e02\u573a\u7a7a\u95f4"
            ),
            "sections": [
                {
                    "section_title": "\u9700\u6c42\u9a8c\u8bc1",
                    "claim": "\u5df2\u51fa\u73b0\u53ef\u8ffd\u8e2a\u9700\u6c42\u4fe1\u53f7\u3002",
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    first_line = markdown.splitlines()[0]
    assert len(first_line) <= 36
    assert "\u4eba\u5f62\u673a\u5668\u4eba\u4ece" not in first_line


def test_repeated_section_titles_are_rewritten_with_claim_context():
    markdown = render_chapter_package(
        {
            "chapter_title": "Workflow adoption",
            "sections": [
                {
                    "section_title": "Market signal",
                    "claim": "Enterprise workflows show repeatable AI agent deployment.",
                    "render_blocks": [{"type": "paragraph", "text": "Enterprise workflows show repeatable AI agent deployment."}],
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                },
                {
                    "section_title": "Market signal",
                    "claim": "Government procurement shows public-sector demand.",
                    "render_blocks": [{"type": "paragraph", "text": "Government procurement shows public-sector demand."}],
                    "evidence_refs": ["E2"],
                    "evidence_backed": True,
                },
            ],
            "table_packages": [],
        },
        1,
    )

    assert markdown.count("### Market signal") == 1
    assert "### Government procurement" in markdown


def test_chapter_lead_drops_source_title_snippet():
    snippet = (
        "AI 时代，唯一确定的是数据｜爱分析访谈 - 电子工程专辑"
        "（2026-05-21T00:00:00+08:00）：数据是穿越周期的壁垒，"
        "以下为本次访谈实录。"
    )
    markdown = render_chapter_package(
        {
            "chapter_title": "Workflow adoption",
            "lead": snippet,
            "sections": [
                {
                    "section_title": "Workflow demand",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise workflows show repeatable AI agent deployment.",
                        }
                    ],
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "电子工程专辑" not in markdown
    assert "以下为本次访谈实录" not in markdown
    assert "Enterprise workflows show repeatable AI agent deployment." in markdown


def test_chapter_lead_gets_first_section_citation_when_rendered():
    markdown = render_chapter_package(
        {
            "chapter_title": "Workflow adoption",
            "lead": "本章判断企业智能体是否已经进入流程部署，并观察样本能否支撑付费转化。",
            "sections": [
                {
                    "section_title": "Workflow demand",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "Enterprise workflows show repeatable AI agent deployment.",
                        }
                    ],
                    "citation_refs": ["[3]"],
                    "evidence_refs": ["E1"],
                    "evidence_backed": True,
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "这一部分观察企业智能体是否已经进入流程部署，并观察样本能否支撑付费转化。[3]" in markdown


def test_executive_summary_omits_factual_bullets_without_public_citation():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {
                    "label": "机会判断",
                    "judgment": "AI Agent在A股上市公司中已形成广泛关注，近3800家公司在年报中提及AI相关内容。",
                },
                {
                    "label": "机会判断",
                    "judgment": "AI Agent已经进入公共部门部署。",
                    "citation_refs": ["[2]"],
                },
            ]
        },
        [],
    )

    assert "近3800家公司" not in markdown
    assert "AI Agent已经进入公共部门部署。[2]" in markdown


def test_executive_summary_omits_market_and_ipo_claims_without_citation():
    markdown = render_executive_summary(
        {
            "core_judgments": [
                {
                    "label": "机会判断",
                    "judgment": "资本市场对AI Agent相关企业给予高估值，反映投资者对市场空间的乐观预期。",
                },
                {
                    "label": "机会判断",
                    "judgment": "AI Agent生态中已出现冲刺IPO的独角兽企业，表明资本市场对工业AI智能体领域的关注。",
                },
                {
                    "label": "机会判断",
                    "judgment": "AI Agent生态中已出现可追踪部署样本，说明企业智能体正在从试点走向流程部署。",
                    "citation_refs": ["[4]"],
                },
            ]
        },
        [],
    )

    assert "高估值" not in markdown
    assert "独角兽" not in markdown
    assert "AI Agent生态中已出现可追踪部署样本，说明企业智能体正在从试点走向流程部署。[4]" in markdown


def test_render_appendix_keeps_public_sources_and_omits_diagnostic_tables():
    rendered = render_appendix(
        [{"ref": "[1]", "title": "来源A", "url": "https://example.org/a"}],
        {
            "metric_normalization_table": [
                {
                    "metric_name": "CAGR",
                    "subject": "AI Agent",
                    "scope": "全球",
                    "period": "2028年",
                    "unit": "%",
                    "value": "41%",
                    "source_level": "B",
                }
            ],
            "table_appendix_rows": [
                {
                    "title": "市场指标与口径表",
                    "headers": ["指标", "范围", "期间", "数值", "单位", "后续影响"],
                    "rows": [
                        [
                            "CAGR",
                            "全球",
                            "2028年",
                            "41",
                            "%",
                            "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。",
                        ],
                        [
                            "市场规模",
                            "全球",
                            "2028年",
                            "100",
                            "亿美元",
                            "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。",
                        ],
                    ],
                    "should_render": True,
                    "validation_status": "passed",
                }
            ],
        },
    )

    assert rendered.count("## 来源附录") == 1
    assert "- [1]" in rendered
    assert "指标口径表" not in rendered
    assert "附录明细" not in rendered
    assert "后续影响" not in rendered
    assert "该指标须" not in rendered


def test_render_appendix_keeps_public_table_rows_without_source_registry():
    rendered = render_appendix(
        [],
        {
            "table_appendix_rows": [
                {
                    "title": "Key metric appendix",
                    "headers": ["metric", "scope", "value"],
                    "rows": [
                        ["adoption", "enterprise", "42%"],
                        ["deployment", "pilot", "18%"],
                    ],
                    "should_render": True,
                    "validation_status": "passed",
                }
            ],
        },
    )

    assert rendered
    assert "Key metric appendix" in rendered
    assert "| metric | scope | value |" in rendered


def test_sanitize_public_markdown_removes_analysis_scaffold_language():
    markdown = (
        "# AI Agent研究报告\n\n"
        "## 1. 需求验证\n"
        "### ch_01\n"
        "### 关键事实与判断依据\n"
        "本章判断应以“市场规模: 100亿美元”为事实锚点。\n"
        "先用 增速: 44.2% 确认本章的事实起点。\n"
        "后续重点跟踪同口径指标、反向样本和执行进展。\n"
        "可复核材料指向：某行业报告显示市场规模增长。[1]\n"
        "这些事实来自不同类型来源且方向一致时，可以支撑较强结论。\n"
        "来源集中、口径不一致或缺少反向样本时，结论会保留边界。\n\n"
        "这张表显示，成本的表内信号是成本 / 62.5%。[1]\n"
        "后续影响：若表内信号继续被高等级来源验证，可纳入章节分析。[1]\n"
        "使用边界：表格优先使用已绑定到本章的正文证据。[1]\n\n"
        "本章需要按连续指标和反向样本拆解，避免把单点信号直接外推为行业结论。\n"
        "这一判断目前更适合作为背景条件，结论强度取决于后续连续指标和相反样本的变化。\n"
        "### 空标题\n"
        "### 被清空标题\n"
        "后续重点跟踪同口径指标。\n"
        "## 来源附录\n"
        "- [1] 来源A | https://example.org/a\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    for phrase in [
        "事实锚点",
        "事实起点",
        "后续重点跟踪",
        "可复核材料指向",
        "这些事实来自不同类型来源",
        "来源集中、口径不一致",
        "这张表显示",
        "后续影响",
        "使用边界",
        "需要按连续指标",
        "避免把单点信号直接外推",
        "更适合作为背景条件",
        "结论强度取决",
        "### ch_01",
        "关键事实与判断依据",
        "### 空标题",
        "### 被清空标题",
    ]:
        assert phrase not in cleaned
    assert "## 来源附录" in cleaned


def test_sanitize_public_markdown_rewrites_review_style_directional_language():
    markdown = (
        "# 行业研究报告\n\n"
        "## 1. 商业化进展\n"
        "这些信号仍受来源覆盖范围和公开披露充分性的限制，应作为方向性观察进入正文。[1]\n"
        "后续应继续观察同类主体、同类场景和相同口径信息是否重复出现。[1]\n"
        "这些材料解释了产业链变化，以及后续判断需要继续观察哪些约束条件，而不是被简单处理成孤立材料。[1]\n\n"
        "## 数据来源\n"
        "- [1] 行业公开资料 | https://example.org/source\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    for phrase in (
        "方向性观察进入正文",
        "方向性观察",
        "后续应继续观察",
        "后续判断需要继续观察",
        "被简单处理成孤立材料",
    ):
        assert phrase not in cleaned
    assert "实际进展" in cleaned
    # Methodology-narration prose (主体行动是否持续 / 影响路径是否更清晰) is now stripped
    # from the body, not rewritten into milder framework language.
    assert "主体行动是否持续" not in cleaned
    assert "影响路径是否" not in cleaned
    assert "资源投入" in cleaned or "具体业务" in cleaned or "实际进展" in cleaned
    assert "相关主体、组织安排和后续决策" not in cleaned


def test_sanitize_public_markdown_repairs_count_metrics_mislabeled_as_cost():
    markdown = (
        "## 风险边界\n"
        "成本方面，相关企业数量已超140家，高参与成本可能制约商业化进程。\n"
        "市场指标校准方面，成本超过140家，为需求空间提供信号。\n"
        "成本维度上，参与企业超过140家。\n"
        "平台覆盖企业超过140家（成本: 超140家）。\n"
        "短期内可能增加企业的合规成本。\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "成本方面，相关企业数量" not in cleaned
    assert "成本超过140家" not in cleaned
    assert "成本维度上，参与企业" not in cleaned
    assert "（成本: 超140家）" not in cleaned
    assert "参与主体方面，相关企业数量已超140家" in cleaned
    assert "参与企业超过140家" in cleaned
    assert "合规成本" in cleaned


def test_sanitize_public_markdown_removes_live_run_dirty_fragments_but_keeps_case_fact():
    markdown = (
        "# AI Agent企业落地研究报告\n\n"
        "## 3. 商业化信号\n"
        "AI Agent在物业管理领域已实现规模化商业落地，通过部署AI物业经理智能体，"
        "已落地超过300个项目，管理面积超2000万平方米，帮助客户降低管理成本60-70%，"
        "提升运营效率3-5倍。 "
        "（二）AI 物业经理智能体建设单位的成本在2000为70%，这一指标用于判断市场空间和兑现节奏。 "
        "（二）AI 物业经理智能体建设单位的成本在2000为5倍，这一指标用于判断市场空间和兑现节奏。 "
        "转化愿景为现实：AI在采购中的应用场景 引言 "
        "这个反向样本提示风险边界仍可能改变结论强度，需要把商业化判断限制在已验证场景内。[1]\n\n"
        "## 来源附录\n"
        "- [1] 广州人工智能典型案例 | https://example.org/case\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "已落地超过300个项目" in cleaned
    assert "降低管理成本60-70%" in cleaned
    assert "提升运营效率3-5倍" in cleaned
    assert "成本在2000为70%" not in cleaned
    assert "成本在2000为5倍" not in cleaned
    assert "转化愿景为现实" not in cleaned
    assert "应用场景 引言" not in cleaned
    assert "反向样本提示" not in cleaned


def test_sanitize_public_markdown_removes_named_institution_claim_when_appendix_lacks_source():
    markdown = (
        "# AI Agent企业落地研究报告\n\n"
        "## 1. 市场空间\n"
        "### Gartner的AI炒作周期显示生成式AI正处...\n"
        "根据Gartner的AI炒作周期，生成式AI当前处于期望顶峰，后续可能进入幻灭期。[1]\n\n"
        "## 来源附录\n"
        "- [1] AI Agents Use Cases in Enterprise | https://example.org/use-cases\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "Gartner" not in cleaned
    assert "炒作周期" not in cleaned
    assert "AI Agents Use Cases" in cleaned


def test_sanitize_public_markdown_keeps_named_institution_claim_when_appendix_has_source():
    markdown = (
        "# AI Agent企业落地研究报告\n\n"
        "## 1. 市场空间\n"
        "根据Gartner的AI炒作周期，生成式AI当前处于期望顶峰。[1]\n\n"
        "## 来源附录\n"
        "- [1] Gartner Hype Cycle for Artificial Intelligence | https://www.gartner.com/example\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    assert "Gartner" in cleaned
    assert "炒作周期" in cleaned


def test_sanitize_public_markdown_removes_evidence_repair_signals_from_public_body():
    markdown = (
        "# AI Agent研究报告\n\n"
        "## 1. 商业化验证\n"
        "商业化证据主要集中在金融、采购、政企等少数行业，其他行业缺乏明确案例；多数证据为2025-2026年报告，时效性有限；来源多为B级或C级，可靠性中等。\n\n"
        "可公开事实显示，采购系统中的智能体部署已经进入供应商、合同和订单管理流程。[1]\n\n"
        "## 来源附录\n"
        "- [1] 来源A | https://example.org/a\n"
    )

    cleaned = sanitize_public_markdown(markdown, mode="enforce")

    for phrase in [
        "商业化证据主要集中",
        "其他行业缺乏明确案例",
        "多数证据为2025-2026年报告",
        "时效性有限",
        "来源多为B级或C级",
        "可靠性中等",
    ]:
        assert phrase not in cleaned
    assert "采购系统中的智能体部署" in cleaned
    assert "## 来源附录" in cleaned


def test_render_table_package_drops_public_diagnostic_columns():
    markdown = render_table_package(
        {
            "should_render": True,
            "title": "竞争格局对照表",
            "headers": ["Metric", "Value", "Competitive signal", "Risk boundary"],
            "rows": [
                {
                    "cells": ["客户部署", "2个案例", "Competitive signal from vendor", "Risk boundary needs review"],
                    "evidence_refs": ["[1]"],
                },
                {
                    "cells": ["产品发布", "3项能力", "Competitive signal from product", "Risk boundary needs review"],
                    "evidence_refs": ["[2]"],
                },
            ],
            "evidence_refs": ["[1]", "[2]"],
        }
    )

    assert markdown
    assert "Competitive signal" not in markdown
    assert "Risk boundary" not in markdown
    assert "客户部署" in markdown
    assert "产品发布" in markdown


def test_render_section_does_not_fill_short_body_with_public_templates(monkeypatch):
    from rag_pipeline.agents.markdown_renderer import render_section

    monkeypatch.setenv("REPORT_RENDER_MIN_SECTION_CHARS", "900")
    lines = render_section(
        {
            "section_id": "s1",
            "section_title": "Workflow deployment signal",
            "block_type": "case_comparison",
            "evidence_backed": True,
            "citation_refs": ["[1]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "Enterprise workflows show repeatable AI agent deployment.",
                }
            ],
        }
    )

    body = "\n".join(lines)
    assert "Enterprise workflows show repeatable AI agent deployment" in body
    assert len(body) < 300


def test_render_table_package_keeps_investment_diagnostic_table_score_only():
    markdown = render_table_package(
        {
            "should_render": True,
            "title": "投资优先级矩阵",
            "table_type": "investment_priority_table",
            "headers": ["对象", "评分", "存疑", "raw URL"],
            "rows": [
                {
                    "cells": ["Vendor A", "85", "来源仍需核验", "https://example.invalid/raw"],
                    "evidence_refs": ["[1]"],
                },
                {
                    "cells": ["Vendor B", "72", "口径不一致", "https://example.invalid/raw2"],
                    "evidence_refs": ["[2]"],
                },
            ],
            "evidence_refs": ["[1]", "[2]"],
        }
    )

    assert markdown == ""
