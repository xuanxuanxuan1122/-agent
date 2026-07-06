from rag_pipeline.agents.final_writer_agent import should_render_chapter
from rag_pipeline.agents.markdown_renderer import render_chapter_package, render_section


def test_render_section_filters_public_boundary_notes_from_render_blocks():
    lines = render_section(
        {
            "section_title": "岗位需求变化",
            "evidence_backed": True,
            "citation_refs": ["[7]", "[8]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "2026年春招周报显示AI智能体相关职位数同比增速达455%，杭州日报等官方渠道同步确认该数据。[7][8]",
                },
                {
                    "type": "paragraph",
                    "text": "边界在于目前关于岗位能力变化的公开样本仍有限，结论需要保留弹性。[7][8]",
                },
                {
                    "type": "paragraph",
                    "text": "后续应继续观察同类主体、同类场景和相同口径信息是否重复出现。[7][8]",
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "同比增速达455%" in markdown
    assert "边界在于" not in markdown
    assert "结论需要保留弹性" not in markdown
    assert "后续应继续观察" not in markdown


def test_render_chapter_package_filters_counter_scope_notes_in_legacy_sections():
    markdown = render_chapter_package(
        {
            "chapter_title": "岗位能力要求变化",
            "sections": [
                {
                    "section_title": "企业招聘能力要求提高",
                    "evidence_backed": True,
                    "citation_refs": ["[1]", "[2]"],
                    "claim": "企业招聘材料显示，会计相关岗位正在增加数据处理、业务系统协同和智能工具应用要求。[1][2]",
                    "reasoning": "招聘侧的变化说明，会计人才要求正在从单一核算能力扩展到数据分析和系统协作能力。[1][2]",
                    "counter_evidence": "数据主要反映春招趋势与部分企业高管预期，未覆盖全行业长期编制变化，适用于判断短期结构性调整方向。[1][2]",
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "会计相关岗位" in markdown
    assert "数据分析和系统协作能力" in markdown
    assert "未覆盖全行业长期编制变化" not in markdown
    assert "适用于判断短期结构性调整方向" not in markdown


def test_render_chapter_package_repairs_incomplete_section_headings():
    markdown = render_chapter_package(
        {
            "chapter_title": "教育培养调整",
            "sections": [
                {
                    "section_title": "地方政府与专业会计师协会正",
                    "evidence_backed": True,
                    "citation_refs": ["[10]"],
                    "claim": "地方政府与专业会计师协会正通过顶层行动方案与行业共识平台，推动会计人才AI能力标准重塑与推广渠道扩展。[10]",
                    "render_blocks": [
                        {
                            "type": "paragraph",
                            "text": "地方政府与专业会计师协会正通过顶层行动方案与行业共识平台，推动会计人才AI能力标准重塑与推广渠道扩展。[10]",
                        }
                    ],
                }
            ],
            "table_packages": [],
        },
        1,
    )

    assert "### 地方政府与专业会计师协会正\n" not in markdown
    assert "地方政府与专业会计师协会" in markdown
    assert "推动" in markdown


def test_render_section_filters_mechanism_scaffold_sentences_but_keeps_facts():
    lines = render_section(
        {
            "section_title": "区域政策信号",
            "evidence_backed": True,
            "citation_refs": ["[13]", "[19]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "静安区发布2026—2028年人工智能创新应用示范区三年行动方案，明确区域级AI技术落地与产业应用规划。"
                        "这类风险信号会影响岗位能力变化，也会改变相关结论的适用范围。"
                        "风险事实会直接影响机会判断，帮助区分已经验证的进展和仍需谨慎对待的假设。[13][19]"
                    ),
                },
                {
                    "type": "paragraph",
                    "text": (
                        "AI技术向泛财经领域的渗透已催生可量化的岗位需求增量。"
                        "岗位能力变化的行业含义在于，它会改变AI技术向泛财经领域的渗透已催生可量化的岗位需求的任务边界、能力组合和评价标准。"
                        "只有这些变化进入连续招聘、课程方案、采购项目或业务流程，才会从单点信号变成可比较的趋势。[19][24]"
                    ),
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "静安区发布2026—2028年人工智能创新应用示范区三年行动方案" in markdown
    assert "AI技术向泛财经领域的渗透已催生可量化的岗位需求增量" in markdown
    assert "这类风险信号会影响" not in markdown
    assert "风险事实会直接影响机会判断" not in markdown
    assert "行业含义在于" not in markdown
    assert "只有这些变化进入" not in markdown


def test_render_section_filters_scope_refinement_sentences():
    lines = render_section(
        {
            "section_title": "区域招聘口径",
            "evidence_backed": True,
            "citation_refs": ["[19]", "[24]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "无锡市人力资源市场运行数据显示，2025年二季度供求总量进入下行通道。[19]",
                },
                {
                    "type": "paragraph",
                    "text": "数据口径限定于无锡市公共就业服务机构，反映区域劳动力市场趋势，外推至全国需考虑产业结构差异与地方经济周期波动。[19]",
                },
                {
                    "type": "paragraph",
                    "text": "数据聚焦于AI智能体大类岗位，未单独剥离纯会计类岗位占比，需结合财务垂直招聘数据进一步细化。[19][24]",
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "供求总量进入下行通道" in markdown
    assert "外推至全国需考虑" not in markdown
    assert "需结合财务垂直招聘数据进一步细化" not in markdown


def test_render_section_filters_mechanism_template_scaffold():
    lines = render_section(
        {
            "section_title": "技术成熟度变化",
            "evidence_backed": True,
            "citation_refs": ["[4]", "[5]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "科技巨头资本支出突破1550亿美元并预计超4700亿美元，宏观研究指出AI技术革命正重构资本与劳动比例。"
                        "技术成熟度的实际影响取决于可靠性、成本、适用场景和执行条件能否同时成立。"
                        "技术能力本身只是入口，稳定运行、责任划分和系统兼容才决定实际影响。"
                        "相关投入只有转化为稳定系统、可控权限、明确成本结构和可复核业务结果，才会真正改变应用边界。"
                        "从技术扩散看，技术成熟度会提高对可靠性、成本可控性和系统接入能力的要求。[4][5]"
                    ),
                }
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "资本支出突破1550亿美元" in markdown
    assert "实际影响取决于" not in markdown
    assert "技术能力本身只是入口" not in markdown
    assert "相关投入只有转化为" not in markdown
    assert "从技术扩散看" not in markdown


def test_render_section_filters_observation_followup_scaffold():
    lines = render_section(
        {
            "section_title": "教育培养端响应",
            "evidence_backed": True,
            "citation_refs": ["[2]", "[14]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "教育培养端已启动实质性响应，高校与市场化培训机构通过跨界课程融合与短期专项培训，加速孵化适配AI场景的复合型财务人才。"
                        "相关变化已经出现具体动作，可继续观察课程结构变化是否进入实际执行。"
                        "课程结构变化之所以重要，是因为它已经落到教育培养端的具体动作或业务安排上。"
                        "如果类似动作持续增加，这一判断就会从个别披露走向更明确的产业趋势；如果样本停留在少数主体，结论仍需保持审慎。[2][14]"
                    ),
                },
                {
                    "type": "paragraph",
                    "text": "无锡市人力资源市场运行数据显示，2025年二季度供求总量进入下行通道，这一事实用于校准需求变化的范围、时间和可比性。这些条件越清楚，越能判断相关变化是否具备持续性。[4]",
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "教育培养端已启动实质性响应" in markdown
    assert "供求总量进入下行通道" in markdown
    assert "相关变化已经出现具体动作" not in markdown
    assert "可继续观察" not in markdown
    assert "之所以重要" not in markdown
    assert "如果类似动作持续增加" not in markdown
    assert "如果样本停留" not in markdown
    assert "这一事实用于校准" not in markdown
    assert "这些条件越清楚" not in markdown


def test_final_writer_does_not_render_limitations_chapter_as_public_body():
    assert not should_render_chapter(
        {
            "chapter_title": "限制与待验证问题",
            "writing_mode": "limitations",
            "sections": [
                {
                    "section_title": "岗位需求口径",
                    "claim": "智联招聘2026年春招周报显示AI智能体相关职位数同比增速达455%。[4][5]",
                    "citation_refs": ["[4]", "[5]"],
                    "evidence_backed": True,
                }
            ],
        }
    )

    assert should_render_chapter(
        {
            "chapter_title": "风险与约束",
            "writing_mode": "core_chapter",
            "sections": [
                {
                    "section_title": "合规风险",
                    "claim": "AI工具进入财务流程后，权限治理和审计留痕要求会提高。[1]",
                    "citation_refs": ["[1]"],
                    "evidence_backed": True,
                }
            ],
        }
    )


def test_render_section_filters_public_caveat_tail_notes():
    lines = render_section(
        {
            "section_title": "教育供给调整",
            "evidence_backed": True,
            "citation_refs": ["[2]", "[14]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "高校与市场化培训机构通过跨界课程融合与短期专项培训，加速孵化适配AI场景的复合型财务人才。[2][14]",
                },
                {
                    "type": "paragraph",
                    "text": "案例集中于个别高校与春招培训动态，尚未形成全国统一的标准化培养体系，适用于观察教育供给侧的早期调整趋势。[2][14]",
                },
                {
                    "type": "paragraph",
                    "text": "平台工具的商业化程度与教育场景的适配性仍在演进，零代码/低代码方案在复杂财务合规与深度业财融合场景中的应用存在局限。[17][18][19]",
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "复合型财务人才" in markdown
    assert "案例集中于个别高校" not in markdown
    assert "适用于观察" not in markdown
    assert "仍在演进" not in markdown
    assert "存在局限" not in markdown


def test_render_section_filters_live_boundary_note_variants():
    lines = render_section(
        {
            "section_title": "会计岗位变化",
            "evidence_backed": True,
            "citation_refs": ["[7]", "[8]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "AI智能体与自动化工具使单人服务户数提升10倍以上，2026年春招中AI智能体相关职位数同比激增455%。[7][8]",
                },
                {
                    "type": "paragraph",
                    "text": "人效数据基于特定代账场景测算，招聘增速反映短期春招窗口期热度，长期岗位稳定性与薪酬结构需结合后续季度数据观察。[7][8]",
                },
                {
                    "type": "paragraph",
                    "text": "结论基于行业调研与技术趋势信号，具体替代比例因企业数字化成熟度与业务复杂度而异。[1][5][6]",
                },
                {
                    "type": "paragraph",
                    "text": "证据聚焦于地方示范区宏观规划，未直接披露会计岗位的具体AI渗透率或就业替代数据，结论适用于判断区域政策对专业服务数智化的催化作用。[13][23]",
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "单人服务户数提升10倍以上" in markdown
    assert "同比激增455%" in markdown
    assert "人效数据基于特定代账场景测算" not in markdown
    assert "需结合后续季度数据观察" not in markdown
    assert "结论基于行业调研" not in markdown
    assert "具体替代比例因" not in markdown
    assert "证据聚焦于地方示范区" not in markdown
    assert "结论适用于判断" not in markdown


def test_render_section_filters_repeated_case_scope_notes_from_live_report():
    lines = render_section(
        {
            "section_title": "教育培养调整",
            "evidence_backed": True,
            "citation_refs": ["[7]", "[15]", "[16]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "会计高等教育领域的智能化转型目前仍高度集中于头部院校，课程改革与跨学科融合的规模化实施面临显著资源门槛。[7][15][16]",
                },
                {
                    "type": "paragraph",
                    "text": "案例集中于头部或中外合作高校，地方普通院校的课程迭代速度、师资储备与实训平台投入存在显著梯度差异。[7][15][16]",
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "智能化转型目前仍高度集中于头部院校" in markdown
    assert "案例集中于头部或中外合作高校" not in markdown
    assert "存在显著梯度差异" not in markdown


def test_renderer_template_expansion_does_not_emit_internal_writing_instructions(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_RENDERER_TEMPLATE_EXPANSION", "true")
    monkeypatch.setenv("REPORT_RENDER_MIN_SECTION_CHARS", "900")

    lines = render_section(
        {
            "section_title": "岗位能力迁移",
            "block_type": "integrated_signal",
            "evidence_backed": True,
            "citation_refs": ["[4]", "[5]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": "企业财务团队正在增加AI工具使用、数据分析与流程优化相关能力要求。[4][5]",
                }
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "企业财务团队正在增加AI工具使用" in markdown
    assert "较稳妥的写法" not in markdown
    assert "不应被过早放大" not in markdown
    assert "边界清楚" not in markdown
    assert "事实能够支撑的最低结论" not in markdown


def test_render_section_filters_embedded_source_scope_sentences_but_keeps_analysis():
    lines = render_section(
        {
            "section_title": "政策驱动与能力迁移",
            "evidence_backed": True,
            "citation_refs": ["[8]", "[13]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "财政部及地方财政部门正通过典型案例征集与十五五规划部署，引导会计职能向数智化核算、管理会计与可持续信息披露转型。"
                        "监管层通过国家级案例征集与地方五年规划联动，将AI与大数据应用纳入会计行业标准与职能拓展路径。"
                        "政策导向明确，但具体落地规模、财政配套资金及企业端合规成本尚未量化，结论适用于政策驱动型市场阶段。[8]"
                    ),
                },
                {
                    "type": "paragraph",
                    "text": (
                        "头部会计师事务所将信息化投入维持在营收5%以上的高比例，表明技术升级已从概念探讨进入企业级预算与常态化运营阶段。"
                        "投入数据仅反映头部机构策略，中小事务所的跟进节奏与资金约束未在本证据中体现，结论适用于行业头部及政策导向明确的细分领域。[8][13]"
                    ),
                },
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "典型案例征集与十五五规划部署" in markdown
    assert "企业级预算与常态化运营阶段" in markdown
    assert "政策导向明确" not in markdown
    assert "结论适用于政策驱动型市场阶段" not in markdown
    assert "投入数据仅反映头部机构策略" not in markdown
    assert "未在本证据中体现" not in markdown


def test_render_section_filters_cross_industry_scope_note_sentence():
    lines = render_section(
        {
            "section_title": "标准体系演进",
            "evidence_backed": True,
            "citation_refs": ["[8]"],
            "render_blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "制造业通过中试标准与质量评估规范将新技术从实验室推向规模化生产，这一成熟度模型同样适用于专业服务领域。"
                        "证据来源于制造业标准文件，作为跨行业类比信号使用，不直接代表会计行业标准现状，适用于推演专业能力培养的标准化演进阶段。[8]"
                    ),
                }
            ],
        }
    )

    markdown = "\n".join(lines)

    assert "制造业通过中试标准" in markdown
    assert "成熟度模型同样适用于专业服务领域" in markdown
    assert "证据来源于制造业标准文件" not in markdown
    assert "不直接代表会计行业标准现状" not in markdown
    assert "适用于推演专业能力培养" not in markdown
