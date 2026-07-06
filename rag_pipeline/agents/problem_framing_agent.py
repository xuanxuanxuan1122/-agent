from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .article_brief import extract_research_subject, is_broad_ai_subject, normalize_article_brief


AGENT_NAME = "problem_framing_agent"
AGENT_DESCRIPTION = "Problem Framing Agent. Converts a user question into testable hypotheses before search."


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _compact(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _decision_context(query: str) -> str:
    if re.search(r"投资|买入|卖出|估值|进入|布局|机会|赛道|值得", query):
        return "行业判断 / 投资筛选 / 产品方向选择"
    if re.search(r"产品|研发|客户|订单|商业化|量产", query):
        return "产品方向选择 / 商业化验证"
    return "行业判断 / 机会筛选"


def _looks_like_ev_material_market(query: str) -> bool:
    text = str(query or "")
    vehicle_or_battery = re.search(r"新能源汽车|新能源车|电动汽车|动力电池|电池|汽车", text)
    material_scope = re.search(r"材料|新材料|新型材料|电池材料|正极|负极|隔膜|电解液|轻量化", text)
    if not (vehicle_or_battery and material_scope):
        return False
    if _looks_like_us_china_policy_multisector(text):
        return False
    return bool(
        re.search(r"行情|市场|价格|订单|产能|毛利|需求|确定性|线索", text)
        or material_scope
    )


def _looks_like_semiconductor_supply_chain(query: str) -> bool:
    return bool(
        re.search(r"半导体|芯片|集成电路|晶圆|光刻|EDA|封测|先进制程|成熟制程|供应链|科技博弈", query)
        and re.search(r"中美|美国|中国|全球|供应链|重构|管制|制裁|国产|机遇|挑战", query)
    )


def _looks_like_us_china_policy_multisector(query: str) -> bool:
    text = str(query or "")
    if not re.search(r"中美|美国.*中国|中国.*美国|US-China", text, re.I):
        return False
    policy_hit = re.search(r"关税|出口管制|市场准入|制裁|贸易壁垒|科技管制|供应链重构|脱钩|再定位|战略竞争", text)
    sector_hits = sum(
        1
        for pattern in [
            r"半导体|芯片|集成电路|先进制程|封测",
            r"新能源|电动车|新能源汽车|光伏|储能|电池",
            r"消费品|零售|品牌|制造业",
            r"互联网|平台|云|数据|数字服务",
        ]
        if re.search(pattern, text)
    )
    return bool(policy_hit and sector_hits >= 2)


HIGH_STAKES_RE = re.compile(
    r"投资|尽调|并购|IPO|估值|买入|卖出|市场进入|进入|布局|值得|优先级|回报|投资价值|"
    r"investment|investor|due diligence|market entry|m&a|valuation|ipo",
    re.I,
)


def _decision_context(query: str) -> str:
    if HIGH_STAKES_RE.search(str(query or "")):
        return "investment_or_market_entry"
    if re.search(r"产品|研发|客户|订单|商业化|量产|product|customer|order|commercial", str(query or ""), re.I):
        return "product_commercialization"
    return "general_research"


def _requires_strong_proof(decision_use: str) -> bool:
    value = str(decision_use or "")
    return value in {"investment_or_market_entry", "company_due_diligence", "investment_memo", "market_entry"} or bool(HIGH_STAKES_RE.search(value))


def _coverage_requirements(decision_context: str) -> Dict[str, Any]:
    if _requires_strong_proof(decision_context):
        return {
            "min_A_or_B_sources": 2,
            "min_counter_sources": 1,
            "min_metric_sources": 1,
            "min_case_sources": 1,
            "source_diversity": [],
        }
    if str(decision_context or "") == "product_commercialization":
        return {
            "min_A_or_B_sources": 1,
            "min_counter_sources": 0,
            "min_metric_sources": 1,
            "min_case_sources": 1,
            "source_diversity": [],
        }
    return {
        "min_A_or_B_sources": 1,
        "min_counter_sources": 0,
        "min_metric_sources": 1,
        "min_case_sources": 0,
        "source_diversity": [],
    }


def _research_subject(query: str) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    text = re.sub(r"^(请问|帮我看看|帮我分析|分析一下|现在|当前|目前)\s*", "", text)
    text = re.sub(r"(企业行研|行业研究|行研|深度研究|研究)?(报告|文档)$", "", text).strip()
    text = re.sub(r"(怎么样|如何|有哪些|怎么看)[？?]?$", "", text).strip()
    text = re.sub(r"(焦虑与机遇|机遇与挑战)$", "", text).strip()
    specific = extract_research_subject(text)
    if specific and not is_broad_ai_subject(specific):
        return specific
    if re.search(r"(中国|国内).*(AI|人工智能)|(?:AI|人工智能).*(中国|国内)", text, re.I):
        return "中国人工智能行业"
    if re.search(r"\bAI\b|人工智能|大模型|生成式", text, re.I) and re.search(r"行业|产业|市场|赛道", text):
        return "人工智能行业"
    return text or str(query or "").strip()


def _looks_like_ai_industry(query: str) -> bool:
    subject = _research_subject(query)
    if subject and not is_broad_ai_subject(subject):
        return False
    return bool(
        re.search(r"AI|人工智能|大模型|生成式AI|AIGC", query, re.I)
        and re.search(r"中国|国内|行业|产业|市场|焦虑|机遇|发展|竞争|应用", query)
    )


ACADEMIC_OR_PROFESSIONAL_FIELDS = {
    "会计学",
    "审计学",
    "财务管理",
    "经济学",
    "金融学",
    "法学",
    "管理学",
    "教育学",
    "心理学",
    "统计学",
    "计算机科学",
    "数据科学",
}


def _looks_like_academic_or_professional_field(query: str) -> bool:
    text = re.sub(r"\s+", "", str(query or "")).strip()
    if not text:
        return False
    if text in ACADEMIC_OR_PROFESSIONAL_FIELDS:
        return True
    if re.search(r"行业|产业|市场|公司|产品|商业化|投资|供应链|机会|风险|竞争格局|客户|订单|采购|上市公司", text):
        return False
    if len(text) <= 12 and re.search(r"(?:学|学科|专业)$", text):
        return True
    return bool(len(text) <= 16 and re.search(r"学科|专业|课程|就业|职业|证书|考研", text))


def _bundle(
    *,
    metric_terms: List[str],
    case_terms: List[str],
    counter_terms: List[str],
    filing_terms: Optional[List[str]] = None,
    expert_terms: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    return {
        "metric": metric_terms,
        "case": case_terms,
        "filing": filing_terms or ["年报", "公告", "招股书", "交易所", "投资者关系"],
        "counter": counter_terms,
        "source_check": ["官方", "协会", "白皮书", "券商研究", "产业研究"],
        "expert": expert_terms or ["券商研报", "协会判断", "产业研究", "专家观点"],
    }


def _hypothesis(
    idx: int,
    *,
    claim: str,
    must_prove: List[str],
    must_disprove: List[str],
    bundle: Dict[str, List[str]],
    decision_use: str,
) -> Dict[str, Any]:
    hypothesis_id = f"H{idx}"
    strong_proof = _requires_strong_proof(decision_use)
    return {
        "hypothesis_id": hypothesis_id,
        "id": hypothesis_id,
        "statement": claim,
        "hypothesis_statement": claim,
        "claim_to_test": claim,
        "decision_use": decision_use,
        "proof_standard": "strong" if strong_proof else "medium",
        "counter_evidence_required": bool(strong_proof),
        "required_source_levels": ["A", "B"],
        "required_sources": ["official/filing", "research/association"],
        "required_evidence_types": ["metric", "source_check", "expert", "counter"] if strong_proof else ["metric", "source_check", "expert"],
        "must_prove": must_prove,
        "must_disprove": must_disprove,
        "evidence_bundle": bundle,
        "minimum_evidence_bundle": "1个A/B来源 + 可用指标或方向性信号；反证和案例按问题风险显式追加",
        "metric_definitions": [
            {"metric_name": item, "scope": "", "period": "", "unit": ""}
            for item in must_prove
            if re.search(r"规模|价格|毛利|产能|销量|渗透率|利用率|增速|订单", item)
        ],
        "falsification_triggers": [
            "A/B来源不支持",
            "反证显示产能过剩或价格下行",
            "客户认证或量产订单缺失",
            "指标缺少scope/period/unit",
            "替代路线削弱需求确定性",
        ],
    }


def _ev_material_hypotheses(query: str, decision_use: str) -> List[Dict[str, Any]]:
    subject = "新能源汽车新型材料"
    return [
        _hypothesis(
            1,
            claim=f"{subject}中，电池功能材料比轻量化/结构材料更具短期放量确定性",
            must_prove=["需求增速", "客户认证", "量产订单", "产能利用率", "价格/毛利"],
            must_disprove=["产能过剩", "替代技术路线", "订单不及预期", "价格下跌"],
            bundle=_bundle(
                metric_terms=["市场规模", "出货量", "渗透率", "价格", "毛利率", "产能利用率"],
                case_terms=["动力电池厂", "整车厂", "客户认证", "定点", "量产供货", "订单"],
                counter_terms=["产能过剩", "价格战", "替代路线", "订单取消", "毛利下滑"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            2,
            claim=f"{subject}的市场行情只有在价格、产能、订单和毛利同时改善时才可判断为向好",
            must_prove=["价格趋势", "产能扩张与利用率", "订单/合同", "毛利率", "库存变化"],
            must_disprove=["价格下行", "库存累积", "开工率不足", "客户延期"],
            bundle=_bundle(
                metric_terms=["价格", "毛利率", "产能", "开工率", "库存", "订单金额"],
                case_terms=["公告订单", "客户定点", "批量供货", "长协合同"],
                counter_terms=["价格下跌", "库存增加", "开工率下降", "产能过剩", "应收账款上升"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            3,
            claim=f"{subject}中已有客户认证和量产记录的材料环节，优先级高于仍停留在概念或试点的环节",
            must_prove=["客户认证", "量产车型/电池包", "批量供货", "复购/长协", "收入兑现"],
            must_disprove=["仅实验室验证", "试点未转量产", "客户未披露", "收入占比低"],
            bundle=_bundle(
                metric_terms=["收入占比", "订单金额", "供货量", "客户数量"],
                case_terms=["车型应用", "电池包应用", "客户认证", "定点公告", "量产"],
                counter_terms=["试点停滞", "认证周期延长", "客户未采购", "商业化收入不足"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            4,
            claim=f"{subject}里热门但缺少A/B来源闭环的方向，只能作为线索观察，不能进入强结论",
            must_prove=["官方/公告/年报来源", "协会或券商交叉验证", "可比指标", "反证检查"],
            must_disprove=["仅媒体热度", "单企业宣传", "无可比口径", "缺少反证"],
            bundle=_bundle(
                metric_terms=["市场规模", "价格", "产能", "毛利率", "渗透率"],
                case_terms=["公告案例", "年报披露客户", "招股书客户", "协会案例"],
                counter_terms=["媒体炒作", "概念验证", "无法量产", "成本过高", "路线被替代"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            5,
            claim=f"{subject}的进入/投资价值应按需求确定性、客户验证、财务质量和反证强度分层",
            must_prove=["需求确定性", "客户验证", "财务质量", "竞争格局", "反证强度"],
            must_disprove=["高竞争低毛利", "客户集中", "技术替代", "产能扩张过快"],
            bundle=_bundle(
                metric_terms=["需求增速", "毛利率", "收入占比", "客户集中度", "产能增速"],
                case_terms=["客户结构", "订单结构", "长协", "量产项目"],
                counter_terms=["客户集中风险", "毛利下滑", "同质化竞争", "替代技术", "价格战"],
            ),
            decision_use=decision_use,
        ),
    ]


def _semiconductor_supply_chain_hypotheses(query: str, decision_use: str) -> List[Dict[str, Any]]:
    return [
        _hypothesis(
            1,
            claim="全球半导体供应链正在从效率优先转向安全优先、区域化和友岸化分工",
            must_prove=["出口管制强度", "产业补贴与本土建厂", "跨境设备/材料限制", "供应链区域迁移", "关键节点集中度"],
            must_disprove=["限制措施明显放松", "全球化分工成本优势重新占优", "关键产能未发生区域迁移", "企业仍以单一效率目标配置产能"],
            bundle=_bundle(
                metric_terms=["出口管制清单", "CHIPS Act补贴", "晶圆厂资本开支", "设备出口许可", "区域产能份额", "供应链集中度"],
                case_terms=["美国CHIPS Act", "日本半导体政策", "欧盟芯片法案", "韩国晶圆厂", "台积电海外建厂", "荷兰光刻设备出口管制"],
                counter_terms=["管制豁免", "跨境供应恢复", "海外建厂延期", "补贴落地不足", "成本压力导致回流放缓"],
                filing_terms=["政府公告", "出口管制规则", "企业年报", "投资者关系", "海关/贸易数据"],
                expert_terms=["产业协会", "券商研报", "智库研究", "半导体产业研究", "政策解读"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            2,
            claim="先进制程、光刻设备、EDA和关键材料仍是中国芯片产业的上限约束",
            must_prove=["先进制程差距", "EUV/DUV设备可得性", "EDA工具限制", "高端材料和零部件国产化率", "高端GPU/AI芯片供给约束"],
            must_disprove=["关键设备稳定获得", "先进制程良率快速追平", "国产EDA完成高端闭环", "核心材料不再受外部限制"],
            bundle=_bundle(
                metric_terms=["制程节点", "良率", "国产化率", "设备交付", "EDA覆盖率", "材料验证周期", "高端芯片出口许可"],
                case_terms=["光刻机", "刻蚀设备", "薄膜沉积", "EDA软件", "先进制程晶圆代工", "AI芯片限制"],
                counter_terms=["禁令升级", "设备维护受限", "良率不足", "生态软件不兼容", "材料认证周期拉长"],
                filing_terms=["出口管制文件", "企业公告", "年报", "招股书", "政府文件"],
                expert_terms=["半导体设备研究", "EDA产业研究", "材料产业研究", "技术路线报告"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            3,
            claim="成熟制程、封测、功率/模拟/车规芯片和国产替代是阶段性机会主线",
            must_prove=["成熟制程需求", "封测订单", "功率/模拟芯片国产替代", "车规认证", "本土客户导入"],
            must_disprove=["成熟制程产能过剩", "价格下行侵蚀盈利", "客户验证慢于预期", "海外替代供应恢复", "同质化竞争加剧"],
            bundle=_bundle(
                metric_terms=["成熟制程产能", "产能利用率", "封测收入", "国产替代率", "车规认证数量", "毛利率", "订单周期"],
                case_terms=["晶圆代工", "封装测试", "功率半导体", "模拟芯片", "汽车电子", "工业控制", "本土客户导入"],
                counter_terms=["产能过剩", "价格战", "库存上升", "客户认证失败", "海外供应降价"],
                filing_terms=["企业年报", "公告订单", "招股书", "投资者关系", "行业协会数据"],
                expert_terms=["券商研报", "产业链调研", "半导体协会", "车规芯片研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            4,
            claim="先进封装、Chiplet、RISC-V和设计生态提供增量突破，但不能完全替代先进制程能力",
            must_prove=["先进封装产能", "Chiplet生态", "RISC-V应用", "设计工具链适配", "客户量产案例"],
            must_disprove=["封装瓶颈未解决", "生态兼容性不足", "高端场景性能差距扩大", "缺少量产客户", "成本不具备优势"],
            bundle=_bundle(
                metric_terms=["先进封装产能", "封装良率", "Chiplet项目数量", "RISC-V出货/应用", "设计工具覆盖率", "量产客户数"],
                case_terms=["2.5D/3D封装", "Chiplet", "RISC-V", "AI加速芯片", "国产设计工具链", "服务器/车载应用"],
                counter_terms=["生态碎片化", "接口标准不统一", "性能不足", "成本过高", "客户未量产"],
                filing_terms=["企业公告", "产品发布", "技术白皮书", "年报", "标准组织资料"],
                expert_terms=["先进封装研究", "Chiplet生态研究", "RISC-V产业研究", "设计工具链研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            5,
            claim="资本开支周期、贸易壁垒、产能错配和全球客户信任重建构成主要挑战",
            must_prove=["资本开支强度", "产能建设节奏", "贸易限制", "客户验证周期", "盈利质量", "政策执行稳定性"],
            must_disprove=["产能利用率持续改善", "海外客户恢复采购", "贸易限制边际缓和", "高端环节实现稳定替代", "盈利质量同步改善"],
            bundle=_bundle(
                metric_terms=["资本开支", "折旧压力", "产能利用率", "库存", "毛利率", "出口数据", "客户集中度"],
                case_terms=["晶圆厂扩建", "设备采购", "海外客户认证", "本土供应链导入", "关税/出口限制", "并购与投资"],
                counter_terms=["产能闲置", "需求下修", "价格下行", "现金流压力", "贸易壁垒升级", "客户流失"],
                filing_terms=["年报", "公告", "投资计划", "财报电话会", "政府规则"],
                expert_terms=["周期研究", "财务质量分析", "国际贸易研究", "供应链安全研究"],
            ),
            decision_use=decision_use,
        ),
    ]


def _us_china_policy_multisector_hypotheses(query: str, decision_use: str) -> List[Dict[str, Any]]:
    return [
        _hypothesis(
            1,
            claim="中美关税、出口管制与市场准入规则正在把行业分化从需求周期问题改写为政策约束与供应链位置问题",
            must_prove=["关税变化", "出口管制清单", "市场准入限制", "供应链迁移", "企业收入/订单影响"],
            must_disprove=["政策边际放松", "企业通过转口或本地化完全对冲", "需求恢复抵消政策冲击"],
            bundle=_bundle(
                metric_terms=["关税税率", "出口许可", "市场准入规则", "贸易额", "产能迁移", "订单变化"],
                case_terms=["半导体", "新能源", "消费品", "互联网", "跨国企业供应链调整"],
                counter_terms=["豁免清单", "政策缓和", "替代市场增长", "本地化成功案例"],
                filing_terms=["政策原文", "商务部/海关/监管文件", "企业年报", "公告", "财报电话会"],
                expert_terms=["产业研究", "贸易政策研究", "券商研报", "智库报告", "行业协会"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            2,
            claim="半导体承压最集中在先进制程、设备、EDA和高端材料，但成熟制程、封测、国产替代和先进封装可能形成阶段性受益",
            must_prove=["出口管制范围", "先进制程差距", "设备/EDA可得性", "成熟制程需求", "国产替代订单"],
            must_disprove=["关键设备稳定获得", "先进制程瓶颈缓解", "成熟制程产能过剩加剧", "客户验证不及预期"],
            bundle=_bundle(
                metric_terms=["出口许可", "制程节点", "国产化率", "封测收入", "设备交付", "订单周期"],
                case_terms=["光刻设备", "EDA", "先进封装", "成熟制程", "车规/工业芯片"],
                counter_terms=["制裁升级", "良率不足", "产能闲置", "客户流失"],
                filing_terms=["出口管制文件", "企业公告", "年报", "招股书", "投资者关系"],
                expert_terms=["半导体产业研究", "设备材料研究", "供应链安全研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            3,
            claim="新能源产业的压力主要来自关税和海外市场准入，受益端则取决于非美市场扩张、本地化产能和关键材料成本曲线",
            must_prove=["海外关税", "本地化产能", "非美市场销量", "材料成本", "利润率"],
            must_disprove=["关税豁免", "海外需求下滑", "材料价格反向上涨", "本地化成本高于预期"],
            bundle=_bundle(
                metric_terms=["关税税率", "出口销量", "海外产能", "电池成本", "毛利率", "市占率"],
                case_terms=["动力电池", "电动车出口", "储能", "光伏", "海外工厂"],
                counter_terms=["贸易壁垒升级", "需求放缓", "价格战", "产能过剩"],
                filing_terms=["海关数据", "企业年报", "公告", "政策文件", "财报电话会"],
                expert_terms=["新能源产业研究", "贸易政策研究", "材料成本研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            4,
            claim="消费品和互联网的影响更偏市场准入、品牌渠道、数据合规和本地监管，分化会体现在区域收入、获客成本和合规成本上",
            must_prove=["区域收入变化", "渠道准入", "数据合规要求", "获客成本", "监管处罚/限制"],
            must_disprove=["消费需求恢复", "渠道替代成功", "合规成本下降", "监管边际缓和"],
            bundle=_bundle(
                metric_terms=["区域收入", "GMV/用户数", "获客成本", "合规成本", "市场份额", "毛利率"],
                case_terms=["品牌出海", "跨境电商", "互联网平台", "云服务", "数据合规"],
                counter_terms=["需求下滑", "监管处罚", "渠道收缩", "本地竞争加剧"],
                filing_terms=["企业年报", "财报电话会", "监管文件", "平台公告"],
                expert_terms=["消费品研究", "互联网政策研究", "跨境电商研究", "数据合规研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            5,
            claim="最终受益者不是单一行业，而是能同时满足政策合规、供应链替代、非美市场扩张和利润质量的企业与环节",
            must_prove=["政策合规能力", "供应链替代能力", "非美市场增长", "利润/现金流质量", "客户验证"],
            must_disprove=["单一市场依赖", "利润率恶化", "现金流承压", "客户验证不足", "反向政策风险"],
            bundle=_bundle(
                metric_terms=["海外收入占比", "毛利率", "现金流", "客户集中度", "订单金额", "产能利用率"],
                case_terms=["本地化生产", "替代供应商", "新客户认证", "区域市场扩张"],
                counter_terms=["客户流失", "成本上升", "政策反转", "订单取消", "产能错配"],
                filing_terms=["企业公告", "年报", "投资者关系", "海关/贸易数据", "监管文件"],
                expert_terms=["产业链研究", "投资策略研究", "风险评估", "贸易政策研究"],
            ),
            decision_use=decision_use,
        ),
    ]


def _ai_industry_hypotheses(query: str, decision_use: str) -> List[Dict[str, Any]]:
    subject = "中国人工智能行业" if re.search(r"中国|国内", query) else "人工智能行业"
    return [
        _hypothesis(
            1,
            claim=f"{subject}仍有产业规模和应用扩张机会，但增长质量需要按细分场景验证",
            must_prove=["产业规模", "市场规模", "增速", "企业数量", "投融资金额"],
            must_disprove=["融资降温", "估值下调", "需求放缓", "企业退出"],
            bundle=_bundle(
                metric_terms=["人工智能产业规模", "AI市场规模", "产业增速", "企业数量", "投融资金额"],
                case_terms=["大模型应用", "行业应用", "客户案例", "采购项目", "落地案例"],
                counter_terms=["融资降温", "估值下调", "需求放缓", "商业化不及预期"],
                filing_terms=["工信部", "信通院", "统计局", "企业年报", "招股书"],
                expert_terms=["人工智能产业研究", "AI行业报告", "券商研报", "协会白皮书"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            2,
            claim=f"{subject}的主要焦虑来自算力、模型成本、数据合规和安全治理约束",
            must_prove=["算力成本", "GPU供给", "模型训练成本", "数据合规", "AI安全"],
            must_disprove=["算力成本下降", "国产算力替代", "合规成本下降", "安全事故减少"],
            bundle=_bundle(
                metric_terms=["算力成本", "GPU供给", "国产算力", "模型训练成本", "能耗"],
                case_terms=["智算中心", "云厂商", "大模型备案", "企业AI部署", "安全事件"],
                counter_terms=["算力过剩", "价格下降", "国产替代加速", "监管放松", "安全风险下降"],
                filing_terms=["政策文件", "监管规则", "企业公告", "云服务财报", "招股书"],
                expert_terms=["算力产业研究", "大模型成本研究", "AI治理研究", "数据合规研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            3,
            claim=f"{subject}机会兑现的关键在实际应用深度、可复制执行能力和持续使用结果",
            must_prove=["实际应用", "可复制执行", "持续使用", "效果反馈", "ROI"],
            must_disprove=["试点停滞", "预算不足", "执行效果不及预期", "持续使用不足"],
            bundle=_bundle(
                metric_terms=["付费率", "渗透率", "采购金额", "续费率", "ROI"],
                case_terms=["金融AI", "制造AI", "政务AI", "医疗AI", "企业采购", "标杆客户"],
                counter_terms=["试点未转量产", "客户预算收缩", "应用效果不佳", "替代方案"],
                filing_terms=["中标公告", "客户案例", "企业年报", "合同公告", "招股书"],
                expert_terms=["企业AI应用研究", "行业数字化报告", "咨询报告", "券商研报"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            4,
            claim=f"{subject}的竞争格局会从模型参数竞争转向算力、数据、场景和生态竞争",
            must_prove=["模型能力", "算力资源", "数据资源", "生态伙伴", "市场份额"],
            must_disprove=["开源模型压缩差距", "价格战", "同质化竞争", "客户迁移"],
            bundle=_bundle(
                metric_terms=["模型榜单", "市场份额", "活跃用户", "调用量", "价格"],
                case_terms=["大模型厂商", "云服务商", "开源模型", "生态合作", "行业解决方案"],
                counter_terms=["价格战", "同质化", "开源替代", "客户流失", "监管处罚"],
                filing_terms=["企业公告", "产品发布", "财报", "开发者生态", "招股书"],
                expert_terms=["大模型竞争格局", "AI云服务研究", "开源模型研究", "产业生态研究"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            5,
            claim=f"{subject}需要把政策支持、国际限制和技术路线变化一起纳入机会与风险判断",
            must_prove=["政策支持", "监管规则", "出口管制", "国产替代", "技术路线"],
            must_disprove=["政策执行不及预期", "出口限制升级", "技术路线替代", "合规成本上升"],
            bundle=_bundle(
                metric_terms=["政策资金", "备案数量", "国产算力占比", "出口限制", "研发投入"],
                case_terms=["政策试点", "大模型备案", "国产芯片适配", "政企项目", "国际限制"],
                counter_terms=["出口管制升级", "监管收紧", "技术替代", "合规成本上升", "安全事件"],
                filing_terms=["政策原文", "监管公告", "商务部", "工信部", "网信办"],
                expert_terms=["AI政策研究", "AI治理白皮书", "科技竞争研究", "产业政策解读"],
            ),
            decision_use=decision_use,
        ),
    ]


def _generic_hypotheses(query: str, decision_use: str) -> List[Dict[str, Any]]:
    subject = _research_subject(query)
    return [
        _hypothesis(
            1,
            claim=f"{subject}是否存在真实需求和可验证市场空间，而不是概念热度",
            must_prove=["市场规模", "需求增速", "明确使用主体", "可复核案例"],
            must_disprove=["需求不可持续", "只停留在早期样本", "缺少预算或资源投入"],
            bundle=_bundle(
                metric_terms=["市场规模", "增速", "渗透率", "活跃用户", "使用规模"],
                case_terms=["使用案例", "应用场景", "业务动作", "可复核样本"],
                counter_terms=["失败案例", "需求放缓", "预算收缩", "项目取消"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            2,
            claim=f"{subject}的竞争格局由哪些玩家、能力、成本和渠道变量决定",
            must_prove=["主要玩家", "产品/服务差异", "价格或成本", "客户获取", "生态合作"],
            must_disprove=["同质化竞争", "价格战", "替代方案", "客户迁移"],
            bundle=_bundle(
                metric_terms=["市场份额", "价格", "成本", "用户/客户规模", "调用量/出货量"],
                case_terms=["产品发布", "客户案例", "生态合作", "渠道合作"],
                counter_terms=["同质化", "价格战", "替代路线", "客户流失"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            3,
            claim=f"{subject}哪些环节已有实际进展，哪些仍处于概念或早期样本",
            must_prove=["实际收入或使用结果", "主体验证", "可复制执行", "留存或持续使用"],
            must_disprove=["仅概念宣传", "早期样本未扩张", "结果未披露"],
            bundle=_bundle(
                metric_terms=["收入", "使用数量", "主体数量", "留存", "执行成本"],
                case_terms=["使用案例", "持续使用案例", "可复制执行", "合同"],
                counter_terms=["试点停滞", "使用不足", "收入披露不足"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            4,
            claim=f"{subject}的技术、供应、监管或替代约束会如何改变机会排序",
            must_prove=["关键瓶颈", "供应/基础设施约束", "监管/合规", "替代路线"],
            must_disprove=["瓶颈缓解", "监管放松", "替代方案成熟", "成本快速下降"],
            bundle=_bundle(
                metric_terms=["成本", "供给", "效率", "合规成本", "研发投入"],
                case_terms=["技术路线", "基础设施", "监管案例", "替代方案"],
                counter_terms=["替代技术", "供给过剩", "监管风险", "需求不及预期"],
            ),
            decision_use=decision_use,
        ),
    ]


def _academic_or_professional_field_hypotheses(query: str, decision_use: str) -> List[Dict[str, Any]]:
    subject = _research_subject(query)
    return [
        _hypothesis(
            1,
            claim=f"{subject}的现实价值应从人才需求、岗位结构和组织财务治理需求验证，而不是套用产业赛道框架",
            must_prove=["就业需求", "岗位招聘", "薪酬或就业质量", "用人单位需求", "人才培养规模"],
            must_disprove=["岗位饱和", "基础岗位被自动化替代", "就业质量下降", "专业培养与岗位需求错配"],
            bundle=_bundle(
                metric_terms=["招生规模", "就业率", "岗位数量", "薪酬", "证书通过率"],
                case_terms=["高校培养方案", "招聘岗位", "事务所岗位", "企业财务岗位", "职业案例"],
                counter_terms=["岗位饱和", "低端岗位自动化替代", "就业质量下降", "技能错配"],
                filing_terms=["教育部", "人社部", "财政部", "高校培养方案", "职业标准"],
                expert_terms=["高校课程体系", "职业教育研究", "CPA协会", "会计人才报告", "就业质量报告"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            2,
            claim=f"{subject}的知识体系需要区分课程基础、专业能力、实务训练和证书要求",
            must_prove=["课程体系", "能力要求", "职业标准", "证书要求", "实务技能"],
            must_disprove=["课程与实务脱节", "证书价值下降", "基础能力被工具替代", "培养目标模糊"],
            bundle=_bundle(
                metric_terms=["课程学分", "专业建设点", "证书报考人数", "考试通过率", "实习实践比例"],
                case_terms=["培养方案", "核心课程", "实训项目", "职业资格", "高校专业介绍"],
                counter_terms=["课程同质化", "实务训练不足", "证书通过率下降", "岗位能力错配"],
                filing_terms=["本科专业目录", "高校培养方案", "会计专业教学标准", "职业资格公告"],
                expert_terms=["教育评估报告", "专业建设研究", "会计教育论文", "人才培养白皮书"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            3,
            claim=f"{subject}的职业路径正在被财务共享、数智化工具和AI改造",
            must_prove=["财务共享", "会计信息化", "电子凭证", "AI财务", "数据分析能力"],
            must_disprove=["基础核算岗位减少", "工具替代人工", "技能升级不足", "数字化投入不足"],
            bundle=_bundle(
                metric_terms=["财务共享中心数量", "电子凭证覆盖", "数字化岗位需求", "AI财务应用", "岗位技能要求"],
                case_terms=["财务共享案例", "电子凭证试点", "数智财务岗位", "企业财务数字化", "AI会计工具"],
                counter_terms=["基础岗位收缩", "自动化替代", "系统落地失败", "合规风险", "技能错配"],
                filing_terms=["电子凭证政策", "财政部通知", "企业数字化案例", "行业协会报告"],
                expert_terms=["数智财务研究", "会计信息化报告", "财务共享白皮书", "AI财务观察"],
            ),
            decision_use=decision_use,
        ),
        _hypothesis(
            4,
            claim=f"{subject}的风险边界来自监管变化、职业伦理、证书门槛和实践经验要求",
            must_prove=["会计准则", "审计监管", "内控合规", "职业伦理", "证书门槛"],
            must_disprove=["监管趋严导致责任上升", "伦理违规案例", "证书门槛提高", "实践经验不足"],
            bundle=_bundle(
                metric_terms=["监管处罚数量", "准则修订", "证书报考人数", "审计处罚", "内控缺陷"],
                case_terms=["监管案例", "审计处罚", "职业伦理案例", "内控整改", "证书要求"],
                counter_terms=["合规成本上升", "审计失败", "职业责任风险", "证书内卷", "经验门槛提高"],
                filing_terms=["财政部", "证监会", "审计署", "会计准则", "监管处罚公告"],
                expert_terms=["会计监管研究", "审计质量报告", "职业伦理研究", "内控合规报告"],
            ),
            decision_use=decision_use,
        ),
    ]


def run_problem_framing_agent(
    *,
    query: str,
    llm_config: Optional[Dict[str, Any]] = None,
    article_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del llm_config
    brief = normalize_article_brief(article_brief, fallback_query=query) if article_brief else {}
    query = str(brief.get("planning_query") or query or "").strip()
    decision_context = _decision_context(query)
    research_object = _research_subject(query)
    if _looks_like_us_china_policy_multisector(query):
        core_question = "中美关税、出口管制与市场准入新格局下，半导体、新能源、消费品与互联网分别承受什么压力，哪些环节可能反而受益？"
        hypotheses = _us_china_policy_multisector_hypotheses(query, decision_context)
    elif _looks_like_semiconductor_supply_chain(query):
        core_question = "中美科技博弈如何重塑全球半导体供应链，中国芯片产业的机会、短板与风险分别在哪里？"
        hypotheses = _semiconductor_supply_chain_hypotheses(query, decision_context)
    elif _looks_like_ev_material_market(query):
        core_question = "新能源汽车新材料当前市场行情是否向好，哪些环节最确定，哪些仍只是线索？"
        hypotheses = _ev_material_hypotheses(query, decision_context)
    elif _looks_like_ai_industry(query):
        core_question = f"{research_object}的机会、焦虑来源和可兑现路径分别是什么，哪些判断能被公开证据支撑？"
        hypotheses = _ai_industry_hypotheses(query, decision_context)
    elif _looks_like_academic_or_professional_field(query):
        core_question = f"{research_object}应从学科知识体系、人才需求、职业路径和数字化/监管变化哪些维度展开，哪些判断能被公开数据和案例支撑？"
        hypotheses = _academic_or_professional_field_hypotheses(query, decision_context)
    else:
        core_question = f"{research_object}当前是否具备可验证的市场机会，哪些判断能被证据证明？"
        hypotheses = _generic_hypotheses(query, decision_context)
    return {
        "agent": AGENT_NAME,
        "core_question": core_question,
        "research_object": research_object,
        "decision_context": decision_context,
        "research_domain": "academic_or_professional_field" if _looks_like_academic_or_professional_field(query) else "general_market_research",
        "article_brief": brief,
        "planning_query": query,
        "article_direction": str(brief.get("direction") or ""),
        "topic_anchor_terms": _topic_anchor_terms(query),
        "hypotheses": hypotheses,
        "coverage_requirements": {
            "per_hypothesis": _coverage_requirements(decision_context)
        },
        "rules": {
            "chapters_come_from_hypotheses": True,
            "core_claim_requires_A_or_B": True,
            "c_level_is_directional_signal": True,
            "single_evidence_cannot_be_claim": True,
            "reject_generic_industry_fallback": True,
        },
    }


ROLE_SPECS = [
    ("support", "正向证据", ["official_data", "market_research"], "official_data"),
    ("metric", "指标口径", ["official_data", "market_research"], "market_data"),
    ("case", "案例验证", ["customer_case", "news_event"], "case"),
    ("counter", "反向证据", ["news_event", "market_research"], "counter"),
    ("filing", "财务/公告", ["filing_company"], "filing"),
    ("source_check", "来源交叉验证", ["official_data", "market_research"], "source_check"),
]

SOURCE_PRIORITY_BY_ROLE = {
    "support": ["官方", "协会", "白皮书", "券商研报", "行业报告"],
    "metric": ["统计", "产业规模", "市场规模", "协会", "信通院"],
    "case": ["客户案例", "采购", "中标", "订单", "企业公告"],
    "counter": ["风险", "下滑", "失败案例", "监管", "负面"],
    "filing": ["年报", "公告", "招股书", "投资者关系", "财报"],
    "source_check": ["官方", "原文", "政策文件", "协会", "白皮书"],
}


_GENERIC_PLANNER_TITLE_RE = re.compile(
    r"^(?:"
    r"市场规模(?:与|和)?增长?|竞争格局|政策(?:与|和)?监管(?:环境)?|技术路线(?:与|和)?产业链|"
    r"资本(?:动向|动态)|产业链全景|发展趋势|机会(?:与|和)?挑战|风险分析|"
    r"市场空间(?:与|和)?商业化|行业现状(?:与|和)?趋势"
    r")$"
)

_SPECIFIC_PLANNER_TITLE_RE = re.compile(
    r"MES|APS|PLM|ERP|CRM|SaaS|eVTOL|AI|"
    r"国产替代|离散制造|落地路径|采购决策|决策链|实施失败|归因|"
    r"半托管|海外仓|履约成本|利润结构|适航|航线|订单兑现|"
    r"执行器|传感器|控制器|液冷|超充|投资回收|运营成本|"
    r"支付方|使用场景|渠道建设|消防|检测认证|连锁化|客单价|"
    r"医生供给|保险支付|AI实训|院校采购|就业导向"
)


def _chapter_title_is_generic_template(title: Any) -> bool:
    text = re.sub(r"\s+", "", str(title or "")).strip()
    if not text:
        return True
    return bool(_GENERIC_PLANNER_TITLE_RE.search(text))


def _chapter_title_is_specific(title: Any, query: str = "") -> bool:
    text = re.sub(r"\s+", "", str(title or "")).strip()
    if not text or _chapter_title_is_generic_template(text):
        return False
    if _SPECIFIC_PLANNER_TITLE_RE.search(text):
        return True
    query_text = re.sub(r"\s+", "", str(query or "")).strip()
    if query_text:
        topic_tokens = [
            token
            for token in re.split(r"[：:、，,；;（）()/\s]+", query_text)
            if len(token) >= 2 and token not in {"中国", "研究", "行业报告", "分析"}
        ]
        if any(token and token in text for token in topic_tokens[:12]):
            return True
    return len(text) >= 12 and not re.search(r"^(市场|竞争|政策|技术|资本|风险|趋势)", text)


def _planner_chapters_are_specific(chapters: List[Dict[str, Any]], *, query: str = "") -> bool:
    if len(chapters) < 3:
        return False
    titles = [str(chapter.get("chapter_title") or chapter.get("title") or "").strip() for chapter in chapters]
    if not any(titles):
        return False
    generic_count = sum(1 for title in titles if _chapter_title_is_generic_template(title))
    specific_count = sum(1 for title in titles if _chapter_title_is_specific(title, query=query))
    return specific_count >= max(2, len(titles) // 2) and generic_count < len(titles)


def _hypothesis_ref_index(*values: Any) -> Optional[int]:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        match = re.search(r"hypothesis[_-]?H?(\d+)", text, re.I)
        if not match:
            match = re.search(r"\bH(\d+)\b", text, re.I)
        if not match:
            match = re.search(r"^hypothesis[_-]?(\d+)$", text, re.I)
        if match:
            try:
                index = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if index > 0:
                return index
    return None


_CHAPTER_ALIGNMENT_TERMS = [
    "采购",
    "决策",
    "实施",
    "失败",
    "风险",
    "落地",
    "路径",
    "技术",
    "产品",
    "能力",
    "需求",
    "政策",
    "成本",
    "渠道",
    "客户",
    "订单",
    "商业化",
    "场景",
    "供应链",
    "认证",
    "适航",
    "海外仓",
    "履约",
    "利润",
    "支付",
    "医生",
    "保险",
    "就业",
    "课程",
]


def _chapter_alignment_score(text: str, chapter: Dict[str, Any]) -> int:
    source = re.sub(r"\s+", "", str(text or "")).lower()
    target = re.sub(
        r"\s+",
        "",
        " ".join(
            str(chapter.get(key) or "")
            for key in ("chapter_title", "core_question", "chapter_question")
        ),
    ).lower()
    if not source or not target:
        return 0
    score = 0
    for term in _CHAPTER_ALIGNMENT_TERMS:
        normalized = term.lower()
        if normalized in source and normalized in target:
            score += 3 if len(term) >= 3 else 2
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{1,}", source):
        if token.lower() in target:
            score += 2
    return score


def _best_chapter_id_for_text(text: str, chapters: List[Dict[str, Any]]) -> str:
    best_id = ""
    best_score = 0
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        if not chapter_id:
            continue
        score = _chapter_alignment_score(text, chapter)
        if score > best_score:
            best_score = score
            best_id = chapter_id
    return best_id if best_score >= 3 else ""


def _remap_hypothesis_chapter_refs(
    items: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chapter_ids = [str(chapter.get("chapter_id") or "").strip() for chapter in chapters]
    chapter_by_id = {str(chapter.get("chapter_id") or "").strip(): chapter for chapter in chapters}
    remapped: List[Dict[str, Any]] = []
    for item in items:
        current_chapter_id = str(item.get("chapter_id") or "").strip()
        mapped_chapter_id = current_chapter_id
        text_for_alignment = " ".join(
            str(item.get(key) or "")
            for key in (
                "query",
                "evidence_goal",
                "question",
                "targets_gap",
                "dimension_name",
                "chapter_title",
                "hypothesis_statement",
            )
        )
        semantic_chapter_id = _best_chapter_id_for_text(text_for_alignment, chapters)
        index = _hypothesis_ref_index(
            item.get("hypothesis_id"),
            item.get("task_id"),
            item.get("goal_id"),
            item.get("evidence_goal_id"),
            item.get("dimension_id"),
            current_chapter_id,
        )
        if index and index > len(chapter_ids):
            index = _hypothesis_ref_index(
                item.get("hypothesis_id"),
                item.get("task_id"),
                item.get("goal_id"),
                item.get("evidence_goal_id"),
            )
        if index and index > len(chapter_ids):
            index = _hypothesis_ref_index(
                item.get("dimension_id"),
                current_chapter_id,
            )
        is_hypothesis_task = any(
            _hypothesis_ref_index(item.get(key)) is not None
            for key in ("hypothesis_id", "task_id", "goal_id", "evidence_goal_id")
        )
        if semantic_chapter_id and (current_chapter_id not in chapter_by_id or is_hypothesis_task):
            mapped_chapter_id = semantic_chapter_id
        elif current_chapter_id not in chapter_by_id and index and index <= len(chapter_ids):
            mapped_chapter_id = chapter_ids[index - 1]
        if not mapped_chapter_id:
            remapped.append(dict(item))
            continue
        next_item = {**item, "chapter_id": mapped_chapter_id}
        if mapped_chapter_id != current_chapter_id:
            next_item.setdefault("source_hypothesis_chapter_id", current_chapter_id)
            dimension_id = str(next_item.get("dimension_id") or "").strip()
            if (
                not dimension_id
                or dimension_id == current_chapter_id
                or dimension_id not in chapter_by_id
                and _hypothesis_ref_index(dimension_id) is not None
            ):
                next_item["dimension_id"] = mapped_chapter_id
            chapter = chapter_by_id.get(mapped_chapter_id) or {}
            if chapter.get("chapter_title") and not next_item.get("chapter_title"):
                next_item["chapter_title"] = chapter.get("chapter_title")
            if chapter.get("core_question") and not next_item.get("chapter_question"):
                next_item["chapter_question"] = chapter.get("core_question")
        remapped.append(next_item)
    return remapped


def _remap_hypotheses_to_planner_chapters(
    hypotheses: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chapter_ids = [str(chapter.get("chapter_id") or "").strip() for chapter in chapters]
    chapter_by_id = {str(chapter.get("chapter_id") or "").strip(): chapter for chapter in chapters}
    remapped: List[Dict[str, Any]] = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        current_dimension_id = str(hypothesis.get("dimension_id") or "").strip()
        text_for_alignment = " ".join(
            str(hypothesis.get(key) or "")
            for key in (
                "statement",
                "claim_to_test",
                "hypothesis_statement",
                "dimension_name",
            )
        )
        text_for_alignment = " ".join(
            [text_for_alignment]
            + [str(item) for item in _as_list(hypothesis.get("must_prove"))]
            + [str(item) for item in _as_list(hypothesis.get("must_disprove"))]
        )
        mapped_chapter_id = _best_chapter_id_for_text(text_for_alignment, chapters)
        if not mapped_chapter_id and index <= len(chapter_ids):
            mapped_chapter_id = chapter_ids[index - 1]
        if not mapped_chapter_id:
            remapped.append(dict(hypothesis))
            continue
        chapter = chapter_by_id.get(mapped_chapter_id) or {}
        next_hypothesis = {
            **hypothesis,
            "dimension_id": mapped_chapter_id,
            "dimension_name": str(chapter.get("chapter_title") or hypothesis.get("dimension_name") or "").strip(),
        }
        if current_dimension_id and current_dimension_id != mapped_chapter_id:
            next_hypothesis.setdefault("source_hypothesis_dimension_id", current_dimension_id)
        remapped.append(next_hypothesis)
    return remapped


ACADEMIC_ROLE_SPECS = [
    ("support", "支撑材料", ["official_data", "market_research"], "official_data"),
    ("metric", "指标口径", ["official_data", "market_research"], "market_data"),
    ("case", "案例/实践材料", ["market_research", "news_event", "official_data"], "case"),
    ("counter", "反向证据", ["news_event", "market_research"], "counter"),
    ("filing", "政策/标准/公告", ["official_data", "market_research"], "filing"),
    ("source_check", "来源交叉验证", ["official_data", "market_research"], "source_check"),
]


ACADEMIC_SOURCE_PRIORITY_BY_ROLE = {
    "support": ["官方", "教育部", "人社部", "财政部", "高校", "协会", "研究报告"],
    "metric": ["统计", "就业数据", "招生数据", "证书数据", "高校", "协会"],
    "case": ["培养方案", "招聘岗位", "职业案例", "高校案例", "监管案例"],
    "counter": ["就业风险", "自动化替代", "岗位饱和", "技能错配", "监管风险"],
    "filing": ["政策文件", "专业目录", "教学标准", "职业资格公告", "监管公告"],
    "source_check": ["官方", "原文", "政策文件", "协会", "高校", "研究报告"],
}


def _dedupe_terms(items: List[str], *, limit: int = 8) -> List[str]:
    result: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _topic_anchor_terms(query: str) -> List[str]:
    subject = _research_subject(query)
    if not subject:
        return []
    terms = [subject]
    if re.search(r"中国|国内|国产", query) and not re.search(r"中国|国内|国产", subject):
        terms.append("中国")
    return _dedupe_terms(terms, limit=3)


def _terms_for_role(hypothesis: Dict[str, Any], role: str) -> List[str]:
    bundle = _as_dict(hypothesis.get("evidence_bundle"))
    if role == "counter":
        return [str(item) for item in hypothesis.get("must_disprove") or bundle.get("counter") or [] if str(item).strip()][:5]
    if role == "metric":
        return [str(item) for item in hypothesis.get("must_prove") or bundle.get("metric") or [] if str(item).strip()][:5]
    if role == "case":
        return [str(item) for item in bundle.get("case") or hypothesis.get("must_prove") or [] if str(item).strip()][:5]
    if role == "filing":
        return [str(item) for item in bundle.get("filing") or [] if str(item).strip()][:5]
    if role == "source_check":
        return [str(item) for item in bundle.get("source_check") or [] if str(item).strip()][:5]
    return [str(item) for item in hypothesis.get("must_prove") or [] if str(item).strip()][:5]


ROLE_SEARCH_HINTS = {
    "support": ["行业研究", "案例", "趋势"],
    "metric": ["指标", "数据", "统计", "收入", "成本"],
    "case": ["案例", "项目", "客户", "落地"],
    "counter": ["失败", "风险", "利用率不足", "盈利困难"],
    "filing": ["公告", "年报", "政策文件", "标准"],
    "source_check": ["原文", "来源", "口径", "交叉验证"],
    "technology_product": ["技术路线", "产品方案", "标准", "专利"],
    "expert": ["研报", "白皮书", "专家", "行业研究"],
}


def _search_query_for_role(
    *,
    query: str,
    hypothesis: Dict[str, Any],
    role: str,
    terms: List[str],
    max_chars: int = 96,
) -> str:
    topic_terms = _topic_anchor_terms(query)
    role_hints = ROLE_SEARCH_HINTS.get(role, [])
    must_disprove = [str(item) for item in hypothesis.get("must_disprove") or [] if str(item).strip()]
    role_terms = terms
    if role == "counter" and must_disprove:
        role_terms = must_disprove[:3]
    if not role_terms:
        role_terms = [str(item) for item in hypothesis.get("must_prove") or [] if str(item).strip()][:3]
    parts = _dedupe_terms([*topic_terms, *role_terms[:4], *role_hints[:3]], limit=8)
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = _compact(" ".join(parts[:6]), max_chars)
    return text or _compact(query, max_chars)


_LEAKY_SEARCH_QUERY_RE = re.compile(
    r"official data|market research|market size price|official filing|customer certification|"
    r"product docs|brokerage association|industry research expert|"
    r"反证\s*风险\s*失败案例|客户不买账|替代方案|负面",
    flags=re.I,
)


def _search_query_needs_compaction(query_text: str, *, query: str = "", hypothesis: Dict[str, Any] | None = None) -> bool:
    text = str(query_text or "")
    if _LEAKY_SEARCH_QUERY_RE.search(text):
        return True
    if len(text) > 72:
        return True
    subject = _research_subject(query)
    if subject and text.count(subject) >= 2:
        return True
    if re.search(r"[:：]", text) and len(text) > 56:
        return True
    statement = str((hypothesis or {}).get("statement") or (hypothesis or {}).get("claim_to_test") or "").strip()
    return bool(statement and statement in text)


def _normalize_search_task_queries(
    *,
    query: str,
    tasks: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hypothesis_by_id = {
        str(item.get("hypothesis_id") or "").strip(): item
        for item in hypotheses
        if isinstance(item, dict) and str(item.get("hypothesis_id") or "").strip()
    }
    normalized: List[Dict[str, Any]] = []
    for task in tasks:
        next_task = dict(task)
        hypothesis = hypothesis_by_id.get(str(next_task.get("hypothesis_id") or "").strip()) or {}
        role = str(next_task.get("proof_role") or next_task.get("intent") or "support").strip() or "support"
        terms = [str(item) for item in _as_list(next_task.get("must_have_terms")) if str(item).strip()]
        if not terms:
            terms = _terms_for_role(hypothesis, role)
        if _search_query_needs_compaction(str(next_task.get("query") or ""), query=query, hypothesis=hypothesis):
            next_task["query"] = _search_query_for_role(query=query, hypothesis=hypothesis, role=role, terms=terms)
            next_task["query_compacted_from_hypothesis"] = True
        normalized.append(next_task)
    return normalized


def _query_aspects(query: str) -> List[str]:
    text = re.sub(r"\s+", "", str(query or "")).strip()
    if not re.search(r"[:：]", text):
        return []
    _, tail = re.split(r"[:：]", text, 1)
    tail = re.sub(r"(研究|报告|分析|深度|专题)$", "", tail).strip()
    raw_parts = re.split(r"[、，,；;]|与|和|及", tail)
    aspects: List[str] = []
    for part in raw_parts:
        item = re.sub(r"(研究|报告|分析|路径|变化)$", "", str(part or "").strip())
        if len(item) < 3:
            continue
        if item not in aspects:
            aspects.append(item)
    return aspects[:5]


def _chapter_title_for_query_aspect(aspect: str) -> str:
    aspect = _compact(aspect, 24).strip(" ：:，,。")
    if re.search(r"场景|需求|用户|客户|物流", aspect):
        return f"{aspect}需求与运行约束"
    if re.search(r"资产利用率|利用率|周转|效率", aspect):
        return f"{aspect}关键影响因素与效率评估"
    if re.search(r"盈利|商业模式|收益|成本|利润", aspect):
        return f"{aspect}与成本收益分析"
    if re.search(r"采购|决策链|选型", aspect):
        return f"{aspect}与用户需求分析"
    if re.search(r"实施|落地", aspect):
        return f"{aspect}与关键成功因素"
    if re.search(r"失败|风险|扰动|法案|监管|合规", aspect):
        return f"{aspect}机制与影响路径"
    if re.search(r"客户结构", aspect):
        return f"{aspect}与可持续性"
    if re.search(r"产能", aspect):
        return f"{aspect}与订单恢复关联"
    return f"{aspect}分析"


def _query_aspect_chapters(query: str) -> List[Dict[str, Any]]:
    aspects = _query_aspects(query)
    if len(aspects) < 2:
        return []
    subject = _research_subject(query)
    chapters: List[Dict[str, Any]] = []
    if subject:
        chapters.append(
            {
                "chapter_id": "ch_01",
                "chapter_title": f"{_compact(subject, 28)}发展背景与研究边界",
                "core_question": f"{subject}的基本边界、需求背景和研究口径是什么？",
                "reason_to_include": "先界定研究对象，避免后续证据和结论跑偏。",
            }
        )
    for aspect in aspects:
        chapter_index = len(chapters) + 1
        title = _chapter_title_for_query_aspect(aspect)
        chapters.append(
            {
                "chapter_id": f"ch_{chapter_index:02d}",
                "chapter_title": title,
                "core_question": f"{title}如何影响行业判断？",
                "reason_to_include": "来自用户题目中的明确研究角度。",
            }
        )
    if not any(re.search(r"风险|失败|边界|约束", str(item.get("chapter_title") or "")) for item in chapters):
        chapter_index = len(chapters) + 1
        chapters.append(
            {
                "chapter_id": f"ch_{chapter_index:02d}",
                "chapter_title": "典型案例、风险边界与策略建议",
                "core_question": "哪些案例可以验证主线，哪些风险会改变结论？",
                "reason_to_include": "补充案例验证和风险边界，避免只写正向判断。",
            }
        )
    return chapters[:7]


def _build_chapter_goal_task_packages(
    *,
    query: str,
    hypotheses: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    chapters: List[Dict[str, Any]] = []
    goals: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    topic_terms = _topic_anchor_terms(query)
    academic_field = _looks_like_academic_or_professional_field(query)
    role_specs = ACADEMIC_ROLE_SPECS if academic_field else ROLE_SPECS
    source_priority_by_role = ACADEMIC_SOURCE_PRIORITY_BY_ROLE if academic_field else SOURCE_PRIORITY_BY_ROLE
    for index, hypothesis in enumerate(hypotheses[:8], start=1):
        hypothesis_id = str(hypothesis.get("hypothesis_id") or f"H{index}").strip()
        statement = _compact(hypothesis.get("statement") or hypothesis.get("claim_to_test") or query, 120)
        chapter_id = f"ch_{index:02d}"
        proof_standard = str(hypothesis.get("proof_standard") or "medium").strip().lower()
        counter_required = bool(hypothesis.get("counter_evidence_required", False))
        chapter_goals: List[Dict[str, Any]] = []
        chapter_tasks: List[Dict[str, Any]] = []
        for role_index, (role, label, lane_targets, evidence_type) in enumerate(role_specs, start=1):
            if role == "counter" and not (counter_required or hypothesis.get("must_disprove")):
                continue
            terms = _terms_for_role(hypothesis, role)
            must_have_terms = _dedupe_terms([*topic_terms, *terms], limit=8)
            goal_id = f"{hypothesis_id}_{role}"
            question = f"{statement}：补齐{label}，并保留来源、时间、范围和口径。"
            goal = {
                "goal_id": goal_id,
                "dimension_id": chapter_id,
                "dimension_name": statement,
                "chapter_id": chapter_id,
                "chapter_title": statement,
                "question": question,
                "expected_metrics": terms,
                "must_have_terms": must_have_terms[:5],
                "forbidden_terms": [],
                "source_priority": source_priority_by_role.get(role, lane_targets),
                "freshness": "recent",
                "min_sources": 2 if proof_standard == "strong" else 1,
                "evidence_type": evidence_type,
                "hypothesis_id": hypothesis_id,
                "hypothesis_statement": statement,
                "proof_standard": proof_standard,
                "decision_use": hypothesis.get("decision_use") or "research",
                "counter_evidence_required": counter_required,
                "proof_role": role,
            }
            task = {
                "task_id": f"{chapter_id}_{role_index:02d}_{role}",
                "agent": "iqs",
                "dimension_id": chapter_id,
                "dimension_name": statement,
                "chapter_id": chapter_id,
                "chapter_title": statement,
                "chapter_question": statement,
                "query": _search_query_for_role(query=query, hypothesis=hypothesis, role=role, terms=terms),
                "evidence_goal": question,
                "evidence_goal_id": goal_id,
                "intent": evidence_type,
                "proof_role": role,
                "lane_targets": lane_targets,
                "min_source_level": "A" if proof_standard == "strong" else "B",
                "must_have_terms": must_have_terms,
                "forbidden_terms": [],
                "source_priority": source_priority_by_role.get(role, lane_targets),
                "hypothesis_id": hypothesis_id,
                "hypothesis_statement": statement,
                "proof_standard": proof_standard,
                "decision_use": hypothesis.get("decision_use") or "research",
                "counter_evidence_required": counter_required,
                "metric_definitions": hypothesis.get("metric_definitions") or [],
            }
            goals.append(goal)
            tasks.append(task)
            chapter_goals.append(goal)
            chapter_tasks.append(task)
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": statement,
                "chapter_question": statement,
                "core_question": statement,
                "reason_to_include": f"检验假设 {hypothesis_id} 是否成立。",
                "dimension_id": chapter_id,
                "dimension_name": statement,
                "source_template_keys": ["official_data", "market_research", "company_filing", "case", "counter_evidence"],
                "required_evidence_mix": ["official_data", "market_research", "company_filing", "case", "counter_evidence"],
                "min_total_sources": 8 if proof_standard == "strong" else 5,
                "min_ab_sources": 2 if proof_standard == "strong" else 1,
                "min_counter_sources": 1 if counter_required else 0,
                "hypothesis_id": hypothesis_id,
                "proof_standard": proof_standard,
                "counter_evidence_required": counter_required,
                "evidence_goals": chapter_goals,
                "search_tasks": chapter_tasks,
            }
        )
    return chapters, goals, tasks


def apply_problem_framing(plan: Dict[str, Any], framing: Dict[str, Any]) -> Dict[str, Any]:
    plan = _as_dict(plan)
    framing = _as_dict(framing)
    hypotheses = [dict(item) for item in framing.get("hypotheses") or [] if isinstance(item, dict)]
    if not hypotheses:
        return plan
    legacy_chapters = [dict(item) for item in plan.get("chapters") or [] if isinstance(item, dict)]
    keep_planner_chapters = _planner_chapters_are_specific(
        legacy_chapters,
        query=str(plan.get("query") or framing.get("planning_query") or ""),
    )
    existing_dropped = plan.get("dropped_template_sections")
    dropped_template_sections = list(existing_dropped) if isinstance(existing_dropped, list) else []
    if legacy_chapters and not keep_planner_chapters:
        dropped_template_sections.append(
            {
                "source": "legacy_planner_chapters",
                "reason": "replaced_by_problem_framing_hypotheses",
                "items": legacy_chapters,
            }
        )
    dimensions: List[Dict[str, Any]] = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        hypothesis_id = str(hypothesis.get("hypothesis_id") or f"H{index}").strip()
        statement = str(hypothesis.get("claim_to_test") or hypothesis.get("statement") or "").strip()
        dimensions.append(
            {
                "dimension_id": f"hypothesis_{hypothesis_id}",
                "dimension_name": statement or f"假设 {index}",
                "purpose": statement,
                "must_have_terms": [statement],
                "forbidden_terms": [],
                "hypothesis_id": hypothesis_id,
            }
        )
        hypothesis.setdefault("dimension_id", f"hypothesis_{hypothesis_id}")
        hypothesis.setdefault("dimension_name", statement or f"假设 {index}")
    framed_chapters, framed_evidence_goals, framed_search_tasks = _build_chapter_goal_task_packages(
        query=str(plan.get("query") or plan.get("core_question") or ""),
        hypotheses=hypotheses,
    )
    query_for_framing = str(plan.get("query") or plan.get("core_question") or "")
    aspect_chapters = [] if keep_planner_chapters else _query_aspect_chapters(query_for_framing)
    if keep_planner_chapters:
        chapters = legacy_chapters
        hypotheses = _remap_hypotheses_to_planner_chapters(hypotheses, chapters)
        dimensions = [
            {
                "dimension_id": str(hypothesis.get("dimension_id") or "").strip(),
                "dimension_name": str(hypothesis.get("dimension_name") or hypothesis.get("statement") or "").strip(),
                "purpose": str(hypothesis.get("statement") or hypothesis.get("claim_to_test") or "").strip(),
                "must_have_terms": _as_list(hypothesis.get("must_prove")),
                "forbidden_terms": [],
                "hypothesis_id": str(hypothesis.get("hypothesis_id") or "").strip(),
            }
            for hypothesis in hypotheses
            if str(hypothesis.get("dimension_id") or "").strip()
        ]
        evidence_goals = _remap_hypothesis_chapter_refs([
            dict(item)
            for item in plan.get("evidence_goals") or []
            if isinstance(item, dict)
        ] or framed_evidence_goals, chapters)
        search_tasks = _remap_hypothesis_chapter_refs([
            dict(item)
            for item in plan.get("search_tasks") or []
            if isinstance(item, dict)
        ] or framed_search_tasks, chapters)
        chapter_source = "llm_planner_with_problem_framing"
        chapters_come_from_hypotheses = False
    elif aspect_chapters:
        chapters = aspect_chapters
        hypotheses = _remap_hypotheses_to_planner_chapters(hypotheses, chapters)
        dimensions = [
            {
                "dimension_id": str(hypothesis.get("dimension_id") or "").strip(),
                "dimension_name": str(hypothesis.get("dimension_name") or hypothesis.get("statement") or "").strip(),
                "purpose": str(hypothesis.get("statement") or hypothesis.get("claim_to_test") or "").strip(),
                "must_have_terms": _as_list(hypothesis.get("must_prove")),
                "forbidden_terms": [],
                "hypothesis_id": str(hypothesis.get("hypothesis_id") or "").strip(),
            }
            for hypothesis in hypotheses
            if str(hypothesis.get("dimension_id") or "").strip()
        ]
        evidence_goals = _remap_hypothesis_chapter_refs(framed_evidence_goals, chapters)
        search_tasks = _remap_hypothesis_chapter_refs(framed_search_tasks, chapters)
        chapter_source = "query_aspect_chapters"
        chapters_come_from_hypotheses = False
    else:
        chapters = framed_chapters
        evidence_goals = framed_evidence_goals
        search_tasks = framed_search_tasks
        chapter_source = "problem_framing_hypotheses"
        chapters_come_from_hypotheses = True
    search_tasks = _normalize_search_task_queries(
        query=query_for_framing,
        tasks=search_tasks,
        hypotheses=hypotheses,
    )
    source_requirements = {
        **_as_dict(plan.get("source_requirements")),
        "core_claim": ["A", "B"],
        "supporting_claim": ["A", "B"],
        "clue_only": ["C"],
        "appendix_only": ["D"],
    }
    quality_rules = {
        **_as_dict(plan.get("quality_rules")),
        "chapters_come_from_hypotheses": chapters_come_from_hypotheses,
        "chapter_source": chapter_source,
        "core_claim_requires_A_or_B": True,
        "c_level_is_directional_signal": True,
        "single_evidence_cannot_be_claim": True,
        "reject_generic_industry_fallback": True,
        "research_domain": framing.get("research_domain") or plan.get("research_domain"),
        "topic_anchor_terms": _as_list(framing.get("topic_anchor_terms")) or _topic_anchor_terms(str(plan.get("query") or "")),
    }
    return {
        **plan,
        "core_question": framing.get("core_question") or plan.get("core_question") or plan.get("query"),
        "research_object": framing.get("research_object") or plan.get("research_object"),
        "decision_context": framing.get("decision_context") or plan.get("decision_context"),
        "research_domain": framing.get("research_domain") or plan.get("research_domain"),
        "problem_framing": framing,
        "legacy_planner_chapters": [] if keep_planner_chapters else legacy_chapters,
        "planner_chapters_preserved": bool(keep_planner_chapters),
        "legacy_planner_dimensions": plan.get("dimensions") or [],
        "legacy_planner_search_tasks": plan.get("search_tasks") or [],
        "hypotheses": hypotheses,
        "chapters": chapters,
        "dimensions": dimensions,
        "evidence_goals": evidence_goals,
        "search_tasks": search_tasks,
        "dropped_template_sections": dropped_template_sections,
        "source_requirements": source_requirements,
        "evidence_coverage_requirements": _as_dict(framing.get("coverage_requirements")),
        "report_depth_target": "deep",
        "quality_rules": quality_rules,
    }
