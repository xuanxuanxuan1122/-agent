from __future__ import annotations

from typing import Any, Dict, Iterable, List


UNIVERSAL_REPORT_MODULES: List[Dict[str, Any]] = [
    {
        "module_key": "executive_summary",
        "label": "报告摘要",
        "core_questions": ["最终结论是什么？", "最关键的证据是什么？", "建议动作是什么？"],
        "default_position": "front",
        "always_consider": True,
        "evidence_types": ["synthesis"],
    },
    {
        "module_key": "industry_definition",
        "label": "行业定义与研究范围",
        "core_questions": ["研究对象边界是什么？", "哪些子行业应该纳入或排除？"],
        "triggers": ["定义", "范围", "是什么", "分类", "边界"],
        "evidence_types": ["official_definition", "classification", "standard"],
    },
    {
        "module_key": "market_size",
        "label": "市场规模与增长空间",
        "core_questions": ["市场规模多大？", "增长来自哪里？", "预测口径是否可靠？"],
        "triggers": ["市场", "规模", "空间", "增速", "TAM", "CAGR", "行情", "价格"],
        "evidence_types": ["official_data", "market_research", "forecast", "counter"],
    },
    {
        "module_key": "demand_driver",
        "label": "需求驱动与边际变量",
        "core_questions": ["需求变化来自哪里？", "哪些变量决定未来 6-12 个月变化？"],
        "triggers": ["需求", "订单", "开工", "销量", "采购", "景气", "边际"],
        "evidence_types": ["official_data", "market_price", "inventory", "case", "counter"],
    },
    {
        "module_key": "industry_chain",
        "label": "产业链分析",
        "core_questions": ["价值链如何分配？", "瓶颈在哪里？", "利润流向谁？"],
        "triggers": ["产业链", "上游", "下游", "供应链", "利润", "成本", "库存"],
        "evidence_types": ["company_filing", "cost", "capacity", "case"],
    },
    {
        "module_key": "technology",
        "label": "技术与产品形态",
        "core_questions": ["技术路径是否成熟？", "产品差异来自哪里？"],
        "triggers": ["技术", "产品", "路线", "性能", "替代", "专利", "标准"],
        "evidence_types": ["technical_standard", "patent", "product_doc", "case"],
    },
    {
        "module_key": "customer",
        "label": "客户与应用场景",
        "core_questions": ["谁在购买？", "痛点、ROI 和替代方案是什么？"],
        "triggers": ["客户", "用户", "场景", "采购", "中标", "ROI", "痛点"],
        "evidence_types": ["customer_case", "procurement", "company_filing", "case"],
    },
    {
        "module_key": "business_model",
        "label": "商业模式",
        "core_questions": ["如何赚钱？", "单位经济性和现金流是否成立？"],
        "triggers": ["商业模式", "盈利", "毛利", "现金流", "收费", "变现"],
        "evidence_types": ["company_filing", "case", "financial_metric"],
    },
    {
        "module_key": "competition",
        "label": "竞争格局",
        "core_questions": ["主要玩家是谁？", "份额、壁垒和竞争强度如何？"],
        "triggers": ["竞争", "格局", "玩家", "份额", "厂商", "替代"],
        "evidence_types": ["company_filing", "market_research", "news_event"],
    },
    {
        "module_key": "policy",
        "label": "政策与监管",
        "core_questions": ["政策条款是什么？", "传导链条和执行风险是什么？"],
        "triggers": ["政策", "监管", "法规", "补贴", "规划", "目录", "审批"],
        "evidence_types": ["policy_original", "official_data", "procurement", "counter"],
    },
    {
        "module_key": "capital",
        "label": "资本与交易信号",
        "core_questions": ["融资、并购或估值是否验证商业化？"],
        "triggers": ["融资", "估值", "并购", "上市", "IPO", "股价", "市值"],
        "evidence_types": ["filing_company", "news_event", "market_data"],
    },
    {
        "module_key": "risk",
        "label": "行业风险分析",
        "core_questions": ["什么情况会推翻结论？", "哪些触发器需要跟踪？"],
        "triggers": ["风险", "不确定", "反证", "触发器", "下滑", "过剩", "价格战"],
        "evidence_types": ["counter", "news_event", "filing", "policy"],
    },
    {
        "module_key": "entry_strategy",
        "label": "进入策略",
        "core_questions": ["下一步应怎么进入、采购、投资或验证？"],
        "triggers": ["进入", "策略", "投资", "采购", "立项", "建议", "动作"],
        "evidence_types": ["synthesis", "case", "counter", "metric"],
    },
]


def module_keys() -> List[str]:
    return [str(item.get("module_key")) for item in UNIVERSAL_REPORT_MODULES if item.get("module_key")]


def module_by_key(module_key: str) -> Dict[str, Any]:
    key = str(module_key or "").strip()
    for item in UNIVERSAL_REPORT_MODULES:
        if item.get("module_key") == key:
            return dict(item)
    return {}


def evidence_mix_for_modules(keys: Iterable[str]) -> List[str]:
    evidence_types: List[str] = []
    seen = set()
    for key in keys:
        module = module_by_key(str(key))
        for evidence_type in module.get("evidence_types") or []:
            text = str(evidence_type or "").strip()
            if text and text not in seen:
                seen.add(text)
                evidence_types.append(text)
    return evidence_types
