# Report Layout Template Assets

这个目录是离线报告布局资产库，暂时不接入 `full_report`、`writer_agent_clean`、`reformatter_agent` 或其他主流程。

目标是先把“高质量正文报告”的结构、证据要求和分析提示词沉淀成可复用资产，后续再决定接入位置。

## 文件

- `layout_templates.json`：机器可读的报告布局模板库。
- `analysis_prompts.md`：可直接复制使用的报告分析、章节写作、编辑补正提示词。
- `template_library.py`：离线读取、列举、选择模板的小工具，不被主流程 import。

## 设计原则

- 正文优先：默认不生成“数据来源”“附录”“参考来源”这类尾部章节。
- 问题驱动：章节标题必须回答具体研究问题，不能套“市场规模、竞争格局、政策监管、技术路线、资本动态”固定五段式。
- 证据嵌入正文：来源编号用于正文事实、数字、政策、公司动作、案例之后，不在尾部堆来源表。
- 先判断证据够不够：证据不够时输出缺口、补证任务和降级结论，不用模板化文字凑篇幅。
- 反向样本前置：每个重要判断都要写清楚什么信号会推翻它。

## 离线使用示例

```powershell
python rag_pipeline\reporting_templates\template_library.py --list
python rag_pipeline\reporting_templates\template_library.py --select "中美关税、出口管制与市场准入影响半导体、新能源、消费品和互联网"
```

这只是资产预览，不会触发报告生成。
