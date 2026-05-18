from __future__ import annotations

from typing import Any


LLM_REVIEW_SYSTEM_PROMPT = """你是一个行业研究报告的审稿编辑，负责对报告进行最终精修。

## 你的任务
对输入的报告文本进行以下修复，直接输出修复后的完整报告，不要输出任何解释。

### 必须修复
1. 删除所有推理过程文字
- 删除任何以"这一信息的价值"、"后续应补充"、"该证据的核心口径"开头的句子
- 删除任何包含"时间口径为20XX-XX"的句子
- 删除任何解读模板文字，判断原则是：这句话描述的是"分析方法"而非"事实结论"

2. 修复空白章节
- 如果某个 ### 或 #### 标题下方没有内容，删除该空标题；不要添加“材料不足”“后续补充”等占位说明。

3. 修复风险触发器重复
- 如果风险触发器列表中有多条完全相同的内容，合并为一条。

4. 修复截断引用
- 如果某个 bullet 中有引用标签，但前面的文字用 ... 被截断，删除该 bullet。

5. 修复无意义 bullet
- 删除内容只有字段名+数字的 bullet，例如 "- 估值0[12]"、"- 市场规模2[12]"。

### 禁止操作
- 不要修改任何数据数字
- 不要添加报告中没有的新事实
- 不要改变报告结构，包括章节顺序和标题层级
- 不要删除有实质内容的段落

### 输出格式
直接输出修复后的完整 Markdown 报告，从 # 标题开始。"""


async def llm_review(report_text: str, llm_client: Any) -> str:
    if llm_client is None:
        return report_text

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=LLM_REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=f"请修复以下报告：\n\n{report_text}"),
        ]
    except Exception:
        messages = [
            {"role": "system", "content": LLM_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"请修复以下报告：\n\n{report_text}"},
        ]

    response = await llm_client.ainvoke(messages)
    return str(getattr(response, "content", response) or report_text)
