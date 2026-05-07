"""Chain-of-Analysis 生成器 — 先分析后写作。

受 FinSight (ACL 2026) 的 CoA 机制启发。
先产出结构化分析链，再基于分析链写报告。
"""

import json
from typing import Any, Dict, List, Optional

from .json_parser import safe_parse_json_with_default


class ChainOfAnalysis:
    """
    CoA 生成器。

    输入: blackboard 中的所有结构化数据
    输出: 3-7 条分析链，每条含 observation + interpretation + implication + confidence + evidence
    """

    PROMPT = """You are a senior financial analyst.
Based on the following data, generate 3-7 structured analysis chains.

Each chain must follow:
- observation: specific data point from the provided data
- interpretation: what this data means in business context
- implication: what this means for investment decision

Rules:
- EVERY observation must have a corresponding data source
- confidence must be based on data completeness
- DO NOT repeat the same data point in multiple chains

Output JSON:
{"chains": [{"observation": "...", "interpretation": "...", "implication": "...", "confidence": 0.85, "evidence": ["src1", "src2"]}]}
"""

    def generate(self, context: Dict[str, Any], llm) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage

        data_str = json.dumps(context, ensure_ascii=False, indent=2)
        prompt = self.PROMPT + "\n\n## 数据\n" + data_str

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            result = self._extract_json(content)
            chains = result.get("chains", [])
            if not chains:
                return {"chains": []}
            return {"chains": chains[:7]}
        except Exception:
            return {"chains": []}

    @staticmethod
    def _extract_json(text: str) -> Dict:
        return safe_parse_json_with_default(text, {"chains": []})
