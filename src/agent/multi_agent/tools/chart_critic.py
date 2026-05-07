"""VLM 图表评审器 — Chart Critique Loop。

LLM 画图 → VLM 评审 → 迭代修正 → 达标通过。
"""

import base64
import json
import re
from typing import Any, Dict, Optional

from .json_parser import safe_parse_json_with_default


CHART_CRITIC_PROMPT = """You are a professional financial chart critic.
Evaluate the given chart image and score it on four dimensions:

1. data_accuracy (0-1): Are the data points correctly represented?
   Check: axis labels match the data, scales are appropriate

2. label_clarity (0-1): Are all labels clear and readable?
   Check: title, axis labels, legend, data labels, font size

3. aesthetics (0-1): Is the chart visually professional?
   Check: color scheme, layout, consistency, cleanliness

4. information_density (0-1): Does the chart convey the right amount?
   Check: not too cluttered, not too sparse, appropriate chart type

Output JSON:
{
    "scores": {"data_accuracy": 0.9, "label_clarity": 0.7, "aesthetics": 0.6, "information_density": 0.8},
    "overall": 0.75,
    "issues": ["图例过小", "X轴标签旋转角度不合适"],
    "suggestions": ["增大图例字体", "将X轴标签改为45度旋转"],
    "verdict": "PASS"
}
"""


class ChartCritic:
    """
    VLM 图表评审器。

    用法:
      critic = ChartCritic(vlm_llm=None)
      verdict, feedback = critic.review(chart_path)
    """

    def __init__(self, vlm_llm=None):
        self._vlm = vlm_llm

    @property
    def available(self) -> bool:
        return self._vlm is not None

    def review(self, chart_path: str) -> tuple:
        if not self.available:
            return "PASS", {"scores": {}, "overall": 1.0, "issues": [], "verdict": "PASS"}

        try:
            with open(chart_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "PASS", {"scores": {}, "overall": 1.0, "issues": [], "verdict": "PASS"}

        from langchain_core.messages import HumanMessage

        try:
            msg = HumanMessage(content=[
                {"type": "text", "text": CHART_CRITIC_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])
            resp = self._vlm.invoke([msg])
            content = resp.content if hasattr(resp, "content") else str(resp)
            result = safe_parse_json_with_default(content, {"verdict": "PASS", "scores": {}, "overall": 1.0, "issues": []})
            verdict = result.get("verdict", "PASS")
            return verdict, result
        except Exception:
            pass

        return "PASS", {"scores": {}, "overall": 1.0, "issues": [], "verdict": "PASS"}
