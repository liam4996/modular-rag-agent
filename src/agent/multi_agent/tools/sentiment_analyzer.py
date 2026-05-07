"""公告事件情感分类器。

分类: 利好 / 利空 / 中性

实现方式：基于关键词规则（Phase 1），未来可升级为轻量模型。
"""

from typing import Dict


class SentimentAnalyzer:
    """
    公告情感分析工具。

    用法:
      analyzer = SentimentAnalyzer()
      result = analyzer.classify("公司宣布回购10亿元股票")
      # → {"sentiment": "利好", "confidence": 0.85}
    """

    POSITIVE_KEYWORDS = ["回购", "增持", "分红", "业绩预增", "中标", "获批", "战略合作", "订单"]
    NEGATIVE_KEYWORDS = ["减持", "预亏", "违规", "处罚", "诉讼", "退市", "ST", "资产冻结"]

    def classify(self, title: str, content: str = "") -> Dict:
        text = f"{title} {content}"
        pos_score = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
        neg_score = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)

        if pos_score > neg_score:
            return {
                "sentiment": "利好",
                "confidence": round(pos_score / (pos_score + neg_score + 1), 2),
                "label": "利好",
            }
        elif neg_score > pos_score:
            return {
                "sentiment": "利空",
                "confidence": round(neg_score / (pos_score + neg_score + 1), 2),
                "label": "利空",
            }
        else:
            return {"sentiment": "中性", "confidence": 0.5, "label": "中性"}
