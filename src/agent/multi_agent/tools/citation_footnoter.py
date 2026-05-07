"""段落级溯源脚注生成器。

在 LLM 生成研报正文后，逐段扫描，
为每个数据点绑定对应的 RAG 检索原文引用。
"""

import re
from typing import Any, Dict, List


class CitationFootnoter:
    """
    段落级溯源脚注生成。

    用法:
      footnoter = CitationFootnoter()
      report_with_footnotes = footnoter.process(report_text, source_docs)
    """

    def process(self, report_text: str, source_docs: List[Dict]) -> str:
        paragraphs = report_text.split("\n\n")
        footnoted = []
        for para in paragraphs:
            stripped = para.strip()
            if not stripped or len(stripped) < 30:
                footnoted.append(para)
                continue
            if stripped.startswith("!["):
                footnoted.append(para)
                continue
            if stripped.startswith("|"):
                footnoted.append(para)
                continue
            if "📖" in stripped or "[来源" in stripped:
                footnoted.append(para)
                continue
            footnote = self._find_source_for_paragraph(stripped, source_docs)
            if footnote:
                footnoted.append(f"{para}\n\n> 📖 {footnote}")
            else:
                footnoted.append(para)
        return "\n\n".join(footnoted)

    def _find_source_for_paragraph(self, para: str, docs: List[Dict]) -> str:
        numbers = re.findall(r'[\d]+(?:\.[\d]+)?%?', para)
        meaningful_numbers = [n for n in numbers if len(n) >= 2]

        best_match = None
        best_score = 0

        for doc in docs:
            content = doc.get("content", "")
            if not content:
                continue
            score = 0
            for num in meaningful_numbers[:5]:
                if num in content:
                    score += 1
            if score > best_score:
                best_score = score
                source = doc.get("source", doc.get("source_path", "unknown"))
                page = doc.get("page", doc.get("chunk_id", ""))
                page_str = f", p.{page}" if page else ""
                best_match = f"数据来源: {source}{page_str}"

        if best_score >= 2:
            return best_match
        return ""
