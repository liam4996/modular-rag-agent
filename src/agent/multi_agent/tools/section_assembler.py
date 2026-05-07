"""章节级 Prompt 拼接器。

读取模板文件 → 渲染变量 → 单次 LLM 调用生成全报告。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


class SectionAssembler:
    """
    章节级 Prompt 管理。

    用法:
      assembler = SectionAssembler(template_dir="templates/finance")
      report = assembler.assemble_single_call(
          report_type="earnings_review",
          data={"fixed_numbers": ..., "market_data": ..., ...},
      )
    """

    def __init__(self, template_dir: str = "templates/finance"):
        self.template_dir = Path(template_dir)

    def load_sections(self, report_type: str) -> List[str]:
        section_dir = self.template_dir / report_type
        if not section_dir.exists():
            return []
        md_files = sorted(section_dir.glob("*.md"))
        sections = []
        for f in md_files:
            sections.append(f.read_text(encoding="utf-8"))
        return sections

    def assemble_single_call(self, report_type: str, data: Dict) -> str:
        sections = self.load_sections(report_type)
        if not sections:
            return ""
        combined = "\n\n".join(sections)

        # 简单变量替换
        for k, v in data.get("fixed_numbers", {}).items():
            placeholder = f"{{{{ {k} }}}}"
            if placeholder in combined:
                combined = combined.replace(placeholder, str(v))

        return combined
