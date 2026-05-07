"""PDF 导出器 — 将 Markdown 研报导出为 PDF。

依赖（按优先级尝试）:
  1. weasyprint     # pip install weasyprint
  2. pandoc+xelatex # 需安装 pandoc 和 texlive
  3. Edge/Chrome    # Windows 自带，无需安装
"""

from pathlib import Path
from typing import Optional
import subprocess
import tempfile


class PDFExporter:
    """
    将 Markdown 研报导出为 PDF。

    用法:
      exporter = PDFExporter()
      pdf_path = exporter.export(
          markdown=report_text,
          output_path="data/reports/宁德时代_2024Q3.pdf",
      )
    """

    def __init__(self, css_path: Optional[str] = None):
        self.css_path = css_path

    def export(self, markdown: str, output_path: str) -> Optional[str]:
        try:
            return self._export_via_weasyprint(markdown, output_path)
        except (ImportError, OSError, RuntimeError):
            result = self._export_via_pandoc(markdown, output_path)
            if result:
                return result
            return self._export_via_browser(markdown, output_path)

    def _build_html(self, markdown: str) -> str:
        import markdown as md_lib
        html = md_lib.markdown(markdown, extensions=["tables", "fenced_code"])

        css = ""
        if self.css_path:
            p = Path(self.css_path)
            if p.exists():
                css = p.read_text(encoding="utf-8")

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{css}</style></head>
<body>{html}</body>
</html>"""

    def _export_via_weasyprint(self, markdown: str, output_path: str) -> Optional[str]:
        from weasyprint import HTML
        full_html = self._build_html(markdown)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=full_html).write_pdf(str(out))
        return str(out.resolve())

    def _export_via_pandoc(self, markdown: str, output_path: str) -> Optional[str]:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
            f.write(markdown)
            md_path = f.name
        try:
            result = subprocess.run(
                ["pandoc", md_path, "-o", output_path, "--pdf-engine=xelatex"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                return str(Path(output_path).resolve())
            return None
        except Exception:
            return None

    def _export_via_browser(self, markdown: str, output_path: str) -> Optional[str]:
        full_html = self._build_html(markdown)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        html_path = out.with_suffix(".html")
        html_path.write_text(full_html, encoding="utf-8")
        file_url = html_path.resolve().as_uri()

        browsers = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]

        for browser in browsers:
            if not Path(browser).exists():
                continue
            try:
                subprocess.run(
                    [browser, "--headless", "--disable-gpu",
                     f"--print-to-pdf={str(out)}", file_url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=60,
                )
                if out.exists() and out.stat().st_size > 0:
                    return str(out.resolve())
            except Exception:
                continue

        return None
