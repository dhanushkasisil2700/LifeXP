"""HTML experience study report with embedded MathJax."""

from __future__ import annotations

import html
import os
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# CSS / JS snippets (inlined for self-contained output)
# ---------------------------------------------------------------------------

_MATHJAX_SCRIPT = """\
<script>
  window.MathJax = {
    tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['$$', '$$']] },
    startup: { typeset: true }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>"""

_CSS = """\
<style>
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #222;
    margin: 2em auto;
    max-width: 1100px;
    padding: 0 1em;
  }
  h1 { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: .3em; }
  h2 { color: #1a3a5c; margin-top: 1.8em; }
  .metadata { background: #f0f4f8; border-left: 4px solid #1a3a5c;
              padding: .6em 1em; margin-bottom: 1.5em; }
  .metadata table { border: none; }
  .metadata td { padding: 2px 8px; }
  table.data { border-collapse: collapse; width: 100%; margin: 1em 0; }
  table.data caption { font-weight: bold; text-align: left;
                        margin-bottom: .4em; color: #444; }
  table.data th { background: #1a3a5c; color: #fff; padding: 6px 10px;
                  text-align: left; white-space: nowrap; }
  table.data td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  table.data tr:nth-child(even) td { background: #f7f9fb; }
  table.data tr:hover td { background: #eaf1fb; }
  .no-data { color: #888; font-style: italic; margin: .5em 0 1em; }
  .formula-block { background: #fafafa; border: 1px solid #ddd;
                   padding: .6em 1em; border-radius: 4px; margin: .8em 0; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                  gap: .6em; margin: .8em 0; }
  .stat-card { background: #f0f4f8; border-radius: 4px; padding: .5em .8em; }
  .stat-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .05em; }
  .stat-value { font-size: 1.2em; font-weight: bold; color: #1a3a5c; }
</style>"""


# ---------------------------------------------------------------------------
# HTMLReport
# ---------------------------------------------------------------------------

class HTMLReport:
    """Self-contained HTML experience study report.

    Parameters
    ----------
    title :
        Report title displayed in the ``<h1>`` header.
    study_metadata :
        Dict of study-level key/value pairs shown in the header block.
        Common keys: ``'period'``, ``'table_used'``, ``'methodology'``.

    Examples
    --------
    >>> rpt = HTMLReport("Mortality Study 2022", {
    ...     "period": "2020–2022", "table_used": "IA 2012", "methodology": "CLT"
    ... })
    >>> rpt.add_ae_table(ae_df, caption="A/E by Age")
    >>> rpt.render("reports/mortality_2022.html")
    """

    def __init__(self, title: str, study_metadata: Dict[str, Any]) -> None:
        self._title    = title
        self._metadata = dict(study_metadata)
        self._sections: List[str] = []

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def add_ae_table(self, df: pd.DataFrame, caption: str = "") -> None:
        """Add an Actual vs Expected table section.

        Appends the key A/E formulas in MathJax, then the data table.
        If *df* is empty, renders a "No data" notice instead.

        Parameters
        ----------
        df :
            DataFrame to render.  Any numeric columns are formatted to
            4 decimal places.
        caption :
            Optional caption shown above the table.
        """
        parts: List[str] = ["<section>"]
        if caption:
            parts.append(f"<h2>{html.escape(caption)}</h2>")

        parts.append('<div class="formula-block">')
        parts.append(
            "A/E Ratio: "
            r"$$\text{A/E} = \frac{\sum \text{Actual claims}}{\sum \text{Expected claims}}"
            r"= \frac{\sum d_x}{\sum \hat{q}_x \cdot E_x^c}$$"
        )
        parts.append(
            "Central Exposed-to-Risk: "
            r"\(\displaystyle E_x^c = \sum_{i} \frac{\text{days in study}}{365.25}\)"
        )
        parts.append("</div>")

        if df.empty:
            parts.append('<p class="no-data">No data available.</p>')
        else:
            parts.append(self._df_to_html(df, caption=""))

        parts.append("</section>")
        self._sections.append("\n".join(parts))

    def add_graduation_table(
        self,
        crude_df: pd.DataFrame,
        graduated_df: pd.DataFrame,
    ) -> None:
        """Add a graduation results section with crude and graduated rates.

        Parameters
        ----------
        crude_df :
            Raw graduated mortality rates (e.g., from Whittaker).
        graduated_df :
            Smoothed graduated rates.
        """
        parts: List[str] = ["<section>", "<h2>Graduation Results</h2>"]

        parts.append('<div class="formula-block">')
        parts.append(
            "Initial exposed-to-risk: "
            r"\(E_x = E_x^c + \tfrac{1}{2}\,d_x\)"
            "<br>"
            r"Crude rate: \(\hat{q}_x = d_x / E_x\)"
        )
        parts.append("</div>")

        parts.append("<h3>Crude Rates</h3>")
        if crude_df.empty:
            parts.append('<p class="no-data">No crude rate data.</p>')
        else:
            parts.append(self._df_to_html(crude_df, caption="Crude mortality rates"))

        parts.append("<h3>Graduated Rates</h3>")
        if graduated_df.empty:
            parts.append('<p class="no-data">No graduated rate data.</p>')
        else:
            parts.append(self._df_to_html(graduated_df, caption="Graduated mortality rates"))

        parts.append("</section>")
        self._sections.append("\n".join(parts))

    def add_summary_stats(self, stats: Dict[str, Any]) -> None:
        """Add a summary statistics section.

        Parameters
        ----------
        stats :
            Dict of label → value pairs.  Values are rendered as-is
            (strings and numbers both supported).
        """
        parts: List[str] = ["<section>", "<h2>Summary Statistics</h2>"]

        parts.append('<div class="formula-block">')
        parts.append(
            r"Credibility-weighted rate: \(q_x^* = Z \hat{q}_x + (1-Z) q_x^{\text{std}}\)"
            "<br>"
            r"Bühlmann credibility factor: \(Z = n / (n + k)\) where \(k = \sigma^2 / \tau^2\)"
        )
        parts.append("</div>")

        parts.append('<div class="summary-grid">')
        for label, value in stats.items():
            escaped_label = html.escape(str(label))
            escaped_value = html.escape(str(value))
            parts.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">{escaped_label}</div>'
                f'<div class="stat-value">{escaped_value}</div>'
                f'</div>'
            )
        parts.append("</div></section>")
        self._sections.append("\n".join(parts))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, output_path: str) -> None:
        """Write the self-contained HTML report to *output_path*.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        output_path :
            Destination file path (e.g., ``'reports/study.html'``).
        """
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="UTF-8">',
            f'<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(self._title)}</title>",
            _CSS,
            _MATHJAX_SCRIPT,
            "</head>",
            "<body>",
            f"<h1>{html.escape(self._title)}</h1>",
            self._metadata_block(),
        ]

        html_parts.extend(self._sections)

        html_parts += ["</body>", "</html>"]

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(html_parts))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metadata_block(self) -> str:
        if not self._metadata:
            return ""
        rows = "".join(
            f"<tr><td><strong>{html.escape(str(k))}</strong></td>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in self._metadata.items()
        )
        return f'<div class="metadata"><table>{rows}</table></div>'

    @staticmethod
    def _df_to_html(df: pd.DataFrame, caption: str = "") -> str:
        """Render *df* as a styled HTML table string."""
        col_headers = "".join(
            f"<th>{html.escape(str(c))}</th>" for c in df.columns
        )
        thead = f"<thead><tr>{col_headers}</tr></thead>"

        body_rows: List[str] = []
        for _, row in df.iterrows():
            cells = []
            for val in row:
                if isinstance(val, float):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{html.escape(str(val))}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        tbody = f"<tbody>{''.join(body_rows)}</tbody>"

        cap_html = f"<caption>{html.escape(caption)}</caption>" if caption else ""
        return f'<table class="data">{cap_html}{thead}{tbody}</table>'
