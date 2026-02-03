"""Report writer to produce report.md."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ReportWriter:
    """Generate analysis report in markdown format."""

    def __init__(self):
        """Initialize report writer."""
        self.sections: List[str] = []

    def add_header(self, title: str, level: int = 1) -> None:
        """
        Add a header to the report.

        Args:
            title: Header text
            level: Header level (1-6)
        """
        prefix = "#" * level
        self.sections.append(f"{prefix} {title}\n")

    def add_text(self, text: str) -> None:
        """
        Add text paragraph.

        Args:
            text: Text content
        """
        self.sections.append(f"{text}\n")

    def add_list(self, items: List[str], ordered: bool = False) -> None:
        """
        Add a list.

        Args:
            items: List items
            ordered: Whether to use ordered list
        """
        for idx, item in enumerate(items):
            if ordered:
                self.sections.append(f"{idx + 1}. {item}\n")
            else:
                self.sections.append(f"- {item}\n")
        self.sections.append("\n")

    def add_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """
        Add a table.

        Args:
            headers: Table headers
            rows: Table rows
        """
        # Header row
        self.sections.append("| " + " | ".join(headers) + " |\n")
        # Separator
        self.sections.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        # Data rows
        for row in rows:
            self.sections.append("| " + " | ".join(str(cell) for cell in row) + " |\n")
        self.sections.append("\n")

    def add_code_block(self, code: str, language: str = "") -> None:
        """
        Add a code block.

        Args:
            code: Code content
            language: Language for syntax highlighting
        """
        self.sections.append(f"```{language}\n{code}\n```\n")

    def add_chart_reference(self, chart_path: Path, caption: Optional[str] = None) -> None:
        """
        Add a reference to a chart.

        Args:
            chart_path: Path to chart file
            caption: Optional caption
        """
        self.sections.append(f"![{caption or chart_path.stem}]({chart_path})\n")

    def add_summary_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Add summary statistics section.

        Args:
            stats: Dictionary of statistics
        """
        self.add_header("Summary Statistics", level=2)

        for col, col_stats in stats.items():
            self.add_header(col, level=3)
            self.add_list([f"{k}: {v}" for k, v in col_stats.items()])

    def add_llm_interactions_section(self, interactions: List[Dict[str, Any]]) -> None:
        """
        Add LLM interactions log section.

        Args:
            interactions: List of LLM interaction records
        """
        if not interactions:
            return
            
        self.add_header("LLM Interactions", level=2)
        self.add_text(
            "This section shows all interactions with the language model during analysis."
        )
        
        for interaction in interactions:
            step_name = interaction.get('step', 'unknown').replace('_', ' ').title()
            self.add_header(step_name, level=3)
            
            # Input
            input_data = interaction.get('input')
            if isinstance(input_data, str):
                self.add_text(f"**Question**: {input_data}")
            elif isinstance(input_data, dict):
                self.add_text("**Input**:")
                self.add_list([f"{k}: {v}" for k, v in input_data.items()])
            
            # Output
            output_data = interaction.get('output', {})
            if output_data:
                self.add_text("**Output**:")
                if isinstance(output_data, dict):
                    self.add_list([f"{k}: {v}" for k, v in output_data.items()])
                else:
                    self.add_text(str(output_data))
            
            self.add_text("")  # Empty line for spacing

    def add_key_metrics_section(self, execution_results: List[Dict[str, Any]]) -> None:
        """
        Add key metrics derived from non-chart tools.

        Args:
            execution_results: List of execution results from tools
        """
        def is_scalar(value: Any) -> bool:
            return isinstance(value, (int, float, str, bool)) or value is None

        def scalar_items(data: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in data.items() if is_scalar(v)}

        dataframe_outputs = []
        metric_sections = []

        for result in execution_results:
            if not result.get("success"):
                continue

            tool = result.get("tool", "unknown")
            data = result.get("result", {})

            if isinstance(data, dict) and data.get("type") == "dataframe":
                dataframe_outputs.append({
                    "tool": tool,
                    "shape": data.get("shape"),
                    "columns": data.get("columns"),
                })
                continue

            metric_sections.append({"tool": tool, "data": data})

        if not metric_sections and not dataframe_outputs:
            return

        self.add_header("Key Metrics", level=2)

        for item in metric_sections:
            tool = item["tool"]
            data = item["data"]

            if isinstance(data, dict):
                nested_dicts = {
                    k: v for k, v in data.items()
                    if isinstance(v, dict) and scalar_items(v)
                }
                if nested_dicts:
                    self.add_header(f"{tool} (summary)", level=3)
                    inner_keys = sorted({key for v in nested_dicts.values() for key in v.keys()})
                    rows = []
                    for outer_key, values in nested_dicts.items():
                        row = [outer_key] + [values.get(k, "") for k in inner_keys]
                        rows.append(row)
                    self.add_table(["item"] + inner_keys, rows)
                    continue

                scalars = scalar_items(data)
                if scalars:
                    self.add_header(f"{tool} (metrics)", level=3)
                    rows = [[k, v] for k, v in scalars.items()]
                    self.add_table(["metric", "value"], rows)
                    continue

                if data:
                    self.add_header(f"{tool} (details)", level=3)
                    self.add_text(str(data))
                continue

            if isinstance(data, list) and data:
                if all(isinstance(item, dict) for item in data):
                    self.add_header(f"{tool} (table)", level=3)
                    headers = sorted({key for row in data for key in row.keys()})
                    rows = [[row.get(h, "") for h in headers] for row in data]
                    self.add_table(headers, rows)
                else:
                    self.add_header(f"{tool} (list)", level=3)
                    self.add_list([str(item) for item in data])
                continue

            if data not in (None, {}, []):
                self.add_header(f"{tool} (value)", level=3)
                self.add_text(str(data))

        if dataframe_outputs:
            self.add_header("DataFrame Outputs", level=3)
            rows = []
            for item in dataframe_outputs:
                shape = item.get("shape") or ""
                columns = item.get("columns") or []
                rows.append([
                    item.get("tool", ""),
                    f"{shape[0]}x{shape[1]}" if shape else "",
                    ", ".join(columns),
                ])
            self.add_table(["tool", "shape", "columns"], rows)
    
    def add_charts_section(self, execution_results: List[Dict[str, Any]]) -> None:
        """
        Add generated charts section with links.

        Args:
            execution_results: List of execution results from tools
        """
        # Extract chart paths from results
        charts = []
        for result in execution_results:
            if not result.get('success'):
                continue
            
            result_data = result.get('result', {})
            chart_path = None
            
            # Check for file type result
            if result_data.get('type') == 'file':
                chart_path = result_data.get('path')
            # Check for value field (string path)
            elif 'value' in result_data:
                value = result_data['value']
                if isinstance(value, str) and any(ext in value for ext in ['.png', '.jpg', '.svg', '.pdf']):
                    chart_path = value
            
            if chart_path:
                charts.append({
                    'tool': result.get('tool', 'unknown'),
                    'path': Path(chart_path)
                })
        
        if not charts:
            return
        
        self.add_header("Generated Visualizations", level=2)
        self.add_text(f"This analysis generated {len(charts)} visualization(s):")
        self.add_text("")  # Empty line
        
        for i, chart in enumerate(charts, 1):
            # Relative path from report location (both in same run directory)
            rel_path = Path('charts') / chart['path'].name
            self.add_header(f"Figure {i}: {chart['tool']}", level=3)
            self.add_chart_reference(rel_path, caption=f"{chart['tool']} visualization")
            self.add_text("")  # Empty line for spacing
    
    def add_provenance_section(
        self,
        template_path: Optional[Path] = None,
        adapted_path: Optional[Path] = None,
        diff_path: Optional[Path] = None,
        tool_calls_path: Optional[Path] = None,
        execution_log_path: Optional[Path] = None,
    ) -> None:
        """
        Add plan provenance section with artifact links.

        Args:
            template_path: Path to template plan
            adapted_path: Path to adapted plan
            diff_path: Path to plan diff
            tool_calls_path: Path to tool calls
            execution_log_path: Path to execution log
        """
        self.add_header("Plan Provenance", level=2)

        self.add_text(
            "This section provides full traceability for the analysis plan and execution."
        )
        self.add_text(
            "The plan selection was based on schema matching and intent classification, "
            "and any adaptations are documented with rationale."
        )

        artifacts = []
        if template_path:
            artifacts.append(f"**Template Plan**: [{template_path.name}]({template_path})")
        if adapted_path:
            artifacts.append(f"**Adapted Plan**: [{adapted_path.name}]({adapted_path})")
        if diff_path:
            artifacts.append(f"**Plan Diff**: [{diff_path.name}]({diff_path})")
        if tool_calls_path:
            artifacts.append(f"**Tool Calls**: [{tool_calls_path.name}]({tool_calls_path})")
        if execution_log_path:
            artifacts.append(
                f"**Execution Log**: [{execution_log_path.name}]({execution_log_path})"
            )

        if artifacts:
            self.add_list(artifacts)

    def write(self, output_path: Path) -> Path:
        """
        Write report to file.

        Args:
            output_path: Path to output file

        Returns:
            Path to written file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(self.sections))

        logger.info(f"Wrote report to {output_path}")
        return output_path

    def to_string(self) -> str:
        """
        Get report as string.

        Returns:
            Report content
        """
        return "".join(self.sections)
