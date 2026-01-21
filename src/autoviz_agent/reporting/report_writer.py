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
