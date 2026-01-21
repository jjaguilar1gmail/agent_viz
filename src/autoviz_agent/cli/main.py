"""CLI main entry point."""

import argparse
import sys
import uuid
from pathlib import Path

from autoviz_agent.config.settings import DEFAULT_SETTINGS
from autoviz_agent.graph.graph_builder import create_pipeline
from autoviz_agent.graph.state import GraphState
from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.planning.template_loader import TemplateLoader
from autoviz_agent.runtime.determinism import set_seeds
from autoviz_agent.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoViz Agent - Deterministic data visualization and analysis"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run analysis")
    run_parser.add_argument("dataset", type=Path, help="Path to dataset (CSV)")
    run_parser.add_argument("question", help="Natural language question")
    run_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output directory"
    )
    run_parser.add_argument("--config", type=Path, help="Configuration file")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        return run_analysis(args)

    return 0


def run_analysis(args) -> int:
    """
    Run analysis command.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    logger.info(f"Running analysis on {args.dataset} with question: {args.question}")

    # Set deterministic seeds
    set_seeds(args.seed)

    # Generate run ID
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Run ID: {run_id}")

    # Setup artifact manager
    artifact_manager = ArtifactManager(run_id=run_id, output_dir=args.output_dir)
    logger.info(f"Output directory: {artifact_manager.run_dir}")

    # Load templates
    templates_dir = DEFAULT_SETTINGS.runtime.templates_dir
    if not templates_dir.exists():
        logger.warning(
            f"Templates directory not found: {templates_dir}. Creating placeholder templates."
        )
        templates_dir.mkdir(parents=True, exist_ok=True)
        _create_placeholder_templates(templates_dir)

    # Create initial state
    initial_state = GraphState(
        run_id=run_id,
        dataset_path=str(args.dataset),
        question=args.question,
    )

    try:
        # Create and execute full agentic pipeline
        logger.info("Creating execution pipeline")
        pipeline = create_pipeline()

        # Create initial state
        initial_state = GraphState(
            run_id=run_id,
            dataset_path=str(args.dataset),
            question=args.question,
        )

        logger.info("Executing agentic pipeline")
        # Execute the full LangGraph pipeline
        result = pipeline.invoke(initial_state)

        logger.info(f"✅ Analysis complete!")
        logger.info(f"📄 Report: {artifact_manager.run_dir / 'report.md'}")
        logger.info(f"📁 All results: {artifact_manager.run_dir}")
        logger.info(f"📊 Observability artifacts:")
        logger.info(f"   - plan_template.json")
        logger.info(f"   - plan_adapted.json")
        logger.info(f"   - plan_diff.md")
        logger.info(f"   - tool_calls.json")
        logger.info(f"   - execution_log.json")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


def _create_placeholder_templates(templates_dir: Path) -> None:
    """Create placeholder template files for testing."""
    import json

    general_eda = {
        "template_id": "general_eda_v1",
        "version": "1.0.0",
        "intents": ["general_eda"],
        "data_shape": ["wide", "long", "unknown"],
        "requires": {"min_rows": 10, "min_columns": 2},
        "prefers": {"has_numeric": True},
        "supports": ["eda", "statistics", "visualization"],
        "steps": [
            {
                "step_id": "load_data",
                "tool": "load_dataset",
                "description": "Load dataset",
                "params": {"path": "$dataset_path"},
            },
            {
                "step_id": "infer_schema",
                "tool": "infer_schema",
                "description": "Infer schema",
                "params": {"df": "$dataset"},
            },
            {
                "step_id": "compute_stats",
                "tool": "compute_summary_stats",
                "description": "Compute summary statistics",
                "params": {"df": "$dataset"},
            },
        ],
    }

    time_series = {
        "template_id": "time_series_investigation_v1",
        "version": "1.0.0",
        "intents": ["time_series_investigation"],
        "data_shape": ["time_series"],
        "requires": {"min_rows": 10, "min_columns": 2},
        "prefers": {"has_datetime": True, "has_numeric": True},
        "supports": ["time_series", "trends", "temporal_analysis"],
        "steps": [
            {
                "step_id": "load_data",
                "tool": "load_dataset",
                "description": "Load dataset",
                "params": {"path": "$dataset_path"},
            },
            {
                "step_id": "parse_dates",
                "tool": "parse_datetime",
                "description": "Parse datetime columns",
                "params": {"df": "$dataset", "columns": ["date"]},
            },
        ],
    }

    with open(templates_dir / "general_eda.json", "w") as f:
        json.dump(general_eda, f, indent=2)

    with open(templates_dir / "time_series_investigation.json", "w") as f:
        json.dump(time_series, f, indent=2)

    logger.info("Created placeholder templates")


if __name__ == "__main__":
    sys.exit(main())
