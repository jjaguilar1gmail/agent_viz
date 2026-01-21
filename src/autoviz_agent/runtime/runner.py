"""Run runner entrypoint to expose run metadata."""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from autoviz_agent.config.settings import Settings
from autoviz_agent.graph.graph_builder import create_pipeline
from autoviz_agent.graph.state import GraphState
from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.models.state import RunState, RunStatus
from autoviz_agent.planning.template_loader import TemplateLoader
from autoviz_agent.runtime.determinism import set_seeds
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class RunRunner:
    """High-level runner for analysis runs."""

    def __init__(self, settings: Settings):
        """
        Initialize run runner.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.template_loader = TemplateLoader(settings.runtime.templates_dir)

    def run(
        self,
        dataset_path: Path,
        question: str,
        output_dir: Optional[Path] = None,
        seed: Optional[int] = 42,
    ) -> RunState:
        """
        Execute a complete analysis run.

        Args:
            dataset_path: Path to dataset
            question: User question
            output_dir: Output directory (None to use settings default)
            seed: Random seed

        Returns:
            RunState with results and metadata
        """
        # Set deterministic seeds
        set_seeds(seed)

        # Generate run ID
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting run: {run_id}")

        # Setup artifact manager
        if output_dir is None:
            output_dir = self.settings.runtime.output_dir

        artifact_manager = ArtifactManager(run_id=run_id, output_dir=output_dir)

        # Load templates
        self.template_loader.load_all()

        # Create initial state
        initial_state = GraphState(
            run_id=run_id,
            dataset_path=str(dataset_path),
            question=question,
        )

        # Create pipeline
        pipeline = create_pipeline()

        try:
            # Execute pipeline (placeholder - will be fully wired in integration)
            logger.info("Executing pipeline")
            # result = pipeline.invoke(initial_state)

            # For now, create a placeholder run state
            run_state = RunState(
                run_id=run_id,
                user_request={"question": question},
                dataset_source={"source_type": "csv", "path": str(dataset_path)},
                status=RunStatus.COMPLETED,
            )

            logger.info(f"Run completed: {run_id}")
            return run_state

        except Exception as e:
            logger.error(f"Run failed: {e}", exc_info=True)

            run_state = RunState(
                run_id=run_id,
                user_request={"question": question},
                dataset_source={"source_type": "csv", "path": str(dataset_path)},
                status=RunStatus.FAILED,
                error_message=str(e),
            )

            return run_state

    def get_run_metadata(self, run_id: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Get metadata for a completed run.

        Args:
            run_id: Run identifier
            output_dir: Output directory

        Returns:
            Run metadata dictionary
        """
        if output_dir is None:
            output_dir = self.settings.runtime.output_dir

        artifact_manager = ArtifactManager(run_id=run_id, output_dir=output_dir)

        try:
            metadata = artifact_manager.load_json("metadata", "run_metadata.json")
            return metadata
        except FileNotFoundError:
            logger.warning(f"No metadata found for run: {run_id}")
            return {"run_id": run_id, "status": "unknown"}
