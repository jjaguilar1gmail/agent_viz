"""Artifact path manager and persistence helpers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ArtifactManager:
    """Manage artifact paths and persistence."""

    def __init__(self, run_id: str, output_dir: Path):
        """
        Initialize artifact manager.

        Args:
            run_id: Unique run identifier
            output_dir: Base output directory
        """
        self.run_id = run_id
        self.output_dir = output_dir
        self.run_dir = output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, artifact_type: str, filename: str) -> Path:
        """
        Get path for an artifact.

        Args:
            artifact_type: Type of artifact (chart, report, plan, log, etc.)
            filename: Filename

        Returns:
            Full path to artifact
        """
        # Create subdirectories for organization
        if artifact_type == "chart":
            subdir = self.run_dir / "charts"
        elif artifact_type in ["plan_template", "plan_adapted", "plan_diff"]:
            subdir = self.run_dir / "plans"
        elif artifact_type in ["execution_log", "tool_calls", "llm_requests"]:
            subdir = self.run_dir / "logs"
        else:
            subdir = self.run_dir

        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / filename

    def save_json(self, data: Dict[str, Any], artifact_type: str, filename: str) -> Path:
        """
        Save JSON artifact.

        Args:
            data: Data to save
            artifact_type: Type of artifact
            filename: Filename (should end with .json)

        Returns:
            Path to saved file
        """
        path = self.get_path(artifact_type, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON artifact: {path}")
        return path

    def save_text(self, content: str, artifact_type: str, filename: str) -> Path:
        """
        Save text artifact.

        Args:
            content: Text content
            artifact_type: Type of artifact
            filename: Filename

        Returns:
            Path to saved file
        """
        path = self.get_path(artifact_type, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved text artifact: {path}")
        return path

    def load_json(self, artifact_type: str, filename: str) -> Dict[str, Any]:
        """
        Load JSON artifact.

        Args:
            artifact_type: Type of artifact
            filename: Filename

        Returns:
            Loaded data
        """
        path = self.get_path(artifact_type, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_text(self, artifact_type: str, filename: str) -> str:
        """
        Load text artifact.

        Args:
            artifact_type: Type of artifact
            filename: Filename

        Returns:
            File content
        """
        path = self.get_path(artifact_type, filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def list_artifacts(self) -> list[Path]:
        """
        List all artifacts in the run directory.

        Returns:
            List of artifact paths
        """
        return list(self.run_dir.rglob("*.*"))

    def save_plan_provenance(
        self,
        template_plan: Dict[str, Any],
        adapted_plan: Dict[str, Any],
        plan_diff: str,
    ) -> Dict[str, Path]:
        """
        Save plan provenance artifacts.

        Args:
            template_plan: Original template plan
            adapted_plan: Adapted plan
            plan_diff: Plan diff markdown

        Returns:
            Dictionary of artifact paths
        """
        paths = {}

        # Save template plan
        paths["template"] = self.save_json(template_plan, "plan_template", "plan_template.json")

        # Save adapted plan
        paths["adapted"] = self.save_json(adapted_plan, "plan_adapted", "plan_adapted.json")

        # Save plan diff
        paths["diff"] = self.save_text(plan_diff, "plan_diff", "plan_diff.md")

        logger.info("Saved plan provenance artifacts")
        return paths
