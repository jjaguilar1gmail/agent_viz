"""Template loader and validator."""

import json
from pathlib import Path
from typing import Any, Dict, List

import jsonschema

from autoviz_agent.planning.template_schema import PLAN_TEMPLATE_SCHEMA
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateLoader:
    """Load and validate plan templates."""

    def __init__(self, templates_dir: Path):
        """
        Initialize template loader.

        Args:
            templates_dir: Directory containing template JSON files
        """
        self.templates_dir = templates_dir
        self._templates: Dict[str, Dict[str, Any]] = {}

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all templates from the templates directory.

        Returns:
            Dictionary of template_id to template data
        """
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_dir}")
            return {}

        for template_file in self.templates_dir.glob("*.json"):
            try:
                template = self.load_template(template_file)
                template_id = template["template_id"]
                self._templates[template_id] = template
                logger.info(f"Loaded template: {template_id}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

        return self._templates

    def load_template(self, path: Path) -> Dict[str, Any]:
        """
        Load and validate a single template.

        Args:
            path: Path to template JSON file

        Returns:
            Validated template data

        Raises:
            jsonschema.ValidationError: If template is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        with open(path, "r", encoding="utf-8") as f:
            template = json.load(f)

        # Validate against schema
        jsonschema.validate(instance=template, schema=PLAN_TEMPLATE_SCHEMA)

        return template

    def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get a loaded template by ID.

        Args:
            template_id: Template identifier

        Returns:
            Template data

        Raises:
            KeyError: If template not found
        """
        if not self._templates:
            self.load_all()

        return self._templates[template_id]

    def list_templates(self) -> List[str]:
        """
        List all loaded template IDs.

        Returns:
            List of template IDs
        """
        if not self._templates:
            self.load_all()

        return list(self._templates.keys())
