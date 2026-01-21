"""Deterministic run verification script."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from autoviz_agent.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def compute_file_hash(path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        path: File path

    Returns:
        Hex digest of hash
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def collect_artifacts(run_dir: Path) -> Dict[str, str]:
    """
    Collect artifacts and compute hashes.

    Args:
        run_dir: Run directory

    Returns:
        Dictionary of relative_path -> hash
    """
    artifacts = {}

    for file_path in run_dir.rglob("*.*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(run_dir)
            file_hash = compute_file_hash(file_path)
            artifacts[str(relative_path)] = file_hash

    return artifacts


def compare_runs(run1_dir: Path, run2_dir: Path) -> bool:
    """
    Compare two runs for determinism.

    Args:
        run1_dir: First run directory
        run2_dir: Second run directory

    Returns:
        True if runs are identical, False otherwise
    """
    logger.info(f"Comparing runs: {run1_dir} vs {run2_dir}")

    # Collect artifacts
    artifacts1 = collect_artifacts(run1_dir)
    artifacts2 = collect_artifacts(run2_dir)

    # Compare artifact lists
    files1 = set(artifacts1.keys())
    files2 = set(artifacts2.keys())

    if files1 != files2:
        missing_in_run2 = files1 - files2
        missing_in_run1 = files2 - files1

        if missing_in_run2:
            logger.error(f"Files in run1 but not run2: {missing_in_run2}")
        if missing_in_run1:
            logger.error(f"Files in run2 but not run1: {missing_in_run1}")

        return False

    # Compare hashes
    differences = []
    for file_path in sorted(files1):
        hash1 = artifacts1[file_path]
        hash2 = artifacts2[file_path]

        if hash1 != hash2:
            differences.append(file_path)
            logger.error(f"Hash mismatch: {file_path}")
            logger.error(f"  Run1: {hash1}")
            logger.error(f"  Run2: {hash2}")

    if differences:
        logger.error(f"Total differences: {len(differences)}")
        return False

    logger.info("✅ Runs are identical (deterministic)")
    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify deterministic run outputs")
    parser.add_argument("run1_dir", type=Path, help="First run directory")
    parser.add_argument("run2_dir", type=Path, help="Second run directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if not args.run1_dir.exists():
        logger.error(f"Run1 directory not found: {args.run1_dir}")
        return 1

    if not args.run2_dir.exists():
        logger.error(f"Run2 directory not found: {args.run2_dir}")
        return 1

    # Compare runs
    is_deterministic = compare_runs(args.run1_dir, args.run2_dir)

    return 0 if is_deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
