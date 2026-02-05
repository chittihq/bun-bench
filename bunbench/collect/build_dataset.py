"""
Dataset building utilities for Bun-Bench.

This module provides classes and functions for creating, validating,
and managing benchmark instances for the Bun-Bench dataset.
"""

import json
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkInstance:
    """
    Represents a single benchmark instance in Bun-Bench.

    A benchmark instance contains all information needed to:
    1. Present a problem to an LLM
    2. Evaluate the LLM's generated patch
    3. Verify correctness via testing
    """

    # Required fields
    instance_id: str
    problem_statement: str
    category: str  # e.g., "bug_fix", "feature", "performance"

    # Repository information
    repo: str = "oven-sh/bun"
    base_commit: str = ""
    version: str = ""

    # Code context (optional, for providing relevant code snippets)
    code_context: str = ""
    relevant_files: List[str] = field(default_factory=list)

    # Ground truth (for evaluation)
    gold_patch: str = ""
    test_patch: str = ""
    pass_to_pass_tests: List[str] = field(default_factory=list)
    fail_to_pass_tests: List[str] = field(default_factory=list)

    # Metadata
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = ""  # e.g., "github_issue", "manual", "synthetic"
    source_url: str = ""

    # Additional context
    hints: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        """Validate instance after initialization."""
        if not self.instance_id:
            raise ValueError("instance_id is required")
        if not self.problem_statement:
            raise ValueError("problem_statement is required")
        if not self.category:
            raise ValueError("category is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkInstance":
        """Create instance from dictionary."""
        return cls(**data)

    def compute_id(self) -> str:
        """
        Compute a deterministic ID based on content hash.

        Returns:
            SHA256 hash of problem statement (first 12 chars).
        """
        content = f"{self.repo}:{self.problem_statement}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# Valid categories for instances
VALID_CATEGORIES = {
    "bug_fix",
    "feature",
    "performance",
    "refactor",
    "security",
    "documentation",
    "test",
}

# Valid difficulty levels
VALID_DIFFICULTIES = {"easy", "medium", "hard", "expert"}

# Required fields for a complete instance
REQUIRED_FIELDS = {"instance_id", "problem_statement", "category"}

# Fields that should be non-empty for evaluation
EVALUATION_FIELDS = {"gold_patch", "fail_to_pass_tests"}


def validate_instance(
    instance: Dict[str, Any],
    strict: bool = False,
) -> tuple[bool, List[str]]:
    """
    Validate a benchmark instance.

    Args:
        instance: Instance dictionary to validate.
        strict: If True, require evaluation fields.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    # Check required fields
    for field_name in REQUIRED_FIELDS:
        if field_name not in instance or not instance[field_name]:
            issues.append(f"Missing required field: {field_name}")

    # Validate category
    category = instance.get("category", "")
    if category and category not in VALID_CATEGORIES:
        issues.append(
            f"Invalid category: {category}. Must be one of: {VALID_CATEGORIES}"
        )

    # Validate difficulty
    difficulty = instance.get("difficulty", "medium")
    if difficulty not in VALID_DIFFICULTIES:
        issues.append(
            f"Invalid difficulty: {difficulty}. Must be one of: {VALID_DIFFICULTIES}"
        )

    # Validate instance_id format
    instance_id = instance.get("instance_id", "")
    if instance_id:
        if not re.match(r"^[A-Z]+-\d+$", instance_id) and not re.match(
            r"^[a-z0-9_-]+$", instance_id
        ):
            issues.append(
                f"Invalid instance_id format: {instance_id}. "
                "Use 'PREFIX-NUMBER' or lowercase alphanumeric with underscores."
            )

    # Strict mode: check evaluation fields
    if strict:
        for field_name in EVALUATION_FIELDS:
            value = instance.get(field_name)
            if not value or (isinstance(value, list) and len(value) == 0):
                issues.append(f"Missing evaluation field for strict mode: {field_name}")

        # Validate gold_patch format
        gold_patch = instance.get("gold_patch", "")
        if gold_patch:
            if not _is_valid_patch(gold_patch):
                issues.append("gold_patch does not appear to be a valid unified diff")

    # Validate lists are actually lists
    list_fields = [
        "relevant_files",
        "pass_to_pass_tests",
        "fail_to_pass_tests",
        "tags",
        "hints",
    ]
    for field_name in list_fields:
        value = instance.get(field_name)
        if value is not None and not isinstance(value, list):
            issues.append(f"Field {field_name} must be a list")

    return len(issues) == 0, issues


def _is_valid_patch(patch: str) -> bool:
    """Check if string appears to be a valid unified diff."""
    indicators = [
        r"^---\s+\S+",
        r"^\+\+\+\s+\S+",
        r"^@@\s+-\d+",
    ]
    for pattern in indicators:
        if re.search(pattern, patch, re.MULTILINE):
            return True
    return False


def create_instance(
    problem_statement: str,
    category: str,
    instance_id: Optional[str] = None,
    **kwargs,
) -> BenchmarkInstance:
    """
    Create a new benchmark instance.

    Args:
        problem_statement: Description of the bug/feature.
        category: Instance category (bug_fix, feature, etc.).
        instance_id: Optional custom ID (auto-generated if not provided).
        **kwargs: Additional instance fields.

    Returns:
        BenchmarkInstance object.

    Raises:
        ValueError: If validation fails.
    """
    # Generate ID if not provided
    if not instance_id:
        # Generate from hash of problem statement
        hash_val = hashlib.sha256(problem_statement.encode()).hexdigest()[:8]
        prefix = category.upper().replace("_", "")[:3]
        instance_id = f"{prefix}-{hash_val}"

    instance = BenchmarkInstance(
        instance_id=instance_id,
        problem_statement=problem_statement,
        category=category,
        **kwargs,
    )

    # Validate
    is_valid, issues = validate_instance(instance.to_dict())
    if not is_valid:
        raise ValueError(f"Invalid instance: {issues}")

    return instance


class DatasetBuilder:
    """
    Builder for creating and managing Bun-Bench datasets.

    Provides methods for adding instances, validating the dataset,
    and exporting to various formats.
    """

    def __init__(self, name: str = "bun-bench", version: str = "1.0.0"):
        """
        Initialize dataset builder.

        Args:
            name: Dataset name.
            version: Dataset version.
        """
        self.name = name
        self.version = version
        self.instances: List[BenchmarkInstance] = []
        self._instance_ids: Set[str] = set()
        self.metadata: Dict[str, Any] = {
            "name": name,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "description": "Bun runtime benchmark dataset for LLM code generation evaluation",
        }

    def add_instance(self, instance: BenchmarkInstance) -> bool:
        """
        Add an instance to the dataset.

        Args:
            instance: BenchmarkInstance to add.

        Returns:
            True if added, False if duplicate ID.
        """
        if instance.instance_id in self._instance_ids:
            logger.warning(f"Duplicate instance ID: {instance.instance_id}")
            return False

        self.instances.append(instance)
        self._instance_ids.add(instance.instance_id)
        return True

    def add_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Add an instance from a dictionary.

        Args:
            data: Instance data dictionary.

        Returns:
            True if added, False if invalid or duplicate.
        """
        is_valid, issues = validate_instance(data)
        if not is_valid:
            logger.error(f"Invalid instance data: {issues}")
            return False

        instance = BenchmarkInstance.from_dict(data)
        return self.add_instance(instance)

    def load_from_file(self, path: str) -> int:
        """
        Load instances from a JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Number of instances loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict):
            if "instances" in data:
                instances = data["instances"]
                # Update metadata if present
                if "metadata" in data:
                    self.metadata.update(data["metadata"])
            else:
                instances = [data]  # Single instance
        else:
            raise ValueError("Invalid JSON format")

        loaded = 0
        for inst_data in instances:
            if self.add_from_dict(inst_data):
                loaded += 1

        logger.info(f"Loaded {loaded} instances from {path}")
        return loaded

    def validate_all(self, strict: bool = False) -> tuple[bool, Dict[str, List[str]]]:
        """
        Validate all instances in the dataset.

        Args:
            strict: If True, require evaluation fields.

        Returns:
            Tuple of (all_valid, dict of instance_id -> issues).
        """
        all_issues = {}

        for instance in self.instances:
            is_valid, issues = validate_instance(instance.to_dict(), strict=strict)
            if not is_valid:
                all_issues[instance.instance_id] = issues

        return len(all_issues) == 0, all_issues

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with various statistics.
        """
        stats = {
            "total_instances": len(self.instances),
            "by_category": {},
            "by_difficulty": {},
            "with_gold_patch": 0,
            "with_tests": 0,
        }

        for instance in self.instances:
            # Category counts
            cat = instance.category
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Difficulty counts
            diff = instance.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # Gold patch presence
            if instance.gold_patch:
                stats["with_gold_patch"] += 1

            # Test presence
            if instance.fail_to_pass_tests or instance.pass_to_pass_tests:
                stats["with_tests"] += 1

        return stats

    def export_json(self, path: str, include_metadata: bool = True):
        """
        Export dataset to JSON file.

        Args:
            path: Output file path.
            include_metadata: Include dataset metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        instances_data = [inst.to_dict() for inst in self.instances]

        if include_metadata:
            output = {
                "metadata": self.metadata,
                "instances": instances_data,
            }
        else:
            output = instances_data

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self.instances)} instances to {path}")

    def export_jsonl(self, path: str):
        """
        Export dataset to JSONL file (one instance per line).

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for instance in self.instances:
                f.write(json.dumps(instance.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(self.instances)} instances to {path}")

    def filter_by(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        has_gold_patch: Optional[bool] = None,
    ) -> List[BenchmarkInstance]:
        """
        Filter instances by criteria.

        Args:
            category: Filter by category.
            difficulty: Filter by difficulty.
            tags: Filter by tags (any match).
            has_gold_patch: Filter by gold patch presence.

        Returns:
            List of matching instances.
        """
        results = self.instances.copy()

        if category:
            results = [i for i in results if i.category == category]

        if difficulty:
            results = [i for i in results if i.difficulty == difficulty]

        if tags:
            tag_set = set(tags)
            results = [i for i in results if tag_set.intersection(set(i.tags))]

        if has_gold_patch is not None:
            if has_gold_patch:
                results = [i for i in results if i.gold_patch]
            else:
                results = [i for i in results if not i.gold_patch]

        return results

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, instance_id: str) -> Optional[BenchmarkInstance]:
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance
        return None


def main():
    """CLI for dataset building operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and manage Bun-Bench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset")
    validate_parser.add_argument("input", help="Input JSON file")
    validate_parser.add_argument(
        "--strict", action="store_true", help="Require evaluation fields"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("input", help="Input JSON file")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert dataset format")
    convert_parser.add_argument("input", help="Input JSON file")
    convert_parser.add_argument("output", help="Output file")
    convert_parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "validate":
        builder = DatasetBuilder()
        builder.load_from_file(args.input)

        is_valid, issues = builder.validate_all(strict=args.strict)

        if is_valid:
            print(f"Dataset is valid! ({len(builder)} instances)")
        else:
            print(f"Dataset has {len(issues)} invalid instances:")
            for instance_id, instance_issues in issues.items():
                print(f"\n  {instance_id}:")
                for issue in instance_issues:
                    print(f"    - {issue}")
            raise SystemExit(1)

    elif args.command == "stats":
        builder = DatasetBuilder()
        builder.load_from_file(args.input)

        stats = builder.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"\n  By category:")
        for cat, count in sorted(stats["by_category"].items()):
            print(f"    {cat}: {count}")
        print(f"\n  By difficulty:")
        for diff, count in sorted(stats["by_difficulty"].items()):
            print(f"    {diff}: {count}")
        print(f"\n  With gold patch: {stats['with_gold_patch']}")
        print(f"  With tests: {stats['with_tests']}")

    elif args.command == "convert":
        builder = DatasetBuilder()
        builder.load_from_file(args.input)

        if args.format == "jsonl":
            builder.export_jsonl(args.output)
        else:
            builder.export_json(args.output)

        print(f"Converted {len(builder)} instances to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
