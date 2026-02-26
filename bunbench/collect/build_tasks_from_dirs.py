#!/usr/bin/env python3
"""
Scan dataset/tasks/ directories and generate tasks_from_dirs.json.

Each task directory should contain:
  - README.md with problem description
  - src/ with buggy code
  - test/ with test files
  - solution/ with fixed code
"""

import json
import os
import re
from pathlib import Path


# Component mapping based on task name patterns
COMPONENT_MAP = {
    "content-length": "http",
    "json-content-type": "http",
    "body-parsing": "http",
    "route-params": "http",
    "error-status": "http",
    "sqlite": "sqlite",
    "file-encoding": "filesystem",
    "file-write": "filesystem",
    "file-exists": "filesystem",
    "spawn": "spawn",
    "test-async": "test",
    "mock-cleanup": "test",
    "expect-type": "test",
    "test-timeout": "test",
    "describe-scope": "test",
    "ws-": "websocket",
    "fetch-": "fetch",
    "build-": "build",
    "password": "crypto",
    "hash": "crypto",
    "uuid": "crypto",
    "hmac": "crypto",
    "env-": "env",
    "import-meta": "env",
    "shell": "shell",
    "cli-args": "shell",
    "tcp-": "network",
    "udp-": "network",
    "redis-": "redis",
    "sql-": "sql",
    "stream": "streams",
    "html-": "html",
    "glob-": "glob",
    "worker-": "worker",
    "cookie-": "cookie",
    "semver-": "semver",
    "snapshot-": "test",
    "compile-": "compile",
}

# Difficulty mapping based on task number ranges
def get_difficulty(task_num: int) -> str:
    if task_num <= 5:
        return "easy"
    elif task_num <= 10:
        return "medium"
    elif task_num <= 20:
        return "easy"
    elif task_num <= 35:
        return "medium"
    elif task_num <= 50:
        return "easy"
    elif task_num <= 60:
        return "medium"
    elif task_num <= 80:
        return "easy"
    return "medium"


def get_component(task_name: str) -> str:
    """Determine component from task name."""
    for pattern, component in COMPONENT_MAP.items():
        if pattern in task_name:
            return component
    return "other"


def extract_problem_statement(readme_path: Path) -> str:
    """Extract problem statement from README.md."""
    if not readme_path.exists():
        return ""

    text = readme_path.read_text()

    # Try to extract from "Problem Description" section
    match = re.search(
        r"##\s*Problem\s*Description\s*\n+(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        desc = match.group(1).strip()
        # Clean up markdown formatting
        desc = re.sub(r"\n+", " ", desc)
        desc = re.sub(r"\s+", " ", desc)
        return desc.strip()

    # Fallback: extract from "Bug Details" section
    match = re.search(
        r"##\s*Bug\s*Details\s*\n+(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        desc = match.group(1).strip()
        desc = re.sub(r"\n+", " ", desc)
        desc = re.sub(r"\s+", " ", desc)
        return desc.strip()

    # Fallback: use first paragraph after the title
    lines = text.split("\n")
    paragraphs = []
    in_paragraph = False
    for line in lines[1:]:  # Skip title
        stripped = line.strip()
        if stripped.startswith("#"):
            if paragraphs:
                break
            continue
        if stripped:
            in_paragraph = True
            paragraphs.append(stripped)
        elif in_paragraph:
            break

    if paragraphs:
        return " ".join(paragraphs)

    return f"Fix the bug in {readme_path.parent.name}"


def extract_code_context(task_dir: Path) -> str:
    """Extract code context from source files."""
    src_dir = task_dir / "src"
    if not src_dir.exists():
        return ""

    code_parts = []
    for ts_file in sorted(src_dir.glob("*.ts")):
        content = ts_file.read_text()
        code_parts.append(f"// File: {ts_file.relative_to(task_dir)}\n{content}")

    return "\n".join(code_parts)


def build_tasks_from_dirs(tasks_dir: str, output_path: str):
    """Scan task directories and generate tasks JSON."""
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        print(f"Error: {tasks_dir} does not exist")
        return

    tasks = []

    for task_dir in sorted(tasks_path.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue

        task_name = task_dir.name
        # Extract task number
        num_match = re.match(r"task-(\d+)", task_name)
        task_num = int(num_match.group(1)) if num_match else 0

        # Read README
        readme_path = task_dir / "README.md"
        problem_statement = extract_problem_statement(readme_path)

        # Extract code context
        code_context = extract_code_context(task_dir)

        # Build task entry
        task_entry = {
            "instance_id": task_name,
            "task_id": task_name,
            "task_dir": f"dataset/tasks/{task_name}",
            "problem_statement": problem_statement,
            "category": "bug_fix",
            "component": get_component(task_name),
            "difficulty": get_difficulty(task_num),
        }

        # Add code_context if available and not too long
        if code_context and len(code_context) < 10000:
            task_entry["code_context"] = code_context

        tasks.append(task_entry)

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(tasks)} task entries in {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build tasks_from_dirs.json")
    parser.add_argument(
        "--tasks-dir",
        default="dataset/tasks",
        help="Path to tasks directory",
    )
    parser.add_argument(
        "--output",
        default="dataset/tasks_from_dirs.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    build_tasks_from_dirs(args.tasks_dir, args.output)
