"""
Bun-Bench CLI - Command line interface for the Bun benchmark suite.

This module provides the main entry point for the bun-bench command.
"""

import argparse
import sys
from pathlib import Path

from bunbench import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="bun-bench",
        description="Benchmark suite for evaluating AI code generation on Bun runtime tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bun-bench evaluate --dataset data/tasks.json --predictions results.json
  bun-bench inference --dataset data/tasks.json --provider anthropic --model claude-sonnet-4-20250514
  bun-bench build-images --bun-version 1.0.0
  bun-bench report results/evaluation_report.json

For more information, visit: https://github.com/bun-bench/bun-bench
        """,
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on model predictions",
        description="Evaluate model-generated patches against the benchmark test suite",
    )
    eval_parser.add_argument(
        "-d", "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file",
    )
    eval_parser.add_argument(
        "-p", "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSON file",
    )
    eval_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)",
    )
    eval_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    eval_parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout per instance in seconds (default: 300)",
    )
    eval_parser.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for this evaluation run",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Inference command
    inf_parser = subparsers.add_parser(
        "inference",
        help="Run inference to generate predictions",
        description="Generate patches using LLM APIs",
    )
    inf_parser.add_argument(
        "-d", "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file",
    )
    inf_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path for predictions JSONL file",
    )
    inf_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    inf_parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name (default depends on provider)",
    )
    inf_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    inf_parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens in response (default: 4096)",
    )
    inf_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Build images command
    build_parser = subparsers.add_parser(
        "build-images",
        help="Build Docker images for evaluation",
        description="Pre-build Docker images for faster evaluation",
    )
    build_parser.add_argument(
        "--bun-version",
        type=str,
        help="Specific Bun version to build (default: latest)",
    )
    build_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without using cache",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="View or compare evaluation reports",
        description="Display evaluation results and compare runs",
    )
    report_parser.add_argument(
        "report_path",
        type=Path,
        help="Path to evaluation report JSON",
    )
    report_parser.add_argument(
        "--compare",
        type=Path,
        help="Path to another report to compare against",
    )
    report_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate dataset or predictions",
        description="Check dataset or predictions file for errors",
    )
    validate_parser.add_argument(
        "file",
        type=Path,
        help="Path to file to validate",
    )
    validate_parser.add_argument(
        "--type",
        choices=["dataset", "predictions"],
        default="dataset",
        help="Type of file to validate (default: dataset)",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation",
    )

    return parser


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    from bunbench.harness.run_evaluation import run_evaluation, EvaluationConfig
    from bunbench.harness.reporting import save_report, print_report_summary

    try:
        config = EvaluationConfig(
            dataset_path=args.dataset,
            predictions_path=args.predictions,
            output_dir=args.output,
            max_workers=args.workers,
            timeout=args.timeout,
            run_id=args.run_id,
            verbose=args.verbose,
        )

        print(f"Running evaluation...")
        print(f"  Dataset: {args.dataset}")
        print(f"  Predictions: {args.predictions}")
        print(f"  Workers: {args.workers}")
        print()

        results = run_evaluation(config)

        # Save and display report
        report_path = args.output / "evaluation_report.json"
        save_report(results, report_path)
        print_report_summary(results)

        print(f"\nReport saved to: {report_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_inference(args: argparse.Namespace) -> int:
    """Run inference command."""
    try:
        from bunbench.inference.run_api import run_inference

        print(f"Running inference...")
        print(f"  Dataset: {args.dataset}")
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model or 'default'}")
        print()

        run_inference(
            dataset_path=args.dataset,
            output_path=args.output,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )

        print(f"\nPredictions saved to: {args.output}")
        return 0

    except ImportError:
        print("Error: Inference dependencies not installed.", file=sys.stderr)
        print("Install with: pip install bun-bench[inference]", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_build_images(args: argparse.Namespace) -> int:
    """Run build-images command."""
    from bunbench.harness.docker_build import build_base_image, build_env_image

    try:
        print("Building Docker images...")

        # Build base image
        print("  Building base image...")
        result = build_base_image(no_cache=args.no_cache)
        if not result.success:
            print(f"Error building base image: {result.error}", file=sys.stderr)
            return 1
        print(f"    Base image: {result.image_name}")

        # Build env image
        bun_version = args.bun_version or "latest"
        print(f"  Building env image for Bun {bun_version}...")
        result = build_env_image(bun_version, no_cache=args.no_cache)
        if not result.success:
            print(f"Error building env image: {result.error}", file=sys.stderr)
            return 1
        print(f"    Env image: {result.image_name}")

        print("\nDocker images built successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Run report command."""
    from bunbench.harness.reporting import load_report, print_report_summary, compare_reports

    try:
        report = load_report(args.report_path)

        if args.compare:
            other_report = load_report(args.compare)
            comparison = compare_reports(report, other_report)

            print(f"Comparison: {args.report_path} vs {args.compare}")
            print(f"  Resolved: {comparison['base_resolved']} -> {comparison['other_resolved']}")
            print(f"  Improvements: {len(comparison.get('improvements', []))}")
            print(f"  Regressions: {len(comparison.get('regressions', []))}")
        else:
            print_report_summary(report)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validate command."""
    import json

    try:
        with open(args.file) as f:
            data = json.load(f)

        if args.type == "dataset":
            from bunbench.collect.build_dataset import validate_instance

            instances = data if isinstance(data, list) else [data]
            errors = []

            for i, instance in enumerate(instances):
                is_valid, error = validate_instance(instance, strict=args.strict)
                if not is_valid:
                    errors.append(f"Instance {i}: {error}")

            if errors:
                print(f"Validation failed with {len(errors)} errors:", file=sys.stderr)
                for error in errors[:10]:
                    print(f"  - {error}", file=sys.stderr)
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
                return 1

            print(f"Validation passed: {len(instances)} instances OK")
            return 0

        else:  # predictions
            predictions = data if isinstance(data, list) else list(data.values())

            for pred in predictions:
                if "instance_id" not in pred:
                    print("Error: Missing instance_id in prediction", file=sys.stderr)
                    return 1
                if "model_patch" not in pred and "patch" not in pred:
                    print(f"Error: Missing patch in prediction {pred.get('instance_id')}", file=sys.stderr)
                    return 1

            print(f"Validation passed: {len(predictions)} predictions OK")
            return 0

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "evaluate": cmd_evaluate,
        "inference": cmd_inference,
        "build-images": cmd_build_images,
        "report": cmd_report,
        "validate": cmd_validate,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
