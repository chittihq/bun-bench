"""
Main CLI entry point for Bun-Bench.

This module provides the command-line interface for running evaluations,
building Docker images, and managing the benchmark suite.
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    import os

    from bunbench.harness.reporting import (
        generate_report,
        print_report_summary,
        save_report,
    )
    from bunbench.harness.run_evaluation import (
        EvaluationConfig,
        run_evaluation,
    )

    try:
        # Create configuration
        config = EvaluationConfig(
            dataset_path=args.dataset,
            predictions_path=args.predictions,
            output_dir=args.output,
            max_workers=args.workers,
            timeout=args.timeout,
            docker_image_prefix=args.docker_prefix,
            force_rebuild=args.force_rebuild,
            verbose=args.verbose,
            instance_ids=args.instance_ids,
        )

        # Run evaluation
        results = run_evaluation(config)

        # Generate and save report
        report = generate_report(
            results,
            config=vars(args),
            include_test_output=not args.no_output,
        )

        report_path = os.path.join(args.output, "evaluation_report.json")
        save_report(report, report_path)

        # Print summary
        print_report_summary(report)
        print(f"\nFull report saved to: {report_path}")

        # Return exit code based on results
        if report.errors > 0:
            return 2
        elif report.unresolved > 0:
            return 1
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


def cmd_build_images(args: argparse.Namespace) -> int:
    """Build Docker images command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from bunbench.harness.run_evaluation import (
        build_docker_image,
        get_docker_image_name,
        load_dataset,
    )

    try:
        # Load dataset
        dataset = load_dataset(args.dataset)

        # Filter instances if specific IDs requested
        if args.instance_ids:
            dataset = [inst for inst in dataset if inst.get("instance_id") in args.instance_ids]

        logger.info(f"Building images for {len(dataset)} instances")

        success_count = 0
        failure_count = 0

        for instance in dataset:
            instance_id = instance.get("instance_id", "unknown")
            image_name = get_docker_image_name(instance, args.docker_prefix)

            logger.info(f"Building image for {instance_id}: {image_name}")

            if build_docker_image(instance, image_name, args.force_rebuild):
                success_count += 1
                logger.info(f"  Success: {image_name}")
            else:
                failure_count += 1
                logger.error(f"  Failed: {image_name}")

        print(f"\nBuild complete: {success_count} succeeded, {failure_count} failed")

        return 0 if failure_count == 0 else 1

    except Exception as e:
        logger.exception(f"Build failed: {e}")
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """View or compare reports command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    import json

    from bunbench.harness.reporting import (
        compare_reports,
        load_report,
        print_report_summary,
    )

    try:
        if args.compare:
            # Compare two reports
            report1 = load_report(args.report)
            report2 = load_report(args.compare)

            comparison = compare_reports(
                report1,
                report2,
                label1=args.report,
                label2=args.compare,
            )

            print("\n" + "=" * 70)
            print("REPORT COMPARISON")
            print("=" * 70)
            print(f"Report 1: {args.report}")
            print(f"Report 2: {args.compare}")
            print("-" * 70)

            r1_rate = comparison["summary"]["resolved_rate"][0]
            r2_rate = comparison["summary"]["resolved_rate"][1]
            diff = comparison["resolved_rate_diff"]

            print(f"Resolved Rate: {r1_rate:.1%} -> {r2_rate:.1%} ({diff:+.1%})")
            print(f"Improvements: {len(comparison['improvements'])}")
            print(f"Regressions:  {len(comparison['regressions'])}")

            if comparison["improvements"]:
                print("\nIMPROVEMENTS:")
                for item in comparison["improvements"]:
                    print(f"  {item['instance_id']}: {item['from']} -> {item['to']}")

            if comparison["regressions"]:
                print("\nREGRESSIONS:")
                for item in comparison["regressions"]:
                    print(f"  {item['instance_id']}: {item['from']} -> {item['to']}")

            print("=" * 70)

            if args.json:
                print("\nJSON Output:")
                print(json.dumps(comparison, indent=2))

        else:
            # View single report
            report = load_report(args.report)

            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print_report_summary(report)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Report file not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Failed to load report: {e}")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Print version information.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (always 0).
    """
    from bunbench import __version__

    print(f"Bun-Bench version {__version__}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="bunbench",
        description="Bun-Bench: Evaluation harness for LLM code generation on Bun.js",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", "-V", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on predictions",
        description="Evaluate model-generated patches against the benchmark",
    )
    eval_parser.add_argument(
        "--dataset", "-d", required=True, help="Path to dataset JSON or HuggingFace identifier"
    )
    eval_parser.add_argument(
        "--predictions", "-p", required=True, help="Path to predictions JSON (instance_id -> patch)"
    )
    eval_parser.add_argument(
        "--output", "-o", default="./results", help="Output directory (default: ./results)"
    )
    eval_parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    eval_parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout per evaluation in seconds (default: 300)",
    )
    eval_parser.add_argument(
        "--docker-prefix", default="bunbench", help="Docker image prefix (default: bunbench)"
    )
    eval_parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuild Docker images"
    )
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    eval_parser.add_argument("--instance-ids", nargs="+", help="Specific instance IDs to evaluate")
    eval_parser.add_argument(
        "--no-output", action="store_true", help="Don't include test output in report"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Build images command
    build_parser = subparsers.add_parser(
        "build-images",
        help="Build Docker images for instances",
        description="Pre-build Docker images for benchmark instances",
    )
    build_parser.add_argument(
        "--dataset", "-d", required=True, help="Path to dataset JSON or HuggingFace identifier"
    )
    build_parser.add_argument(
        "--docker-prefix", default="bunbench", help="Docker image prefix (default: bunbench)"
    )
    build_parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuild existing images"
    )
    build_parser.add_argument("--instance-ids", nargs="+", help="Specific instance IDs to build")
    build_parser.set_defaults(func=cmd_build_images)

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="View or compare evaluation reports",
        description="View evaluation reports or compare two reports",
    )
    report_parser.add_argument("report", help="Path to evaluation report JSON")
    report_parser.add_argument("--compare", "-c", help="Path to second report for comparison")
    report_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    report_parser.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle version flag
    if args.version:
        return cmd_version(args)

    # Handle no command
    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handler
    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
