#!/usr/bin/env bash
#
# run_evaluation.sh - Bash wrapper script for Bun-Bench evaluation
#
# This script provides a convenient way to run evaluations with common options
# and handles virtual environment activation if available.
#
# Usage:
#   ./scripts/run_evaluation.sh --dataset ./data/bun-bench.json --predictions ./predictions.json
#   ./scripts/run_evaluation.sh -d ./data/bun-bench.json -p ./predictions.json -w 8 -v
#
# Environment Variables:
#   BUNBENCH_VENV      - Path to virtualenv (default: ./venv or ./.venv)
#   BUNBENCH_WORKERS   - Number of parallel workers (default: 4)
#   BUNBENCH_TIMEOUT   - Timeout per evaluation in seconds (default: 300)
#   BUNBENCH_OUTPUT    - Output directory (default: ./results)
#

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DEFAULT_WORKERS="${BUNBENCH_WORKERS:-4}"
DEFAULT_TIMEOUT="${BUNBENCH_TIMEOUT:-300}"
DEFAULT_OUTPUT="${BUNBENCH_OUTPUT:-./results}"

# Colors for output (if terminal supports it)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Print usage information
usage() {
    cat << EOF
Bun-Bench Evaluation Runner

Usage: $(basename "$0") [OPTIONS]

Required:
  -d, --dataset PATH       Path to dataset JSON or HuggingFace identifier
  -p, --predictions PATH   Path to predictions JSON (instance_id -> patch)

Options:
  -o, --output DIR         Output directory (default: $DEFAULT_OUTPUT)
  -w, --workers NUM        Number of parallel workers (default: $DEFAULT_WORKERS)
  -t, --timeout SECS       Timeout per evaluation (default: $DEFAULT_TIMEOUT)
  --docker-prefix PREFIX   Docker image prefix (default: bunbench)
  --force-rebuild          Force rebuild Docker images
  --instance-ids IDS       Space-separated instance IDs to evaluate
  -v, --verbose            Enable verbose output
  -h, --help               Show this help message

Examples:
  # Basic evaluation
  $(basename "$0") -d ./data/bun-bench.json -p ./predictions.json

  # With more workers and verbose output
  $(basename "$0") -d ./data/bun-bench.json -p ./predictions.json -w 8 -v

  # Evaluate specific instances
  $(basename "$0") -d ./data/bun-bench.json -p ./predictions.json \\
      --instance-ids "bun-123 bun-456"

  # Using environment variables
  BUNBENCH_WORKERS=8 $(basename "$0") -d ./data/bun-bench.json -p ./predictions.json

Environment Variables:
  BUNBENCH_VENV      Path to virtualenv directory
  BUNBENCH_WORKERS   Number of parallel workers
  BUNBENCH_TIMEOUT   Timeout per evaluation in seconds
  BUNBENCH_OUTPUT    Output directory

EOF
}

# Activate virtual environment if available
activate_venv() {
    local venv_paths=(
        "${BUNBENCH_VENV:-}"
        "$PROJECT_ROOT/venv"
        "$PROJECT_ROOT/.venv"
        "$HOME/.virtualenvs/bunbench"
    )

    for venv_path in "${venv_paths[@]}"; do
        if [[ -n "$venv_path" && -f "$venv_path/bin/activate" ]]; then
            log_info "Activating virtualenv: $venv_path"
            # shellcheck source=/dev/null
            source "$venv_path/bin/activate"
            return 0
        fi
    done

    log_warning "No virtualenv found, using system Python"
    return 0
}

# Check dependencies
check_dependencies() {
    local missing=()

    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    # Check if Docker is running
    if command -v docker &> /dev/null; then
        if ! docker info &> /dev/null; then
            log_error "Docker is installed but not running"
            exit 1
        fi
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing[*]}"
        exit 1
    fi

    log_info "All dependencies satisfied"
}

# Parse command line arguments
parse_args() {
    DATASET=""
    PREDICTIONS=""
    OUTPUT="$DEFAULT_OUTPUT"
    WORKERS="$DEFAULT_WORKERS"
    TIMEOUT="$DEFAULT_TIMEOUT"
    DOCKER_PREFIX="bunbench"
    FORCE_REBUILD=""
    INSTANCE_IDS=""
    VERBOSE=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dataset)
                DATASET="$2"
                shift 2
                ;;
            -p|--predictions)
                PREDICTIONS="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT="$2"
                shift 2
                ;;
            -w|--workers)
                WORKERS="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --docker-prefix)
                DOCKER_PREFIX="$2"
                shift 2
                ;;
            --force-rebuild)
                FORCE_REBUILD="--force-rebuild"
                shift
                ;;
            --instance-ids)
                INSTANCE_IDS="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="--verbose"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$DATASET" ]]; then
        log_error "Missing required argument: --dataset"
        usage
        exit 1
    fi

    if [[ -z "$PREDICTIONS" ]]; then
        log_error "Missing required argument: --predictions"
        usage
        exit 1
    fi

    # Validate files exist (for local paths)
    if [[ -f "$DATASET" ]]; then
        DATASET="$(realpath "$DATASET")"
    elif [[ ! "$DATASET" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$ ]]; then
        # Not a HuggingFace identifier format
        log_error "Dataset file not found: $DATASET"
        exit 1
    fi

    if [[ ! -f "$PREDICTIONS" ]]; then
        log_error "Predictions file not found: $PREDICTIONS"
        exit 1
    fi
    PREDICTIONS="$(realpath "$PREDICTIONS")"
}

# Run the evaluation
run_evaluation() {
    log_info "Starting Bun-Bench evaluation"
    log_info "Dataset: $DATASET"
    log_info "Predictions: $PREDICTIONS"
    log_info "Output: $OUTPUT"
    log_info "Workers: $WORKERS"
    log_info "Timeout: ${TIMEOUT}s"

    # Build command
    local cmd=(
        python3 -m bunbench evaluate
        --dataset "$DATASET"
        --predictions "$PREDICTIONS"
        --output "$OUTPUT"
        --workers "$WORKERS"
        --timeout "$TIMEOUT"
        --docker-prefix "$DOCKER_PREFIX"
    )

    if [[ -n "$FORCE_REBUILD" ]]; then
        cmd+=("$FORCE_REBUILD")
    fi

    if [[ -n "$VERBOSE" ]]; then
        cmd+=("$VERBOSE")
    fi

    if [[ -n "$INSTANCE_IDS" ]]; then
        # shellcheck disable=SC2206
        cmd+=(--instance-ids $INSTANCE_IDS)
    fi

    # Change to project root
    cd "$PROJECT_ROOT"

    # Run evaluation
    log_info "Running: ${cmd[*]}"
    echo ""

    local start_time
    start_time=$(date +%s)

    if "${cmd[@]}"; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))

        echo ""
        log_success "Evaluation completed in ${duration}s"
        log_success "Results saved to: $OUTPUT"
    else
        local exit_code=$?
        log_error "Evaluation failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Main function
main() {
    parse_args "$@"
    check_dependencies
    activate_venv
    run_evaluation
}

# Run main
main "$@"
