"""
Docker build utilities for Bun-Bench.

This module provides functions to build Docker images for the Bun-Bench
evaluation harness, including base images, environment images with specific
Bun versions, and instance images for individual evaluations.
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Image naming conventions
IMAGE_PREFIX = "bunbench"
BASE_IMAGE_NAME = f"{IMAGE_PREFIX}.base"
ENV_IMAGE_PREFIX = f"{IMAGE_PREFIX}.env"
EVAL_IMAGE_PREFIX = f"{IMAGE_PREFIX}.eval"


@dataclass
class BuildResult:
    """Result of a Docker build operation."""

    success: bool
    image_name: str
    image_id: str | None = None
    error: str | None = None
    cached: bool = False


def get_docker_dir() -> Path:
    """Get the path to the docker directory containing Dockerfiles."""
    return Path(__file__).parent.parent.parent / "docker"


def get_env_image_name(version: str) -> str:
    """
    Get the image name for a specific Bun version.

    Args:
        version: Bun version string (e.g., "1.0.0", "latest")

    Returns:
        Image name string (e.g., "bunbench.env.1.0.0")
    """
    # Sanitize version string for use in image name
    sanitized = version.replace("/", "-").replace(":", "-")
    return f"{ENV_IMAGE_PREFIX}.{sanitized}"


def get_eval_image_name(instance_id: str) -> str:
    """
    Get the image name for a specific evaluation instance.

    Args:
        instance_id: Unique identifier for the evaluation instance

    Returns:
        Image name string (e.g., "bunbench.eval.abc123")
    """
    # Sanitize instance_id for use in image name
    sanitized = instance_id.replace("/", "-").replace(":", "-")
    return f"{EVAL_IMAGE_PREFIX}.{sanitized}"


def image_exists(image_name: str) -> bool:
    """
    Check if a Docker image exists locally.

    Args:
        image_name: Name of the image to check

    Returns:
        True if image exists, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name], capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking image existence: {e}")
        return False


def get_image_id(image_name: str) -> str | None:
    """
    Get the ID of a Docker image.

    Args:
        image_name: Name of the image

    Returns:
        Image ID if found, None otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        logger.error(f"Error getting image ID: {e}")
        return None


def build_base_image(force_rebuild: bool = False, no_cache: bool = False) -> BuildResult:
    """
    Build the base Docker image for Bun-Bench.

    The base image contains:
    - Ubuntu 22.04
    - System dependencies (git, curl, build-essential, etc.)
    - Base Bun installation
    - Non-root user configuration

    Args:
        force_rebuild: If True, rebuild even if image exists
        no_cache: If True, build without using Docker cache

    Returns:
        BuildResult with success status and image info
    """
    image_name = BASE_IMAGE_NAME

    # Check if image already exists
    if not force_rebuild and image_exists(image_name):
        logger.info(f"Base image {image_name} already exists (using cache)")
        return BuildResult(
            success=True, image_name=image_name, image_id=get_image_id(image_name), cached=True
        )

    logger.info(f"Building base image: {image_name}")

    docker_dir = get_docker_dir()
    dockerfile_path = docker_dir / "Dockerfile.base"

    if not dockerfile_path.exists():
        return BuildResult(
            success=False, image_name=image_name, error=f"Dockerfile not found: {dockerfile_path}"
        )

    # Build command
    cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(dockerfile_path),
    ]

    if no_cache:
        cmd.append("--no-cache")

    # Use docker directory as context
    cmd.append(str(docker_dir))

    try:
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"Successfully built base image: {image_name}")
            return BuildResult(
                success=True, image_name=image_name, image_id=get_image_id(image_name)
            )
        else:
            logger.error(f"Failed to build base image: {result.stderr}")
            return BuildResult(success=False, image_name=image_name, error=result.stderr)
    except subprocess.TimeoutExpired:
        return BuildResult(
            success=False, image_name=image_name, error="Build timed out after 30 minutes"
        )
    except Exception as e:
        return BuildResult(success=False, image_name=image_name, error=str(e))


def build_env_image(
    bun_version: str = "latest",
    force_rebuild: bool = False,
    no_cache: bool = False,
    package_json_path: Path | None = None,
) -> BuildResult:
    """
    Build an environment image with a specific Bun version.

    Args:
        bun_version: Bun version to install (e.g., "1.0.0", "latest")
        force_rebuild: If True, rebuild even if image exists
        no_cache: If True, build without using Docker cache
        package_json_path: Optional path to package.json to include

    Returns:
        BuildResult with success status and image info
    """
    image_name = get_env_image_name(bun_version)

    # Check if image already exists
    if not force_rebuild and image_exists(image_name):
        logger.info(f"Env image {image_name} already exists (using cache)")
        return BuildResult(
            success=True, image_name=image_name, image_id=get_image_id(image_name), cached=True
        )

    # Ensure base image exists
    base_result = build_base_image()
    if not base_result.success:
        return BuildResult(
            success=False,
            image_name=image_name,
            error=f"Failed to build base image: {base_result.error}",
        )

    logger.info(f"Building env image: {image_name} with Bun {bun_version}")

    docker_dir = get_docker_dir()
    dockerfile_path = docker_dir / "Dockerfile.env"

    if not dockerfile_path.exists():
        return BuildResult(
            success=False, image_name=image_name, error=f"Dockerfile not found: {dockerfile_path}"
        )

    # Create a build context directory
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as build_context:
        build_context_path = Path(build_context)

        # Copy Dockerfile
        shutil.copy(dockerfile_path, build_context_path / "Dockerfile")

        # Copy package.json if provided
        if package_json_path and package_json_path.exists():
            shutil.copy(package_json_path, build_context_path / "package.json")

        # Build command
        cmd = [
            "docker",
            "build",
            "-t",
            image_name,
            "-f",
            str(build_context_path / "Dockerfile"),
            "--build-arg",
            f"BASE_IMAGE={BASE_IMAGE_NAME}",
            "--build-arg",
            f"BUN_VERSION={bun_version}",
        ]

        if no_cache:
            cmd.append("--no-cache")

        cmd.append(str(build_context_path))

        try:
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Successfully built env image: {image_name}")
                return BuildResult(
                    success=True, image_name=image_name, image_id=get_image_id(image_name)
                )
            else:
                logger.error(f"Failed to build env image: {result.stderr}")
                return BuildResult(success=False, image_name=image_name, error=result.stderr)
        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False, image_name=image_name, error="Build timed out after 30 minutes"
            )
        except Exception as e:
            return BuildResult(success=False, image_name=image_name, error=str(e))


def build_instance_image(
    instance_id: str,
    bun_version: str = "latest",
    base_commit: str = "main",
    test_files_dir: Path | None = None,
    force_rebuild: bool = False,
    no_cache: bool = False,
) -> BuildResult:
    """
    Build an instance image for a specific evaluation.

    Args:
        instance_id: Unique identifier for this evaluation instance
        bun_version: Bun version for the environment
        base_commit: Git commit hash or branch to checkout in bun repo
        test_files_dir: Optional directory containing test files to copy
        force_rebuild: If True, rebuild even if image exists
        no_cache: If True, build without using Docker cache

    Returns:
        BuildResult with success status and image info
    """
    image_name = get_eval_image_name(instance_id)
    env_image_name = get_env_image_name(bun_version)

    # Check if image already exists
    if not force_rebuild and image_exists(image_name):
        logger.info(f"Instance image {image_name} already exists (using cache)")
        return BuildResult(
            success=True, image_name=image_name, image_id=get_image_id(image_name), cached=True
        )

    # Ensure env image exists
    env_result = build_env_image(bun_version)
    if not env_result.success:
        return BuildResult(
            success=False,
            image_name=image_name,
            error=f"Failed to build env image: {env_result.error}",
        )

    logger.info(f"Building instance image: {image_name}")

    docker_dir = get_docker_dir()
    dockerfile_path = docker_dir / "Dockerfile.instance"

    if not dockerfile_path.exists():
        return BuildResult(
            success=False, image_name=image_name, error=f"Dockerfile not found: {dockerfile_path}"
        )

    # Create a build context directory
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as build_context:
        build_context_path = Path(build_context)

        # Copy Dockerfile
        shutil.copy(dockerfile_path, build_context_path / "Dockerfile")

        # Copy test files if provided
        if test_files_dir and test_files_dir.exists():
            for item in test_files_dir.iterdir():
                if item.is_file():
                    shutil.copy(item, build_context_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, build_context_path / item.name)

        # Build command
        cmd = [
            "docker",
            "build",
            "-t",
            image_name,
            "-f",
            str(build_context_path / "Dockerfile"),
            "--build-arg",
            f"ENV_IMAGE={env_image_name}",
            "--build-arg",
            f"INSTANCE_ID={instance_id}",
            "--build-arg",
            f"BASE_COMMIT={base_commit}",
        ]

        if no_cache:
            cmd.append("--no-cache")

        cmd.append(str(build_context_path))

        try:
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 60 minute timeout for cloning repo
            )

            if result.returncode == 0:
                logger.info(f"Successfully built instance image: {image_name}")
                return BuildResult(
                    success=True, image_name=image_name, image_id=get_image_id(image_name)
                )
            else:
                logger.error(f"Failed to build instance image: {result.stderr}")
                return BuildResult(success=False, image_name=image_name, error=result.stderr)
        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False, image_name=image_name, error="Build timed out after 60 minutes"
            )
        except Exception as e:
            return BuildResult(success=False, image_name=image_name, error=str(e))


def cleanup_images(
    prefix: str | None = None, keep_base: bool = True, keep_env: bool = True
) -> dict:
    """
    Clean up Bun-Bench Docker images.

    Args:
        prefix: Only remove images with this prefix (default: all bunbench images)
        keep_base: If True, keep the base image
        keep_env: If True, keep environment images

    Returns:
        Dict with removed images and any errors
    """
    prefix = prefix or IMAGE_PREFIX

    try:
        # List all images with the prefix
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", prefix + "*"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"removed": [], "errors": [result.stderr]}

        images = [img for img in result.stdout.strip().split("\n") if img]
        removed = []
        errors = []

        for image in images:
            # Skip base image if requested
            if keep_base and image.startswith(BASE_IMAGE_NAME):
                continue

            # Skip env images if requested
            if keep_env and image.startswith(ENV_IMAGE_PREFIX):
                continue

            # Remove image
            rm_result = subprocess.run(
                ["docker", "rmi", "-f", image], capture_output=True, text=True
            )

            if rm_result.returncode == 0:
                removed.append(image)
                logger.info(f"Removed image: {image}")
            else:
                errors.append(f"Failed to remove {image}: {rm_result.stderr}")
                logger.error(f"Failed to remove {image}: {rm_result.stderr}")

        return {"removed": removed, "errors": errors}

    except Exception as e:
        return {"removed": [], "errors": [str(e)]}


def get_cache_key(
    image_type: str,
    version: str | None = None,
    instance_id: str | None = None,
    base_commit: str | None = None,
) -> str:
    """
    Generate a cache key for an image build.

    Args:
        image_type: Type of image ("base", "env", "instance")
        version: Bun version (for env/instance)
        instance_id: Instance ID (for instance)
        base_commit: Base commit (for instance)

    Returns:
        Cache key string
    """
    data = {
        "type": image_type,
        "version": version,
        "instance_id": instance_id,
        "base_commit": base_commit,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
