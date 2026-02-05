"""
Docker utility functions for Bun-Bench.

This module provides functions to manage Docker containers for running
evaluations, including container creation, file operations, command
execution, and cleanup.
"""

import io
import logging
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_MEMORY_LIMIT = "2g"
DEFAULT_CPU_LIMIT = "2"


@dataclass
class ContainerInfo:
    """Information about a Docker container."""
    container_id: str
    name: str
    image: str
    status: str = "created"
    exit_code: int | None = None


@dataclass
class ExecutionResult:
    """Result of executing a command in a container."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float  # seconds
    timed_out: bool = False


@dataclass
class ContainerConfig:
    """Configuration for creating a container."""
    image: str
    name: str | None = None
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: str = DEFAULT_CPU_LIMIT
    environment: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    network_mode: str = "bridge"
    working_dir: str = "/home/bunuser/workspace/tests"
    user: str = "bunuser"


def generate_container_name(prefix: str = "bunbench") -> str:
    """
    Generate a unique container name.

    Args:
        prefix: Prefix for the container name

    Returns:
        Unique container name
    """
    unique_id = uuid.uuid4().hex[:8]
    timestamp = int(time.time())
    return f"{prefix}-{timestamp}-{unique_id}"


def create_container(config: ContainerConfig) -> tuple[ContainerInfo | None, str | None]:
    """
    Create a Docker container from an image.

    Args:
        config: Container configuration

    Returns:
        Tuple of (ContainerInfo, error_message)
    """
    container_name = config.name or generate_container_name()

    cmd = [
        "docker", "create",
        "--name", container_name,
        "--memory", config.memory_limit,
        "--cpus", config.cpu_limit,
        "--network", config.network_mode,
        "--workdir", config.working_dir,
        "--user", config.user,
    ]

    # Add environment variables
    for key, value in config.environment.items():
        cmd.extend(["-e", f"{key}={value}"])

    # Add volume mounts
    for host_path, container_path in config.volumes.items():
        cmd.extend(["-v", f"{host_path}:{container_path}"])

    # Add image name
    cmd.append(config.image)

    # Add default command (sleep to keep container running)
    cmd.extend(["tail", "-f", "/dev/null"])

    try:
        logger.debug(f"Creating container: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            container_id = result.stdout.strip()
            logger.info(f"Created container: {container_name} ({container_id[:12]})")
            return ContainerInfo(
                container_id=container_id,
                name=container_name,
                image=config.image,
                status="created"
            ), None
        else:
            logger.error(f"Failed to create container: {result.stderr}")
            return None, result.stderr

    except subprocess.TimeoutExpired:
        return None, "Container creation timed out"
    except Exception as e:
        return None, str(e)


def start_container(container: ContainerInfo) -> tuple[bool, str | None]:
    """
    Start a Docker container.

    Args:
        container: Container info

    Returns:
        Tuple of (success, error_message)
    """
    try:
        result = subprocess.run(
            ["docker", "start", container.container_id],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            container.status = "running"
            logger.info(f"Started container: {container.name}")
            return True, None
        else:
            logger.error(f"Failed to start container: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Container start timed out"
    except Exception as e:
        return False, str(e)


def stop_container(
    container: ContainerInfo,
    timeout: int = 10
) -> tuple[bool, str | None]:
    """
    Stop a Docker container.

    Args:
        container: Container info
        timeout: Seconds to wait before force killing

    Returns:
        Tuple of (success, error_message)
    """
    try:
        result = subprocess.run(
            ["docker", "stop", "-t", str(timeout), container.container_id],
            capture_output=True,
            text=True,
            timeout=timeout + 30
        )

        if result.returncode == 0:
            container.status = "stopped"
            logger.info(f"Stopped container: {container.name}")
            return True, None
        else:
            logger.error(f"Failed to stop container: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        # Force kill
        subprocess.run(
            ["docker", "kill", container.container_id],
            capture_output=True
        )
        container.status = "killed"
        return True, "Container was force killed"
    except Exception as e:
        return False, str(e)


def copy_to_container(
    container: ContainerInfo,
    source_path: Path,
    dest_path: str
) -> tuple[bool, str | None]:
    """
    Copy files to a container.

    Args:
        container: Container info
        source_path: Local path to copy from
        dest_path: Container path to copy to

    Returns:
        Tuple of (success, error_message)
    """
    if not source_path.exists():
        return False, f"Source path does not exist: {source_path}"

    try:
        result = subprocess.run(
            ["docker", "cp", str(source_path), f"{container.container_id}:{dest_path}"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.debug(f"Copied {source_path} to {container.name}:{dest_path}")
            return True, None
        else:
            logger.error(f"Failed to copy to container: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Copy operation timed out"
    except Exception as e:
        return False, str(e)


def copy_from_container(
    container: ContainerInfo,
    source_path: str,
    dest_path: Path
) -> tuple[bool, str | None]:
    """
    Copy files from a container.

    Args:
        container: Container info
        source_path: Container path to copy from
        dest_path: Local path to copy to

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["docker", "cp", f"{container.container_id}:{source_path}", str(dest_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.debug(f"Copied {container.name}:{source_path} to {dest_path}")
            return True, None
        else:
            logger.error(f"Failed to copy from container: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Copy operation timed out"
    except Exception as e:
        return False, str(e)


def copy_content_to_container(
    container: ContainerInfo,
    content: str,
    dest_path: str,
    filename: str = "file.txt"
) -> tuple[bool, str | None]:
    """
    Copy string content to a file in the container.

    Args:
        container: Container info
        content: String content to write
        dest_path: Container directory path
        filename: Name for the file

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Create a tar archive in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            content_bytes = content.encode('utf-8')
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(content_bytes)
            tar.addfile(tarinfo, io.BytesIO(content_bytes))

        tar_stream.seek(0)

        # Use docker cp with stdin
        process = subprocess.Popen(
            ["docker", "cp", "-", f"{container.container_id}:{dest_path}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate(input=tar_stream.read(), timeout=60)

        if process.returncode == 0:
            logger.debug(f"Copied content to {container.name}:{dest_path}/{filename}")
            return True, None
        else:
            return False, stderr.decode()

    except subprocess.TimeoutExpired:
        process.kill()
        return False, "Copy operation timed out"
    except Exception as e:
        return False, str(e)


def execute_command(
    container: ContainerInfo,
    command: list[str],
    timeout: int = DEFAULT_TIMEOUT,
    working_dir: str | None = None,
    environment: dict[str, str] | None = None,
    user: str | None = None
) -> ExecutionResult:
    """
    Execute a command in a container with timeout.

    Args:
        container: Container info
        command: Command to execute as list of strings
        timeout: Maximum execution time in seconds
        working_dir: Override working directory
        environment: Additional environment variables
        user: Override user

    Returns:
        ExecutionResult with command output and status
    """
    start_time = time.time()

    cmd = ["docker", "exec"]

    if working_dir:
        cmd.extend(["-w", working_dir])

    if user:
        cmd.extend(["-u", user])

    if environment:
        for key, value in environment.items():
            cmd.extend(["-e", f"{key}={value}"])

    cmd.append(container.container_id)
    cmd.extend(command)

    try:
        logger.debug(f"Executing in {container.name}: {' '.join(command)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        duration = time.time() - start_time

        return ExecutionResult(
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration,
            timed_out=False
        )

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        logger.warning(f"Command timed out after {timeout}s in {container.name}")

        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout=e.stdout.decode() if e.stdout else "",
            stderr=e.stderr.decode() if e.stderr else "Command timed out",
            duration=duration,
            timed_out=True
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error executing command: {e}")

        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration=duration,
            timed_out=False
        )


def execute_script(
    container: ContainerInfo,
    script: str,
    timeout: int = DEFAULT_TIMEOUT,
    working_dir: str | None = None,
    environment: dict[str, str] | None = None
) -> ExecutionResult:
    """
    Execute a shell script in a container.

    Args:
        container: Container info
        script: Shell script content
        timeout: Maximum execution time in seconds
        working_dir: Override working directory
        environment: Additional environment variables

    Returns:
        ExecutionResult with command output and status
    """
    # Copy script to container
    script_name = f"script_{uuid.uuid4().hex[:8]}.sh"
    dest_dir = working_dir or "/home/bunuser/workspace/tests"

    success, error = copy_content_to_container(
        container, script, dest_dir, script_name
    )

    if not success:
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Failed to copy script: {error}",
            duration=0,
            timed_out=False
        )

    # Make script executable and run it
    script_path = f"{dest_dir}/{script_name}"
    return execute_command(
        container,
        ["bash", script_path],
        timeout=timeout,
        working_dir=working_dir,
        environment=environment
    )


def get_container_logs(
    container: ContainerInfo,
    tail: int | None = None,
    since: str | None = None
) -> tuple[str, str]:
    """
    Get logs from a container.

    Args:
        container: Container info
        tail: Number of lines to show from end
        since: Show logs since timestamp (e.g., "10m", "2023-01-01")

    Returns:
        Tuple of (stdout, stderr)
    """
    cmd = ["docker", "logs"]

    if tail:
        cmd.extend(["--tail", str(tail)])

    if since:
        cmd.extend(["--since", since])

    cmd.append(container.container_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout, result.stderr

    except Exception as e:
        logger.error(f"Error getting container logs: {e}")
        return "", str(e)


def get_container_status(container: ContainerInfo) -> str | None:
    """
    Get the current status of a container.

    Args:
        container: Container info

    Returns:
        Status string or None if error
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", container.container_id, "--format", "{{.State.Status}}"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            status = result.stdout.strip()
            container.status = status
            return status
        return None

    except Exception as e:
        logger.error(f"Error getting container status: {e}")
        return None


def cleanup_container(
    container: ContainerInfo,
    force: bool = True,
    remove_volumes: bool = False
) -> tuple[bool, str | None]:
    """
    Remove a container and optionally its volumes.

    Args:
        container: Container info
        force: Force removal even if running
        remove_volumes: Also remove associated volumes

    Returns:
        Tuple of (success, error_message)
    """
    cmd = ["docker", "rm"]

    if force:
        cmd.append("-f")

    if remove_volumes:
        cmd.append("-v")

    cmd.append(container.container_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            container.status = "removed"
            logger.info(f"Removed container: {container.name}")
            return True, None
        else:
            logger.error(f"Failed to remove container: {result.stderr}")
            return False, result.stderr

    except Exception as e:
        return False, str(e)


def cleanup_containers_by_prefix(
    prefix: str = "bunbench",
    force: bool = True
) -> dict[str, Any]:
    """
    Clean up all containers matching a name prefix.

    Args:
        prefix: Container name prefix to match
        force: Force removal even if running

    Returns:
        Dict with removed containers and errors
    """
    try:
        # List containers matching prefix
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.ID}}\t{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return {"removed": [], "errors": [result.stderr]}

        removed = []
        errors = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            container_id = parts[0]
            container_name = parts[1] if len(parts) > 1 else container_id

            # Create a container info object
            container = ContainerInfo(
                container_id=container_id,
                name=container_name,
                image=""
            )

            success, error = cleanup_container(container, force=force)

            if success:
                removed.append(container_name)
            else:
                errors.append(f"Failed to remove {container_name}: {error}")

        return {"removed": removed, "errors": errors}

    except Exception as e:
        return {"removed": [], "errors": [str(e)]}


def list_containers(
    prefix: str = "bunbench",
    all_containers: bool = True
) -> list[ContainerInfo]:
    """
    List containers matching a name prefix.

    Args:
        prefix: Container name prefix to match
        all_containers: Include stopped containers

    Returns:
        List of ContainerInfo objects
    """
    cmd = ["docker", "ps", "--filter", f"name={prefix}",
           "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"]

    if all_containers:
        cmd.insert(2, "-a")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return []

        containers = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 4:
                containers.append(ContainerInfo(
                    container_id=parts[0],
                    name=parts[1],
                    image=parts[2],
                    status=parts[3]
                ))

        return containers

    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        return []


def wait_for_container(
    container: ContainerInfo,
    timeout: int = 60
) -> tuple[bool, int | None]:
    """
    Wait for a container to finish execution.

    Args:
        container: Container info
        timeout: Maximum wait time in seconds

    Returns:
        Tuple of (success, exit_code)
    """
    try:
        result = subprocess.run(
            ["docker", "wait", container.container_id],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            exit_code = int(result.stdout.strip())
            container.exit_code = exit_code
            container.status = "exited"
            return True, exit_code

        return False, None

    except subprocess.TimeoutExpired:
        return False, None
    except Exception as e:
        logger.error(f"Error waiting for container: {e}")
        return False, None
