"""
Utility functions for Bun-Bench inference.

This module provides utilities for extracting patches from model responses,
repairing common patch issues, and counting tokens.
"""

import re
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_patch(response: str) -> Optional[str]:
    """
    Extract a unified diff patch from a model response.

    Handles multiple formats:
    - ```diff``` code blocks
    - <patch>...</patch> tags
    - <diff>...</diff> tags
    - Raw diff content starting with --- or diff --git

    Args:
        response: The full model response text.

    Returns:
        The extracted patch string, or None if no patch found.
    """
    if not response:
        return None

    # Try to extract from ```diff``` code blocks (most common)
    diff_block_pattern = r'```(?:diff|patch|unified)?\s*\n(.*?)```'
    matches = re.findall(diff_block_pattern, response, re.DOTALL | re.IGNORECASE)

    if matches:
        # Filter to only include actual diff content
        for match in matches:
            if _looks_like_diff(match):
                return match.strip()

    # Try <patch>...</patch> tags
    patch_tag_pattern = r'<patch>\s*(.*?)\s*</patch>'
    matches = re.findall(patch_tag_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        for match in matches:
            if _looks_like_diff(match):
                return match.strip()

    # Try <diff>...</diff> tags
    diff_tag_pattern = r'<diff>\s*(.*?)\s*</diff>'
    matches = re.findall(diff_tag_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        for match in matches:
            if _looks_like_diff(match):
                return match.strip()

    # Try to find raw diff content
    raw_diff = _extract_raw_diff(response)
    if raw_diff:
        return raw_diff

    logger.warning("No patch found in response")
    return None


def _looks_like_diff(text: str) -> bool:
    """
    Check if text appears to be a unified diff.

    Args:
        text: Text to check.

    Returns:
        True if text looks like a diff, False otherwise.
    """
    indicators = [
        r'^---\s+\S+',  # --- a/file
        r'^\+\+\+\s+\S+',  # +++ b/file
        r'^@@\s+-\d+',  # @@ -line
        r'^diff\s+--git',  # diff --git
        r'^Index:\s+',  # Index: (svn style)
    ]

    for pattern in indicators:
        if re.search(pattern, text, re.MULTILINE):
            return True

    return False


def _extract_raw_diff(response: str) -> Optional[str]:
    """
    Extract raw diff content from response without code blocks.

    Args:
        response: The full response text.

    Returns:
        Extracted diff or None.
    """
    lines = response.split('\n')
    diff_lines = []
    in_diff = False

    for line in lines:
        # Start of diff
        if re.match(r'^(diff\s+--git|---\s+a/|Index:)', line):
            in_diff = True

        if in_diff:
            # Check for end of diff (non-diff content)
            if line and not _is_diff_line(line):
                # Allow some non-diff lines (empty, comments)
                if not re.match(r'^(\s*$|#|//)', line):
                    # Check if we have a complete diff
                    if diff_lines and _looks_like_diff('\n'.join(diff_lines)):
                        break

            diff_lines.append(line)

    if diff_lines and _looks_like_diff('\n'.join(diff_lines)):
        return '\n'.join(diff_lines).strip()

    return None


def _is_diff_line(line: str) -> bool:
    """
    Check if a line is a valid diff line.

    Args:
        line: Line to check.

    Returns:
        True if line is a valid diff line.
    """
    patterns = [
        r'^diff\s+--git',
        r'^index\s+[0-9a-f]+',
        r'^---\s+',
        r'^\+\+\+\s+',
        r'^@@\s+',
        r'^[-+\s]',  # Changed/context lines
        r'^\\ No newline',
        r'^Binary files',
        r'^new file mode',
        r'^deleted file mode',
        r'^similarity index',
        r'^rename from',
        r'^rename to',
        r'^old mode',
        r'^new mode',
    ]

    for pattern in patterns:
        if re.match(pattern, line):
            return True

    return False


def repair_patch(patch: str) -> str:
    """
    Attempt to repair common issues in patches.

    Fixes:
    - Missing file prefixes (a/, b/)
    - Incorrect line numbers in hunks
    - Missing newlines at end
    - Windows line endings
    - Whitespace issues in hunk headers

    Args:
        patch: The patch string to repair.

    Returns:
        The repaired patch string.
    """
    if not patch:
        return patch

    # Normalize line endings
    patch = patch.replace('\r\n', '\n').replace('\r', '\n')

    lines = patch.split('\n')
    repaired_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix --- lines missing a/ prefix
        if re.match(r'^---\s+(?!a/)(\S+)', line):
            match = re.match(r'^---\s+(\S+)(.*)', line)
            if match:
                filepath = match.group(1)
                rest = match.group(2)
                # Don't add prefix if it's /dev/null
                if filepath != '/dev/null':
                    line = f'--- a/{filepath}{rest}'
                    logger.debug(f"Fixed --- line: {line}")

        # Fix +++ lines missing b/ prefix
        elif re.match(r'^\+\+\+\s+(?!b/)(\S+)', line):
            match = re.match(r'^\+\+\+\s+(\S+)(.*)', line)
            if match:
                filepath = match.group(1)
                rest = match.group(2)
                if filepath != '/dev/null':
                    line = f'+++ b/{filepath}{rest}'
                    logger.debug(f"Fixed +++ line: {line}")

        # Fix hunk headers with incorrect spacing
        elif re.match(r'^@@.*@@', line):
            # Normalize hunk header format
            match = re.match(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@(.*)$', line)
            if match:
                old_start = match.group(1)
                old_count = match.group(2) or '1'
                new_start = match.group(3)
                new_count = match.group(4) or '1'
                context = match.group(5)
                line = f'@@ -{old_start},{old_count} +{new_start},{new_count} @@{context}'

        # Fix context lines that lost their leading space
        elif i > 0 and repaired_lines:
            prev_line = repaired_lines[-1]
            # If previous was a hunk header and this line doesn't start with diff marker
            if prev_line.startswith('@@') and line and not line[0] in ['+', '-', ' ', '@', '\\']:
                # Likely a context line missing its space
                line = ' ' + line
                logger.debug(f"Added missing space to context line")

        repaired_lines.append(line)
        i += 1

    # Ensure patch ends with newline
    result = '\n'.join(repaired_lines)
    if result and not result.endswith('\n'):
        result += '\n'

    return result


def validate_patch(patch: str) -> Tuple[bool, List[str]]:
    """
    Validate a patch for common issues.

    Args:
        patch: The patch to validate.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    if not patch:
        return False, ["Empty patch"]

    if not _looks_like_diff(patch):
        return False, ["Does not appear to be a valid diff"]

    lines = patch.split('\n')

    has_file_header = False
    has_hunk = False
    in_hunk = False
    hunk_additions = 0
    hunk_deletions = 0
    hunk_context = 0
    expected_old_lines = 0
    expected_new_lines = 0

    for i, line in enumerate(lines):
        # Check for file headers
        if line.startswith('---'):
            has_file_header = True
            if not re.match(r'^---\s+(a/\S+|/dev/null)', line):
                issues.append(f"Line {i+1}: --- line may be missing 'a/' prefix")

        elif line.startswith('+++'):
            if not re.match(r'^\+\+\+\s+(b/\S+|/dev/null)', line):
                issues.append(f"Line {i+1}: +++ line may be missing 'b/' prefix")

        # Check hunk headers
        elif line.startswith('@@'):
            has_hunk = True
            in_hunk = True
            hunk_additions = 0
            hunk_deletions = 0
            hunk_context = 0

            match = re.match(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@', line)
            if match:
                expected_old_lines = int(match.group(2) or 1)
                expected_new_lines = int(match.group(4) or 1)
            else:
                issues.append(f"Line {i+1}: Invalid hunk header format")

        # Count lines in hunk
        elif in_hunk and line:
            if line.startswith('+') and not line.startswith('+++'):
                hunk_additions += 1
            elif line.startswith('-') and not line.startswith('---'):
                hunk_deletions += 1
            elif line.startswith(' '):
                hunk_context += 1
            elif line.startswith('\\'):
                pass  # "\ No newline at end of file"
            elif line.startswith('diff ') or line.startswith('index '):
                in_hunk = False
            else:
                issues.append(f"Line {i+1}: Unexpected line in hunk: {line[:50]}")

    if not has_file_header:
        issues.append("Missing file header (--- and +++ lines)")

    if not has_hunk:
        issues.append("Missing hunk (no @@ markers found)")

    return len(issues) == 0, issues


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text for a given model.

    Uses tiktoken for OpenAI models or estimates for others.

    Args:
        text: Text to count tokens for.
        model: Model name for encoding selection.

    Returns:
        Estimated token count.
    """
    try:
        import tiktoken

        # Map model names to encodings
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        elif "claude" in model.lower():
            # Claude uses similar tokenization to GPT models
            # Use cl100k_base as approximation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except ImportError:
        # Fallback: estimate ~4 characters per token
        logger.warning("tiktoken not installed, using character-based estimation")
        return len(text) // 4


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """
    Estimate API cost based on token counts.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model name.

    Returns:
        Estimated cost in USD.
    """
    # Pricing per 1K tokens (as of 2026, update as needed)
    pricing = {
        # OpenAI models
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        # Anthropic models
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-opus-4-5-20251101": {"input": 0.015, "output": 0.075},
    }

    # Find matching pricing
    model_lower = model.lower()
    price_info = None

    for model_key, prices in pricing.items():
        if model_key in model_lower:
            price_info = prices
            break

    if not price_info:
        logger.warning(f"Unknown model {model}, using default pricing")
        price_info = {"input": 0.01, "output": 0.03}

    input_cost = (input_tokens / 1000) * price_info["input"]
    output_cost = (output_tokens / 1000) * price_info["output"]

    return input_cost + output_cost


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate.
        max_tokens: Maximum number of tokens.
        model: Model name for encoding.

    Returns:
        Truncated text.
    """
    try:
        import tiktoken

        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)

    except ImportError:
        # Fallback: estimate 4 chars per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
