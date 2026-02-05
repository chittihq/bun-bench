"""
Prompt templates for Bun-Bench inference.

This module contains the system and user prompt templates used for code fixing tasks,
along with utilities for formatting prompts for different model styles.
"""

from typing import Any

# System prompt for code fixing tasks
SYSTEM_PROMPT = """You are an expert software engineer specializing in the Bun JavaScript runtime.
Your task is to analyze bug reports or feature requests and generate patches to fix or implement them.

When generating patches:
1. Analyze the problem statement carefully
2. Identify the root cause of the bug or the requirements for the feature
3. Generate a minimal, focused patch that addresses the issue
4. Follow the existing code style and conventions
5. Include necessary test modifications if applicable

Output your solution as a unified diff patch that can be applied with `git apply` or `patch -p1`.

Format your patch response as follows:
```diff
--- a/path/to/file.ts
+++ b/path/to/file.ts
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```

Important guidelines:
- Use the exact file paths as they appear in the codebase
- Include sufficient context lines (typically 3) around changes
- Make minimal changes - only modify what's necessary to fix the issue
- Preserve existing formatting and style
- If multiple files need changes, include all of them in the same patch
- Do not include any explanation outside the diff block unless specifically asked
"""

# User prompt template for code fixing tasks
USER_PROMPT = """## Problem Statement

{problem_statement}

## Relevant Code Context

{code_context}

## Instructions

Please analyze the problem and generate a unified diff patch to fix this issue.
Wrap your patch in ```diff``` code blocks.
"""

# Alternative user prompt for minimal context
USER_PROMPT_MINIMAL = """## Problem Statement

{problem_statement}

## Instructions

Please analyze the problem and generate a unified diff patch to fix this issue.
Wrap your patch in ```diff``` code blocks.
"""

# Example patch format for reference
EXAMPLE_PATCH = '''```diff
--- a/src/bun.js/api/server.zig
+++ b/src/bun.js/api/server.zig
@@ -1234,7 +1234,7 @@ fn calculateContentLength(body: []const u8) usize {
-    return body.len;
+    return std.mem.len(body);
 }

--- a/test/js/bun/http/serve.test.ts
+++ b/test/js/bun/http/serve.test.ts
@@ -100,6 +100,15 @@ describe("Bun.serve", () => {
+  it("handles multi-byte UTF-8 content length correctly", async () => {
+    const response = new Response("hello world");
+    expect(response.headers.get("Content-Length")).toBe("15");
+  });
 });
```'''


def format_prompt(
    problem_statement: str,
    code_context: str | None = None,
    style: str = "default",
    additional_instructions: str | None = None,
) -> dict[str, str]:
    """
    Format prompts for different model styles and configurations.

    Args:
        problem_statement: The bug report or feature request description.
        code_context: Optional relevant code snippets or file contents.
        style: Prompt style - 'default', 'minimal', 'detailed', or 'chat'.
        additional_instructions: Optional extra instructions to append.

    Returns:
        Dictionary with 'system' and 'user' prompt strings.

    Raises:
        ValueError: If an unknown style is provided.
    """
    if style == "minimal":
        user_prompt = USER_PROMPT_MINIMAL.format(
            problem_statement=problem_statement
        )
    elif style == "default":
        user_prompt = USER_PROMPT.format(
            problem_statement=problem_statement,
            code_context=code_context if code_context else "No additional context provided.",
        )
    elif style == "detailed":
        user_prompt = USER_PROMPT.format(
            problem_statement=problem_statement,
            code_context=code_context if code_context else "No additional context provided.",
        )
        user_prompt += f"\n\n## Example Patch Format\n\n{EXAMPLE_PATCH}"
    elif style == "chat":
        # Chat style combines system and user into a single user message
        user_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{USER_PROMPT.format(problem_statement=problem_statement, code_context=code_context if code_context else 'No additional context provided.')}"
        return {"system": "", "user": user_prompt}
    else:
        raise ValueError(f"Unknown prompt style: {style}. Use 'default', 'minimal', 'detailed', or 'chat'.")

    if additional_instructions:
        user_prompt += f"\n\n## Additional Instructions\n\n{additional_instructions}"

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }


def format_for_openai(
    problem_statement: str,
    code_context: str | None = None,
    style: str = "default",
) -> list:
    """
    Format prompts as OpenAI-style messages list.

    Args:
        problem_statement: The bug report or feature request description.
        code_context: Optional relevant code snippets.
        style: Prompt style to use.

    Returns:
        List of message dictionaries for OpenAI API.
    """
    prompts = format_prompt(problem_statement, code_context, style)
    messages = []

    if prompts["system"]:
        messages.append({"role": "system", "content": prompts["system"]})

    messages.append({"role": "user", "content": prompts["user"]})

    return messages


def format_for_anthropic(
    problem_statement: str,
    code_context: str | None = None,
    style: str = "default",
) -> dict[str, Any]:
    """
    Format prompts for Anthropic API format.

    Args:
        problem_statement: The bug report or feature request description.
        code_context: Optional relevant code snippets.
        style: Prompt style to use.

    Returns:
        Dictionary with 'system' and 'messages' for Anthropic API.
    """
    prompts = format_prompt(problem_statement, code_context, style)

    return {
        "system": prompts["system"],
        "messages": [{"role": "user", "content": prompts["user"]}],
    }
