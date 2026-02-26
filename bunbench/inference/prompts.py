"""
Prompt templates for Bun-Bench inference.

This module contains the system and user prompt templates used for code fixing tasks,
along with utilities for formatting prompts for different model styles.
"""

from typing import Dict, Any, Optional

# System prompt for code fixing tasks
SYSTEM_PROMPT = """You are an expert Bun JavaScript runtime developer.

Your task is to generate a PATCH (diff) to fix bugs in the provided source code.

STRICT RULES - MUST FOLLOW:
1. ONLY modify the files shown in the code context below
2. Do NOT add new functions, classes, or files that don't exist
3. Do NOT rewrite existing code - only fix the specific bug
4. Match the EXACT indentation, spacing, and coding style
5. The patch must apply cleanly with `git apply`

Your response must ONLY contain the diff patch in ```diff``` blocks.
Do NOT include any explanation, comments, or text outside the diff blocks."""

# User prompt template for code fixing tasks
USER_PROMPT = """## BUG TO FIX

{problem_statement}

## SOURCE CODE (FIX THIS EXACT CODE)

{code_context}

## CRITICAL INSTRUCTIONS

1. Look at the source code ABOVE - this is the ONLY code you can modify
2. The bug is: {problem_statement}
3. Generate a patch that fixes ONLY this specific bug
4. Do NOT create new classes, functions, or rewrite the code
5. Do NOT change anything except what's needed to fix the bug
6. Match the exact indentation and style of the existing code
7. Your patch must start with --- a/src/xxx and +++ b/src/xxx

Output ONLY the diff patch in ```diff``` blocks. No explanation needed.
"""

# Alternative user prompt for minimal context
USER_PROMPT_MINIMAL = """## BUG TO FIX

{problem_statement}

## Instructions

1. Analyze the bug described above
2. Generate a minimal patch to fix ONLY this bug
3. Do NOT rewrite or add new code
4. Match the existing code style exactly

Output ONLY the diff patch in ```diff``` blocks.
"""

# Example patch format for reference
EXAMPLE_PATCH = '''```diff
--- a/src/server.ts
+++ b/src/server.ts
@@ -5,7 +5,7 @@ const server = Bun.serve({
   fetch(req) {
     if (req.url.includes("/api/user")) {
-      return new Response(JSON.stringify({name: "test"}));
+      return new Response(JSON.stringify({name: "test"}), { headers: { "Content-Type": "application/json" } });
     }
     if (req.url.includes("/api/items")) {
-      return new Response(JSON.stringify([]));
+      return new Response(JSON.stringify([]), { headers: { "Content-Type": "application/json" } });
     }
     return new Response("Not Found");
   }
```'''


def format_prompt(
    problem_statement: str,
    code_context: Optional[str] = None,
    style: str = "default",
    additional_instructions: Optional[str] = None,
) -> Dict[str, str]:
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
    code_context: Optional[str] = None,
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
    code_context: Optional[str] = None,
    style: str = "default",
) -> Dict[str, Any]:
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
