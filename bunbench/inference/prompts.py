"""
Prompt templates for Bun-Bench inference.
"""

from typing import Dict, Any, Optional

# Simple system prompt
SYSTEM_PROMPT = """You are an expert Bun.js developer.

IMPORTANT: Read the task description carefully to understand what needs to be fixed.

Return ONLY the complete fixed file content.
Do NOT output diffs, patches, or unified diffs.
Do NOT use ```diff code blocks.
Do NOT output explanations."""

# Simple user prompt
USER_PROMPT = """## Task Description

{problem_statement}

## Current Code

{code_context}

## Output

Return ONLY the complete fixed file in this format:

```typescript
// File: <filename>
// Write the COMPLETE file content here
export function example() {{
  // complete implementation
}}
```

IMPORTANT - Check the task description:

1. If the task involves:
   - inline snapshots, toMatchInlineSnapshot(), snapshot mismatches
   - updating snapshots or "stale" snapshots
   - Tests are failing but the source code is correct
   THEN: Update only the TEST FILE's inline snapshot values to match current behavior
   - Do NOT modify source files
   - Only update the toMatchInlineSnapshot() expected values

2. If the task involves:
   - fixing bugs in source code
   - broken/incorrect functionality
   THEN: Fix the SOURCE CODE files"""


def format_prompt(
    problem_statement: str,
    code_context: str = "",
    style: str = "default",
    additional_instructions: str = None,
) -> Dict[str, str]:
    """Format prompts for inference."""

    if code_context:
        user_prompt = USER_PROMPT.format(
            problem_statement=problem_statement,
            code_context=code_context,
        )
    else:
        user_prompt = USER_PROMPT.format(
            problem_statement=problem_statement,
            code_context="No code provided.",
        )

    if additional_instructions:
        user_prompt += f"\n\n{additional_instructions}"

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }


def format_for_openai(
    problem_statement: str,
    code_context: str = "",
    style: str = "default",
) -> list:
    """Format prompts for OpenAI API."""
    prompts = format_prompt(problem_statement, code_context, style)
    messages = []
    if prompts["system"]:
        messages.append({"role": "system", "content": prompts["system"]})
    messages.append({"role": "user", "content": prompts["user"]})
    return messages


def format_for_anthropic(
    problem_statement: str,
    code_context: str = "",
    style: str = "default",
) -> Dict[str, Any]:
    """Format prompts for Anthropic API."""
    prompts = format_prompt(problem_statement, code_context, style)
    return {
        "system": prompts["system"],
        "messages": [{"role": "user", "content": prompts["user"]}],
    }
