"""
Prompt templates for Bun-Bench inference.
"""

from typing import Dict, Any, Optional

# Simple system prompt
SYSTEM_PROMPT = """You are an expert Bun.js developer. Fix the provided code.

Return ONLY the complete fixed file content.
Do NOT output diffs, patches, or unified diffs.
Do NOT use ```diff code blocks.
Do NOT omit unchanged code.
Do NOT output explanations."""

# Simple user prompt
USER_PROMPT = """## Bug Description

{problem_statement}

## Code

{code_context}

## Output

Return ONLY the complete fixed source code in this format:

```typescript
// File: src/<filename>
// Write the COMPLETE fixed file content here - do NOT use diff format
// Include ALL code, not just the changes
export function fixedFunction() {{
  // complete implementation
}}
```

IMPORTANT: Do NOT use ```diff or ```patch - only use ```typescript"""


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
