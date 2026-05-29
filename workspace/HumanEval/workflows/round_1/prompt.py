# HumanEval Workflow Prompts

VALIDATE_CODE_PROMPT = """Analyze the given Python code and verify:
1. The function is properly defined with the specified name: {function_name}
2. The code contains a return statement
3. The code has valid Python syntax
4. The code is complete (no missing parts)

Code to validate:
{code}

Return only "VALID" if all checks pass, or "INVALID" if any check fails.
Do not include any other text in the response."""

CODE_GENERATE_PROMPT = """Complete the following Python function.
Return only the function implementation with proper syntax.

{problem}

Function name: {entry_point}
"""
