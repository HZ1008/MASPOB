# Prompt Configuration for MASPOB Workflows

from typing import Dict, List

# Import prompt templates from scripts
from scripts.prompts.prompt import *


# Prompt configs: templates and generation goals
# Note: 'goal' describes what the prompt should instruct the model to do and what format to output
PROMPT_CONFIGS = {
    "ANSWER_GENERATION_PROMPT": {
        "template": ANSWER_GENERATION_PROMPT,
        "goal": "Instruct the model to answer a question. The prompt should tell the model to put its reasoning in <thought> tags and final answer in <answer> tags.",
    },
    "SOLVE_PROMPT": {
        "template": SOLVE_PROMPT,
        "goal": "Instruct the model to read a passage and answer a question by calculating or identifying the specific answer. The prompt should tell the model to put reasoning in <thought> tags and final answer in <answer> tags.",
    },
    "FORMAT_PROMPT": {
        "template": FORMAT_PROMPT,
        "goal": "Instruct the model to format the final answer to be concise and accurate. The answer should be a short phrase, name, or few words without additional explanation.",
    },
    "FORMAT_ANSWER_PROMPT": {
        "template": FORMAT_ANSWER_PROMPT,
        "goal": "Instruct the model to extract and format a concise final answer from the given question and best answer. The answer should be a short phrase, name, or number in <answer> tags.",
    },
    "SC_ENSEMBLE_PROMPT": {
        "template": SC_ENSEMBLE_PROMPT,
        "goal": "Instruct the model to select the most frequent answer from multiple candidates. The prompt should tell the model to put analysis in <thought> tags and the letter (A/B/C) in <solution_letter> tags.",
    },
    "MD_ENSEMBLE_PROMPT": {
        "template": MD_ENSEMBLE_PROMPT,
        "goal": "Instruct the model to select the best solution from multiple candidates. The prompt should tell the model to put analysis in <thought> tags and the letter (A/B/C) in <solution_letter> tags.",
    },
    "REVIEW_PROMPT": {
        "template": REVIEW_PROMPT,
        "goal": "Instruct the model to verify if a solution is correct. The prompt should tell the model to output <review_result>true/false</review_result> and <feedback>explanation</feedback>.",
    },
    "REVISE_PROMPT": {
        "template": REVISE_PROMPT,
        "goal": "Instruct the model to revise an incorrect solution based on feedback. The prompt should tell the model to output corrected code.",
    },
    "REFLECTION_ON_PUBLIC_TEST_PROMPT": {
        "template": REFLECTION_ON_PUBLIC_TEST_PROMPT,
        "goal": "Instruct the model to analyze why code failed tests and fix it. The prompt should tell the model to output in <reflection_and_solution> tags.",
    },
    # GSM8K-specific prompts
    "GSM8K_SOLVE_PROMPT": {
        "template": GSM8K_SOLVE_PROMPT,
        "goal": "Instruct the model to solve a math problem step by step with clear breakdown of given information, calculations, and final numerical answer marked with ** **.",
    },
    "GSM8K_EXTRACT_PROMPT": {
        "template": GSM8K_EXTRACT_PROMPT,
        "goal": "Instruct the model to extract only the final numerical answer from a solution. Should return ONLY the number with no text or symbols.",
    },
    # MATH-specific prompts
    "PYTHON_CODE_VERIFIER_PROMPT": {
        "template": PYTHON_CODE_VERIFIER_PROMPT,
        "goal": "Instruct the model to write Python code to solve a mathematical problem. The code should define a function named 'solve' that returns the result.",
    },
    "REFINE_ANSWER_PROMPT": {
        "template": REFINE_ANSWER_PROMPT,
        "goal": "Instruct the model to refine and format a mathematical solution based on code execution output. The final answer should be in \\boxed{} LaTeX notation.",
    },
    "GENERATE_SOLUTION_PROMPT": {
        "template": GENERATE_SOLUTION_PROMPT,
        "goal": "Instruct the model to solve a mathematical problem step by step with LaTeX notation. The final answer should be in \\boxed{} notation.",
    },
    "DETAILED_SOLUTION_PROMPT": {
        "template": DETAILED_SOLUTION_PROMPT,
        "goal": "Instruct the model to provide a comprehensive, educational solution to a mathematical problem with detailed explanations and LaTeX notation.",
    },
    # MBPP code generation prompts
    "CODE_GENERATE_PROMPT": {
        "template": CODE_GENERATE_PROMPT,
        "goal": "Instruct the model to generate a Python function to solve a coding problem. The function name should match the specified entry point.",
    },
    "FIX_CODE_PROMPT": {
        "template": FIX_CODE_PROMPT,
        "goal": "Instruct the model to analyze why code failed tests and fix it. Keep the function signature unchanged.",
    },
    # HumanEval CustomCodeGenerate prompts
    "CUSTOM_CODE_GENERATE_PROMPT": {
        "template": CUSTOM_CODE_GENERATE_PROMPT,
        "goal": "Instruct the model to generate a Python function for HumanEval. The prompt must use {problem} and {entry_point} placeholders. Output should be clean Python code only.",
    },
    # HumanEval code validation prompts
    "VALIDATE_CODE_PROMPT": {
        "template": VALIDATE_CODE_PROMPT,
        "goal": "Instruct the model to validate Python code by checking function definition, return statement, syntax validity, and completeness. Output should be only 'VALID' or 'INVALID'.",
    },
}

# For backward compatibility
PROMPT_TYPES = [cfg["template"] for cfg in PROMPT_CONFIGS.values()]
PROMPT_NAMES = list(PROMPT_CONFIGS.keys())

