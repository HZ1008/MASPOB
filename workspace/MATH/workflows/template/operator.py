import concurrent.futures
import sys
import traceback
import atexit
import re
from typing import List, Optional

from scripts.formatter import FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.MATH.workflows.template.operator_an import *
from workspace.MATH.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio


from scripts.operators import Operator


# Prompt placeholder validation
def _extract_placeholders(template: str) -> set:
    """Extract all {placeholder} patterns from a template."""
    return set(re.findall(r'\{(\w+)\}', template))


def _validate_and_format(template: str, required_placeholders: List[str], **kwargs) -> str:
    """Validate placeholders and format the template safely.

    If required placeholders are missing from template, log an error and
    return a fallback that uses the original default template logic.

    Also checks for extra placeholders that are not provided in kwargs.
    """
    present = _extract_placeholders(template)
    required_set = set(required_placeholders)
    provided_set = set(kwargs.keys())

    missing = required_set - present
    # Check for placeholders in template that are not provided
    extra_in_template = present - provided_set

    if missing:
        logger.error(f"[MIPRO] Prompt template missing placeholders: {missing}")
        logger.error(f"[MIPRO] Template preview: {template[:200]}...")
        raise KeyError(f"Missing placeholders in prompt: {missing}")

    if extra_in_template:
        logger.error(f"[MIPRO] Prompt template has extra placeholders not provided: {extra_in_template}")
        logger.error(f"[MIPRO] Template preview: {template[:200]}...")
        raise KeyError(f"Extra placeholders in prompt not provided: {extra_in_template}")

    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"[MIPRO] format() KeyError: {e}")
        raise


# Global process pool for code execution with forced timeout support
_GLOBAL_PROCESS_EXECUTOR: Optional[concurrent.futures.ProcessPoolExecutor] = None
_PROCESS_EXECUTOR_MAX_WORKERS = 8


def get_global_process_executor() -> concurrent.futures.ProcessPoolExecutor:
    """Return the global ProcessPoolExecutor, creating it if necessary."""
    global _GLOBAL_PROCESS_EXECUTOR
    if _GLOBAL_PROCESS_EXECUTOR is None:
        _GLOBAL_PROCESS_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=_PROCESS_EXECUTOR_MAX_WORKERS)
        atexit.register(cleanup_process_executor)
    return _GLOBAL_PROCESS_EXECUTOR


def cleanup_process_executor():
    """Shut down the global ProcessPoolExecutor."""
    global _GLOBAL_PROCESS_EXECUTOR
    if _GLOBAL_PROCESS_EXECUTOR is not None:
        try:
            _GLOBAL_PROCESS_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _GLOBAL_PROCESS_EXECUTOR = None


def get_global_thread_executor():
    """Alias kept for backward compatibility."""
    return get_global_process_executor()


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

# Script template used by subprocess-based code execution
_SUBPROCESS_SCRIPT_TEMPLATE = '''
import sys
import json

code = """
{code}
"""

try:
    global_namespace = {{}}
    exec(code, global_namespace)
    if 'solve' in global_namespace and callable(global_namespace['solve']):
        result = global_namespace['solve']()
        print(json.dumps({{"status": "Success", "result": str(result)}}))
    else:
        print(json.dumps({{"status": "Error", "result": "Function 'solve' not found"}}))
except Exception as e:
    import traceback
    tb_str = traceback.format_exc()
    print(json.dumps({{"status": "Error", "result": f"Execution error: {{str(e)}}\\n{{tb_str[-500:]}}"}}))
'''


def run_code(code: str, timeout: int = 10) -> tuple:
    """Execute code in a subprocess with forced timeout support.

    Args:
        code: source code to execute
        timeout: timeout in seconds (default 10)
    Returns:
        (status, result) tuple
    """
    import subprocess
    import json

    # Quick check for disallowed imports before spawning subprocess
    disallowed_imports = [
        "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
        "pylab", "tkinter", "PyQt5", "wx", "pyglet"
    ]
    for lib in disallowed_imports:
        if f"import {lib}" in code or f"from {lib}" in code:
            return "Error", f"Prohibited import: {lib}"

    escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    script = _SUBPROCESS_SCRIPT_TEMPLATE.format(code=escaped_code)

    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        stdout = result.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                return data.get("status", "Error"), data.get("result", "Unknown error")
            except json.JSONDecodeError:
                # Non-JSON output (e.g. raw print statements)
                if result.returncode == 0:
                    return "Success", stdout
                else:
                    return "Error", f"Non-JSON output: {stdout[:200]}"
        else:
            stderr = result.stderr.strip()
            if stderr:
                return "Error", f"Stderr: {stderr[:500]}"
            return "Error", "No output from subprocess"

    except subprocess.TimeoutExpired:
        return "Error", f"Code execution timed out after {timeout} seconds"
    except Exception as e:
        return "Error", f"Subprocess error: {str(e)}"
    

class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)
        self.prompt = PYTHON_CODE_VERIFIER_PROMPT

    async def exec_code(self, code, timeout=15):
        """Execute code asynchronously; run_code handles timeout via subprocess."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, run_code, code, timeout)
            return result
        except Exception as e:
            return "Error", f"Execution error: {str(e)}"

    REQUIRED_PLACEHOLDERS = ["problem", "analysis", "feedback"]

    async def code_generate(self, problem, analysis, feedback, mode):
        """Generate code using the LLM."""
        prompt = _validate_and_format(
            self.prompt, self.REQUIRED_PLACEHOLDERS,
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    async def __call__(self, problem: str, analysis: str = "None"):
        """Generate code and execute; retries up to 3 times."""
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """
    REQUIRED_PLACEHOLDERS = ["question", "solutions"]

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)
        self.prompt = SC_ENSEMBLE_PROMPT

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = _validate_and_format(
            self.prompt, self.REQUIRED_PLACEHOLDERS,
            question=problem, solutions=solution_text
        )
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        # Fall back to first solution if parsing fails
        if answer and answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}
        else:
            return {"response": solutions[0] if solutions else ""}


class RefineAnswer(Operator):
    """Format a detailed solution from the code execution result (with LaTeX)."""
    REQUIRED_PLACEHOLDERS = ["input"]

    def __init__(self, llm: AsyncLLM, name: str = "RefineAnswer"):
        super().__init__(llm, name)
        self.prompt = REFINE_ANSWER_PROMPT

    async def __call__(self, input: str) -> dict:
        """
        Args:
            input: problem + code output text
        Returns:
            {"response": formatted solution}
        """
        prompt = _validate_and_format(self.prompt, self.REQUIRED_PLACEHOLDERS, input=input)
        response = await self._fill_node(RefineAnswerOp, prompt, mode="single_fill")
        return response


class GenerateSolution(Operator):
    """Generate a step-by-step math solution using pure LLM reasoning."""
    REQUIRED_PLACEHOLDERS = ["input"]

    def __init__(self, llm: AsyncLLM, name: str = "GenerateSolution"):
        super().__init__(llm, name)
        self.prompt = GENERATE_SOLUTION_PROMPT

    async def __call__(self, input: str) -> dict:
        """
        Args:
            input: math problem
        Returns:
            {"response": step-by-step solution}
        """
        prompt = _validate_and_format(self.prompt, self.REQUIRED_PLACEHOLDERS, input=input)
        response = await self._fill_node(GenerateSolutionOp, prompt, mode="single_fill")
        return response


class DetailedSolution(Operator):
    """Generate a comprehensive math solution with concept explanations."""
    REQUIRED_PLACEHOLDERS = ["input"]

    def __init__(self, llm: AsyncLLM, name: str = "DetailedSolution"):
        super().__init__(llm, name)
        self.prompt = DETAILED_SOLUTION_PROMPT

    async def __call__(self, input: str) -> dict:
        """
        Args:
            input: math problem
        Returns:
            {"response": detailed solution}
        """
        prompt = _validate_and_format(self.prompt, self.REQUIRED_PLACEHOLDERS, input=input)
        response = await self._fill_node(DetailedSolutionOp, prompt, mode="single_fill")
        return response