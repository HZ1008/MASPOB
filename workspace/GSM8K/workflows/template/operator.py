import concurrent.futures
import sys
import traceback
import atexit
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.GSM8K.workflows.template.operator_an import *
from workspace.GSM8K.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio


from scripts.operators import Operator


# Global ProcessPoolExecutor to avoid spawning a new process per call
_GLOBAL_EXECUTOR: Optional[concurrent.futures.ProcessPoolExecutor] = None
_EXECUTOR_MAX_WORKERS = 4


def get_global_executor() -> concurrent.futures.ProcessPoolExecutor:
    """Return the global ProcessPoolExecutor, creating it if necessary."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=_EXECUTOR_MAX_WORKERS)
        atexit.register(cleanup_executor)
    return _GLOBAL_EXECUTOR


def cleanup_executor():
    """Shut down the global ProcessPoolExecutor."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is not None:
        try:
            _GLOBAL_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _GLOBAL_EXECUTOR = None


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class Solve(Operator):
    """GSM8K solve operator; prompt can be overridden by MASPOB."""
    def __init__(self, llm: AsyncLLM, name: str = "Solve"):
        super().__init__(llm, name)
        self.prompt = """
Solve this math problem step by step. Show all your work clearly and end with a numerical answer.
Break down the solution into:
1. Given information
2. Step-by-step calculations
3. Final numerical answer clearly marked with ** **

Make sure to:
- Include all mathematical operations
- Show intermediate calculations
- Double check your arithmetic
- Consider all given values in the problem
- Verify your answer makes logical sense

Problem:
{input}
"""

    async def __call__(self, input: str):
        prompt = self.prompt.format(input=input)
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class Extract(Operator):
    """GSM8K answer extraction operator; prompt can be overridden by MASPOB."""
    def __init__(self, llm: AsyncLLM, name: str = "Extract"):
        super().__init__(llm, name)
        self.prompt = """
Extract only the final numerical answer from the solution. Return ONLY the number, with no text or symbols.
If there are multiple numbers, extract the one marked with ** **.
Compare this answer with the Programmer solution provided and return the Programmer's solution if it differs significantly.

Solution to extract from:
{input}
"""

    async def __call__(self, input: str):
        prompt = self.prompt.format(input=input)
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

def run_code(code):
    try:
        import inspect
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            solve_func = global_namespace['solve']
            # Check function signature
            sig = inspect.signature(solve_func)
            required_params = [
                p for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
            ]
            if required_params:
                # Function has required parameters, cannot call without args
                return "Error", f"Function 'solve' requires arguments: {[p.name for p in required_params]}"
            result = solve_func()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class Programmer(Operator):
    """Code generation + execution operator; prompt can be overridden by MASPOB."""
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)
        self.prompt = PYTHON_CODE_VERIFIER_PROMPT

    async def exec_code(self, code, timeout=30):
        """Execute code asynchronously using the global process pool."""
        loop = asyncio.get_running_loop()
        executor = get_global_executor()
        try:
            # Submit run_code task to the global process pool
            future = loop.run_in_executor(executor, run_code, code)
            # Wait for the task to complete or timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return "Error", "Code execution timed out"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode):
        """Generate code using the LLM."""
        prompt = self.prompt.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
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
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


class ScEnsemble(Operator):
    """
    Self-consistency ensemble operator.
    Refs: arxiv.org/abs/2203.11171, arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)
        self.prompt = SC_ENSEMBLE_PROMPT

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = self.prompt.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        # Fall back to first solution if parsing fails
        if answer and answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}
        else:
            return {"response": solutions[0] if solutions else ""}