import sys
import traceback
from typing import List

from workspace.MBPP.workflows.template.operator_an import *
from workspace.MBPP.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger

from scripts.utils.code import extract_test_cases_from_jsonl, test_case_2_test_function

from scripts.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response


class CodeGenerate(Operator):
    """Generate Python code for MBPP problems."""
    def __init__(self, llm: AsyncLLM, name: str = "CodeGenerate"):
        super().__init__(llm, name)
        self.prompt = CODE_GENERATE_PROMPT

    async def __call__(self, problem: str, entry_point: str) -> dict:
        """
        Args:
            problem: problem description
            entry_point: function name/signature
        Returns:
            {"code": generated code}
        """
        prompt = self.prompt.format(problem=problem, entry_point=entry_point)
        response = await self._fill_node(CodeGenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response


class FixCode(Operator):
    """Fix code based on error messages."""
    def __init__(self, llm: AsyncLLM, name: str = "FixCode"):
        super().__init__(llm, name)
        self.prompt = FIX_CODE_PROMPT

    async def __call__(self, problem: str, solution: str, error: str, entry_point: str) -> dict:
        """
        Args:
            problem: original problem
            solution: failing code
            error: error message
            entry_point: function name
        Returns:
            {"code": fixed code}
        """
        prompt = self.prompt.format(problem=problem, solution=solution, error=error)
        response = await self._fill_node(FixCodeOp, prompt, mode="code_fill", function_name=entry_point)
        return response


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
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

class Test(Operator):
    """Test code against public test cases and iteratively fix errors."""
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)
        self.prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT

    def exec_code(self, solution, entry_point):
        """Run test cases and return failures or 'no error'."""
        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="MBPP")

        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                return {"exec_fail_case": str(e)}

        if fail_cases:
            return fail_cases
        else:
            return "no error"

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 5
    ):
        """
        Test and iteratively fix the solution.

        Args:
            problem: problem description
            solution: code to test
            entry_point: function name
            test_loop: max retry iterations
        Returns:
            {"result": bool, "solution": final code}
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif isinstance(result, dict) and "exec_fail_case" in result:
                error_msg = result["exec_fail_case"]
                prompt = self.prompt.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {error_msg}",
                    test_fail="executed unsuccessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response.get("response", solution)
            else:
                prompt = self.prompt.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response.get("response", solution)

        # Final test
        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}