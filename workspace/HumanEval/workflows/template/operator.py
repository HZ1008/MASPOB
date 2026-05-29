import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.HumanEval.workflows.template.operator_an import *
from workspace.HumanEval.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio

from scripts.utils.code import extract_test_cases_from_jsonl, test_case_2_test_function
from scripts.prompts.prompt import CUSTOM_CODE_GENERATE_PROMPT, REFLECTION_ON_PUBLIC_TEST_PROMPT


from scripts.operators import Operator



class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
class CustomCodeGenerate(Operator):
    """Code generation operator for HumanEval; uses {problem} and {entry_point} placeholders."""
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)
        self.prompt = CUSTOM_CODE_GENERATE_PROMPT

    async def __call__(self, problem, entry_point):
        full_prompt = self.prompt.format(problem=problem, entry_point=entry_point)
        response = await self._fill_node(GenerateOp, full_prompt, mode="code_fill", function_name=entry_point)
        return response


class ValidateCode(Operator):
    """Validate generated code: checks function definition, return statement, syntax, completeness."""
    def __init__(self, llm: AsyncLLM, name: str = "ValidateCode"):
        super().__init__(llm, name)
        self.prompt = """Analyze the given Python code and verify:
1. The function is properly defined with the specified name: {function_name}
2. The code contains a return statement
3. The code has valid Python syntax
4. The code is complete (no missing parts)

Code to validate:
{code}

Return only "VALID" if all checks pass, or "INVALID" if any check fails.
Do not include any other text in the response."""

    async def __call__(self, code: str, entry_point: str):
        prompt = self.prompt.format(code=code, function_name=entry_point)
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        result = response.get("response", "").strip().upper()
        is_valid = "VALID" in result and "INVALID" not in result
        return {"is_valid": is_valid, "response": result}


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

        return {"response": solutions[answer_mapping[answer]]}

class Test(Operator):
    """Test and iteratively fix generated code against public test cases."""
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)
        self.prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT

    def exec_code(self, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="HumanEval")

        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
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
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 5
    ):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = self.prompt.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]
            else:
                prompt = self.prompt.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]

        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}