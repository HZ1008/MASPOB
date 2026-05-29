import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.DROP.workflows.template.operator_an import *
from workspace.DROP.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re


from scripts.operators import Operator



class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)
        # self.prompts_domain = generate_prompts_from_initial(llm=self.llm, initial_prompt=ANSWER_GENERATION_PROMPT, num_prompts=20, task_description="Your task: {input}")
        self.prompt = ANSWER_GENERATION_PROMPT

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = self.prompt.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response

class Solve(Operator):
    """Solve operator for DROP: reading comprehension + numerical reasoning."""

    def __init__(self, llm: AsyncLLM, name: str = "Solve"):
        super().__init__(llm, name)
        self.prompt = SOLVE_PROMPT

    async def __call__(self, input: str, mode: str = None) -> Dict[str, str]:
        prompt = self.prompt.format(input=input)
        response = await self._fill_node(SolveOp, prompt, mode="xml_fill")
        return response


class Format(Operator):
    """Format the final answer into a concise form."""

    def __init__(self, llm: AsyncLLM, name: str = "Format"):
        super().__init__(llm, name)
        self.prompt = FORMAT_PROMPT

    async def __call__(self, problem: str, solution: str, mode: str = None) -> Dict[str, str]:
        prompt = self.prompt.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode="single_fill")
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

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = self.prompt.format(solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}
