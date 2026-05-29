import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.HotpotQA.workflows.template.operator_an import *
from workspace.HotpotQA.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re


from scripts.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)
        self.prompt = FORMAT_PROMPT

    async def __call__(self, input, instruction=None):
        actual_instruction = instruction if instruction is not None else self.prompt
        prompt = actual_instruction + "\n" + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)
        self.prompt = ANSWER_GENERATION_PROMPT

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = self.prompt.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
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

    async def __call__(self, solutions: List[str], problem: str = ""):
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


class FormatAnswer(Operator):
    """Normalize the ensemble-selected answer into a concise final form."""

    def __init__(self, llm: AsyncLLM, name: str = "FormatAnswer"):
        super().__init__(llm, name)
        self.prompt = FORMAT_ANSWER_PROMPT

    async def __call__(self, question: str, best_answer: str) -> Dict[str, str]:
        """
        Args:
            question: original question
            best_answer: answer selected by ScEnsemble
        Returns:
            {"answer": formatted concise answer}
        """
        input_text = f"Question: {question}\nBest answer: {best_answer}"
        prompt = self.prompt.format(input=input_text)
        response = await self._fill_node(FormatAnswerOp, prompt, mode="xml_fill")
        return {"answer": response.get("answer", best_answer)}