from typing import List, Literal
import workspace.HotpotQA.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType


class Workflow:
    """
    HotpotQA Workflow: AnswerGenerate×3 → ScEnsemble → FormatAnswer.
    Designed for multi-hop QA; evaluated with F1 score.
    """
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Operators
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.format_answer = operator.FormatAnswer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []

        for _ in range(3):
            initial_response = await self.answer_generate(input=problem)
            initial_answer = initial_response['answer']
            solutions.append(initial_answer)

        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
        best_answer = ensemble_result['response']

        format_result = await self.format_answer(
            question=problem,
            best_answer=best_answer
        )

        final_answer = format_result['answer']
        return final_answer, self.llm.get_usage_summary()["total_cost"]