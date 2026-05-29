from typing import List, Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType


class Workflow:
    """
    GSM8K Workflow: Solve×3 → ScEnsemble → Programmer → Extract.
    1. Solve×3: generate multiple solutions
    2. ScEnsemble: vote for the best solution
    3. Programmer: verify with code execution
    4. Extract: extract the final numerical answer
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
        self.solve = operator.Solve(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.extract = operator.Extract(self.llm)

    async def __call__(self, problem: str):
        """
        Workflow: Solve×3 → ScEnsemble → Programmer → Extract

        Returns:
            (final_answer: str, total_cost: float)
        """
        # Step 1: Generate multiple solutions
        solutions = []
        for _ in range(3):
            try:
                solution = await self.solve(input=problem)
                solutions.append(solution.get('response', ''))
            except Exception:
                continue

        # No solutions generated
        if not solutions:
            return "", self.llm.get_usage_summary()["total_cost"]

        # Step 2: Vote for the best solution
        if len(solutions) == 1:
            best_solution = solutions[0]
        else:
            try:
                ensemble_result = await self.sc_ensemble(
                    solutions=solutions,
                    problem=problem,
                )
                best_solution = ensemble_result.get("response", solutions[0])
            except Exception:
                best_solution = solutions[0]

        # Step 3: Verify with Programmer
        try:
            prog_result = await self.programmer(
                problem=problem,
                analysis=best_solution
            )
            prog_output = prog_result.get("output", "")
        except Exception:
            prog_output = ""

        # Step 4: Extract final answer
        try:
            extract_input = best_solution + "\nProgrammer solution: " + prog_output
            final_result = await self.extract(input=extract_input)
            final_answer = final_result.get('response', '')
        except Exception:
            final_answer = best_solution

        return final_answer, self.llm.get_usage_summary()["total_cost"]