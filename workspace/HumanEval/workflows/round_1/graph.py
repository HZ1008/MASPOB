from typing import List, Literal
import workspace.HumanEval.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType


class Workflow:
    """
    HumanEval Workflow: CustomCodeGenerate×3 → ScEnsemble → Test.
    MASPOB injects optimized prompts via operator.prompt attributes.
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

        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Workflow: CustomCodeGenerate×3 → ScEnsemble → Test.

        Returns:
            (final_code: str, total_cost: float)
        """
        # Step 1: Generate multiple solutions
        solutions = []
        for _ in range(3):  # Generate 3 different solutions
            try:
                solution = await self.custom_code_generate(
                    problem=problem,
                    entry_point=entry_point
                )
                if solution.get('response'):
                    solutions.append(solution['response'])
            except Exception:
                continue

        if not solutions:
            return "", self.llm.get_usage_summary()["total_cost"]

        # Step 2: Ensemble to pick the best solution
        if len(solutions) >= 2:
            try:
                best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
                best_code = best_solution.get('response', solutions[0])
            except Exception:
                best_code = solutions[0]
        else:
            best_code = solutions[0]

        # Step 3: Test and refine the selected solution
        try:
            test_result = await self.test(
                problem=problem,
                solution=best_code,
                entry_point=entry_point,
                test_loop=5
            )
            return test_result.get("solution", best_code), self.llm.get_usage_summary()["total_cost"]
        except Exception:
            pass

        # Fall back to best ensemble code
        return best_code, self.llm.get_usage_summary()["total_cost"]