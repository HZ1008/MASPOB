from typing import Literal

from scripts.async_llm import create_llm_instance
import workspace.DROP.workflows.template.operator as operator


# Locally define DatasetType to avoid importing the full evaluator/benchmark stack
DatasetType = Literal[
    "HumanEval",
    "MBPP",
    "GSM8K",
    "MATH",
    "HotpotQA",
    "DROP",
    "LiveCodeBench",
]


class Workflow:
    """
    DROP Workflow: Solve → Format.
    - Solve: reading comprehension + numerical reasoning
    - Format: normalize the final answer
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

        # 2 operators for the workflow
        self.solve = operator.Solve(self.llm)
        self.format = operator.Format(self.llm) if hasattr(operator, 'Format') else None

    async def __call__(self, problem: str):
        """Workflow: Solve → Format.

        Returns:
            (final_answer: str, total_cost: float)
        """
        # Step 1: Solve
        try:
            solve_result = await self.solve(input=problem)
            solution = solve_result.get("answer", solve_result.get("response", ""))
        except Exception as e:
            return f"Error: {e}", self.llm.get_usage_summary()["total_cost"]

        # Step 2: Format (if available)
        if self.format and solution:
            try:
                format_result = await self.format(problem=problem, solution=solution)
                final_answer = format_result.get("solution", format_result.get("response", solution))
            except Exception:
                final_answer = solution
        else:
            final_answer = solution

        return final_answer, self.llm.get_usage_summary()["total_cost"]
