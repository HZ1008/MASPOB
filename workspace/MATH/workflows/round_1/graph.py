import asyncio
from typing import List
import workspace.MATH.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType


class Workflow:
    """
    MATH Workflow: parallel branches → ScEnsemble.
    Branches: Programmer→RefineAnswer, DetailedSolution, GenerateSolution×2.
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

        self.programmer = operator.Programmer(self.llm)
        self.refine_answer = operator.RefineAnswer(self.llm)
        self.generate_solution = operator.GenerateSolution(self.llm)
        self.detailed_solution = operator.DetailedSolution(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Run all branches in parallel then ensemble the results.

        Returns:
            (final_answer: str, total_cost: float)
        """

        async def branch_programmer_refine() -> str:
            """Branch 1: Programmer → RefineAnswer."""
            try:
                code_result = await self.programmer(problem=problem)
                code_output = code_result.get("output", "")
                if code_output and "Error" not in str(code_output) and "No code" not in str(code_output):
                    refine_input = f"{problem}\nCode output:\n{code_output}"
                    refined = await self.refine_answer(input=refine_input)
                    if refined.get("response"):
                        return refined["response"]
            except KeyError as e:
                from scripts.logs import logger
                logger.error(f"[MIPRO] branch_programmer_refine KeyError: {e}")
            except Exception:
                pass
            return None

        async def branch_detailed() -> str:
            """Branch 2: DetailedSolution."""
            try:
                detailed = await self.detailed_solution(input=problem)
                if detailed.get("response"):
                    return detailed["response"]
            except KeyError as e:
                from scripts.logs import logger
                logger.error(f"[MIPRO] branch_detailed KeyError: {e}")
            except Exception:
                pass
            return None

        async def branch_generate() -> str:
            """Branch 3/4: GenerateSolution."""
            try:
                generated = await self.generate_solution(input=problem)
                if generated.get("response"):
                    return generated["response"]
            except KeyError as e:
                from scripts.logs import logger
                logger.error(f"[MIPRO] branch_generate KeyError: {e}")
            except Exception:
                pass
            return None

        # Run all branches in parallel
        results = await asyncio.gather(
            branch_programmer_refine(),
            branch_detailed(),
            branch_generate(),
            branch_generate(),
            return_exceptions=True
        )

        # Collect valid results
        solutions: List[str] = []
        for r in results:
            if isinstance(r, str) and r:
                solutions.append(r)

        if not solutions:
            return "", self.llm.get_usage_summary()["total_cost"]

        # Single solution: return directly
        if len(solutions) == 1:
            return solutions[0], self.llm.get_usage_summary()["total_cost"]

        # ScEnsemble: vote for the best answer
        try:
            ensemble_result = await self.sc_ensemble(
                solutions=solutions,
                problem=problem,
            )
            final_answer = ensemble_result.get("response", solutions[0])
        except Exception:
            final_answer = solutions[0]

        return final_answer, self.llm.get_usage_summary()["total_cost"]