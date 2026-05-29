import asyncio
from typing import List
import workspace.MBPP.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType





class Workflow:
    """
    MBPP Workflow: CodeGenerate×3 → ScEnsemble → Test → FixCode.
    Generates 3 code solutions in parallel, ensembles, tests, and fixes if needed.
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

        self.code_generate = operator.CodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)
        self.fix_code = operator.FixCode(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Workflow: CodeGenerate×3 (parallel) → ScEnsemble → Test → FixCode.

        Returns:
            (final_code: str, total_cost: float)
        """

        async def generate_code() -> str:
            """Generate one code solution."""
            try:
                result = await self.code_generate(
                    problem=problem,
                    entry_point=entry_point
                )
                code = result.get("code", "")
                if code:
                    return code
            except Exception:
                pass
            return None

        # Step 1: Generate 3 code solutions in parallel
        results = await asyncio.gather(
            generate_code(),
            generate_code(),
            generate_code(),
            return_exceptions=True
        )

        # Collect valid solutions
        candidate_codes: List[str] = []
        for r in results:
            if isinstance(r, str) and r:
                candidate_codes.append(r)

        # No code generated
        if not candidate_codes:
            return "", self.llm.get_usage_summary()["total_cost"]

        # Single solution: skip ensemble
        if len(candidate_codes) == 1:
            best_code = candidate_codes[0]
        else:
            # Step 2: ScEnsemble to pick the best code
            try:
                ensemble_result = await self.sc_ensemble(
                    solutions=candidate_codes,
                    problem=problem
                )
                best_code = ensemble_result.get("response", candidate_codes[0])
            except Exception:
                best_code = candidate_codes[0]

        # Step 3: Test the code
        try:
            test_result = await self.test(
                problem=problem,
                solution=best_code,
                entry_point=entry_point,
                test_loop=2  # fewer loops since FixCode is the fallback
            )

            if test_result.get("result"):
                return test_result.get("solution", best_code), self.llm.get_usage_summary()["total_cost"]
            else:
                # Step 4: FixCode as fallback
                error_msg = str(test_result.get("solution", "Test failed"))
                try:
                    fix_result = await self.fix_code(
                        problem=problem,
                        solution=best_code,
                        error=error_msg,
                        entry_point=entry_point
                    )
                    fixed_code = fix_result.get("code", best_code)
                    return fixed_code, self.llm.get_usage_summary()["total_cost"]
                except Exception:
                    return best_code, self.llm.get_usage_summary()["total_cost"]
        except Exception:
            return best_code, self.llm.get_usage_summary()["total_cost"]