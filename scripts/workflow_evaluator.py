"""Evaluate an existing workflow round without performing any optimization.

This module provides a small helper class `WorkflowEvaluator` which reuses the
same evaluation utilities as `Optimizer`, but **never** calls the optimization
LLM and **never** writes new workflow code. It only:

1. Loads an existing `graph.py` from `workspace/<dataset>/workflows/round_k`.
2. Runs the standard evaluation loop on the validation set.
3. Returns the average score.

You can safely use this when you want to test a fixed workflow without letting
MASPOB modify it.
"""

import asyncio
from typing import List, Literal

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.prompts.prompt import *


QuestionType = Literal["math", "code", "qa"]


async def generate_prompts_from_initial(
    llm: None,
    initial_prompt: str,
    num_prompts: int = 20,
    task_description: str = "Your task: {input}",
) -> List[str]:
    """Generate new prompts from an initial prompt using the LLM.

    This function uses the meta-prompt ``GENERATE_PROMPT`` (defined in
    ``op_prompt.py``) to ask the LLM to rewrite / optimize the given
    ``initial_prompt``. It repeats this process ``num_prompts`` times
    and returns a list of the generated prompts.

    Args:
        llm: The AsyncLLM instance to call.
        initial_prompt: The base prompt to be improved (e.g. ANSWER_GENERATION_PROMPT).
        num_prompts: How many new prompts to generate.

    Returns:
        A list of newly generated prompt strings.
    """

    prompts: List[str] = []
    for _ in range(num_prompts):
        meta_prompt = GENERATE_PROMPT.format(initial_prompt=initial_prompt)
        new_prompt = await llm(meta_prompt)
        full_prompt = new_prompt + "\n" + task_description
        prompts.append(full_prompt)
    return prompts


class WorkflowEvaluator:
    """Evaluate existing workflow rounds without optimizing them.

    This class mirrors the parts of :class:`Optimizer` that are needed for
    evaluation, but deliberately excludes all optimization logic. It assumes
    that the workflow code already exists under::

        workspace/<dataset>/workflows/round_<round_number>
    """

    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        exec_llm_config,
        optimized_path: str = "workspace",
        validation_rounds: int = 1,
    ) -> None:
        # Basic configuration
        self.dataset = dataset
        self.type = question_type
        self.execute_llm_config = exec_llm_config

        # Paths and helpers (reuse the same utilities as Optimizer)
        self.root_path = f"{optimized_path}/{self.dataset}"
        self.validation_rounds = validation_rounds

        self.graph = None
        # For pure evaluation we do not want to automatically create new
        # round_k directories (like empty round_3) when just running tests.
        # Therefore we disable auto-creation here.
        self.graph_utils = GraphUtils(self.root_path, auto_create_round_dirs=False)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)

    def evaluate(self, round_number: int, initial: bool = False) -> float:
        """Synchronously evaluate a given workflow round.

        Args:
            round_number: Which workflow round to evaluate (e.g., 1 for
                ``round_1``).
            initial: Whether to tag this evaluation as an "initial" run in the
                logs/results (mirrors the ``initial`` flag in
                :meth:`EvaluationUtils.evaluate_graph`).

        Returns:
            Average score over the validation runs for the specified round.
        """

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._evaluate_async(round_number, initial=initial)
            )
        finally:
            loop.close()

    async def _evaluate_async(self, round_number: int, initial: bool = False) -> float:
        """Async core for :meth:`evaluate`.

        This method only loads and evaluates an existing workflow. It does **not**
        call any optimization LLMs or write new workflow files.
        """

        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"

        # Load historical results (for appending new evaluation records)
        data = self.data_utils.load_results(graph_path)

        # Ensure the directory for this round exists (no-op if it already does)
        directory = self.graph_utils.create_round_directory(graph_path, round_number)

        # Load the existing workflow class for this round
        self.graph = self.graph_utils.load_graph(round_number, graph_path)

        # Run the standard evaluation loop; this will update results.json
        avg_score = await self.evaluation_utils.evaluate_graph(
            self, directory, validation_n, data, initial=initial
        )

        return avg_score
