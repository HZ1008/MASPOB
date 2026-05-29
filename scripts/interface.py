import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.logs import logger
from scripts.async_llm import LLMsConfig


def load_best_round(dataset: str, optimized_path: str = "workspace") -> int:
    """Return the round with the best validation score."""
    data_utils = DataUtils(f"{optimized_path}/{dataset}")

    # Use get_top_rounds to find the highest-scoring round
    top_rounds = data_utils.get_top_rounds(sample=2, mode="Graph")
    if not top_rounds[1]:
        return 1

    return top_rounds[1]["round"]


def load_workflow_class(graph_path: str):
    """Dynamically load the Workflow class from a graph.py file."""
    spec = importlib.util.spec_from_file_location("workflow_module", graph_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_module"] = module
    spec.loader.exec_module(module)
    return module.Workflow


async def maspob_inference(
    dataset: DatasetType,
    question: str,
    entry_point: Optional[str] = None,
    round: Optional[int] = None,
    llm_name: str = "gpt-4o-mini",
    optimized_path: str = "workspace",
) -> Tuple[str, float]:
    """Run MASPOB inference on a single question.

    Args:
        dataset: Dataset name (e.g., "GSM8K", "DROP").
        question: Input question string.
        round: Workflow round to use; defaults to the best round if None.
        llm_name: LLM model name.
        optimized_path: Path where optimized workflow rounds are stored.

    Returns:
        Tuple of (answer, cost).
    """
    # Default to the best round if not specified
    if round is None:
        round = load_best_round(dataset, optimized_path)

    logger.info(f"Using round {round} for inference")

    # Build and load the workflow
    graph_path = Path(optimized_path) / dataset / "workflows" / f"round_{round}" / "graph.py"
    if not graph_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {graph_path}")

    WorkflowClass = load_workflow_class(str(graph_path))

    llm_config = LLMsConfig.default().get(llm_name)
    workflow = WorkflowClass(
        name=f"{dataset}_workflow",
        llm_config=llm_config,
        dataset=dataset,
    )

    # Code tasks require an additional entry_point argument
    if dataset in ["MBPP", "HumanEval"]:
        answer, cost = await workflow(question, entry_point=entry_point)
    else:
        answer, cost = await workflow(question)

    return answer, cost


if __name__ == "__main__":
    asyncio.run(
        maspob_inference(
            dataset="MBPP",
            question="write a function named add_two_numbers to calculate the sum of two numbers",
            entry_point="add_two_numbers",
        )
    )
