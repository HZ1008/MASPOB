# Evaluation for different datasets

from typing import Dict, Literal, Tuple

from benchmarks.benchmark import BaseBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark
from benchmarks.livecodebench import LiveCodeBench

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "LiveCodeBench"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
            "LiveCodeBench": LiveCodeBench,
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f" Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # Use params to configure the graph and benchmark
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None  # For test data, generally use None to test all
        else:
            va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(configured_graph, va_list)

    async def evaluate_workflow_instance(
        self, dataset: DatasetType, workflow, path: str, is_test: bool = False
    ) -> Tuple[float, float, float]:
        """Evaluate an already-instantiated Workflow on the given dataset.

        Args:
            dataset: Dataset name, e.g. "DROP", "GSM8K".
            workflow: Instantiated workflow object (callable: await workflow(input)).
            path: Log and result save path (typically the round's workflows directory).
            is_test: If True, evaluate on test set; otherwise validation set.

        Returns:
            (score, avg_cost, total_cost)
        """
        if dataset not in self.dataset_configs:
            raise ValueError(f" Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # va_list=None means evaluate on the full dataset (consistent with graph_evaluate)
        va_list = None
        return await benchmark.run_evaluation(workflow, va_list)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"data/datasets/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
