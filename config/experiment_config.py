# Experiment Configuration for MASPOB Workflows

from typing import Dict, List


class ExperimentConfig:
    """Configuration for a specific dataset experiment."""
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


def get_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Get experiment configurations.

    Returns:
        Dictionary mapping dataset names to ExperimentConfig objects
    """
    return {
        "DROP": ExperimentConfig("DROP", "qa", ["Solve", "Format"]),
        "HotpotQA": ExperimentConfig("HotpotQA", "qa", ["AnswerGenerate", "ScEnsemble", "FormatAnswer"]),
        "MATH": ExperimentConfig("MATH", "math", ["Programmer", "RefineAnswer", "DetailedSolution", "GenerateSolution", "ScEnsemble"]),
        "GSM8K": ExperimentConfig("GSM8K", "math", ["Solve", "ScEnsemble", "Programmer", "Extract"]),
        "MBPP": ExperimentConfig("MBPP", "code", ["CodeGenerate", "ScEnsemble", "Test", "FixCode"]),
        "HumanEval": ExperimentConfig("HumanEval", "code", ["CustomCodeGenerate", "ScEnsemble", "Test"]),
        "LiveCodeBench": ExperimentConfig("LiveCodeBench", "code", ["CustomCodeGenerate", "ValidateCode", "Test"]),
    }


# Default configs (used for parse_args choices)
EXPERIMENT_CONFIGS = get_experiment_configs()


# Default sample counts per dataset (validate/test split sizes)
DATASET_SAMPLE_CONFIGS: Dict[str, Dict[str, int]] = {
    "DROP":      {"validate": 100, "test": 800},   # total: 200 / 800
    "HotpotQA":  {"validate": 100, "test": 800},   # total: 200 / 800
    "GSM8K":     {"validate": 150, "test": 1055},  # total: 264 / 1055
    "MATH":      {"validate": 100, "test": 486},   # total: 119 / 486
    "MBPP":      {"validate": 86,  "test": 341},   # total: 86  / 341
    "HumanEval": {"validate": 33,  "test": 131},   # total: 33  / 131
    "LiveCodeBench": {"validate": 50, "test": 200},  # estimated
}

