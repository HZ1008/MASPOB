# Workflow Topology Configuration for MASPOB

from typing import Dict, List


# Topology configs (complete workflows)
WORKFLOW_TOPOLOGIES: Dict[str, List[Dict]] = {
    # DROP: Solve → Format (reading comprehension + numerical calculation)
    "DROP": [
        {"name": "Solve", "prompt_domain": "SOLVE_PROMPT",
         "attr_path": "solve.prompt", "dependencies": []},
        {"name": "Format", "prompt_domain": "FORMAT_PROMPT",
         "attr_path": "format.prompt", "dependencies": ["Solve"]},
    ],
    # HotpotQA: Generate×3 → Ensemble → FormatAnswer (multi-hop QA, F1 score)
    "HotpotQA": [
        {"name": "AnswerGenerate", "prompt_domain": "ANSWER_GENERATION_PROMPT",
         "attr_path": "answer_generate.prompt", "dependencies": []},
        {"name": "ScEnsemble", "prompt_domain": "SC_ENSEMBLE_PROMPT",
         "attr_path": "sc_ensemble.prompt", "dependencies": ["AnswerGenerate"]},
        {"name": "FormatAnswer", "prompt_domain": "FORMAT_ANSWER_PROMPT",
         "attr_path": "format_answer.prompt", "dependencies": ["ScEnsemble"]},
    ],
    # GSM8K: Solve×3 → ScEnsemble → Programmer → Extract
    "GSM8K": [
        {"name": "Solve", "prompt_domain": "GSM8K_SOLVE_PROMPT",
         "attr_path": "solve.prompt", "dependencies": []},
        {"name": "ScEnsemble", "prompt_domain": "SC_ENSEMBLE_PROMPT",
         "attr_path": "sc_ensemble.prompt", "dependencies": ["Solve"]},
        {"name": "Programmer", "prompt_domain": "PYTHON_CODE_VERIFIER_PROMPT",
         "attr_path": "programmer.prompt", "dependencies": ["ScEnsemble"]},
        {"name": "Extract", "prompt_domain": "GSM8K_EXTRACT_PROMPT",
         "attr_path": "extract.prompt", "dependencies": ["Programmer"]},
    ],
    # MATH: Programmer → RefineAnswer + DetailedSolution + GenerateSolution×2 → Ensemble
    "MATH": [
        {"name": "Programmer", "prompt_domain": "PYTHON_CODE_VERIFIER_PROMPT",
         "attr_path": "programmer.prompt", "dependencies": []},
        {"name": "RefineAnswer", "prompt_domain": "REFINE_ANSWER_PROMPT",
         "attr_path": "refine_answer.prompt", "dependencies": ["Programmer"]},
        {"name": "DetailedSolution", "prompt_domain": "DETAILED_SOLUTION_PROMPT",
         "attr_path": "detailed_solution.prompt", "dependencies": []},
        {"name": "GenerateSolution", "prompt_domain": "GENERATE_SOLUTION_PROMPT",
         "attr_path": "generate_solution.prompt", "dependencies": []},
        {"name": "ScEnsemble", "prompt_domain": "SC_ENSEMBLE_PROMPT",
         "attr_path": "sc_ensemble.prompt", "dependencies": ["RefineAnswer", "DetailedSolution", "GenerateSolution"]},
    ],
    # HumanEval: CustomCodeGenerate×3 → ScEnsemble → Test
    "HumanEval": [
        {"name": "CustomCodeGenerate", "prompt_domain": "CUSTOM_CODE_GENERATE_PROMPT",
         "attr_path": "custom_code_generate.prompt", "dependencies": []},
        {"name": "ScEnsemble", "prompt_domain": "SC_ENSEMBLE_PROMPT",
         "attr_path": "sc_ensemble.prompt", "dependencies": ["CustomCodeGenerate"]},
        {"name": "Test", "prompt_domain": "REFLECTION_ON_PUBLIC_TEST_PROMPT",
         "attr_path": "test.prompt", "dependencies": ["ScEnsemble"]},
    ],
    # MBPP: CodeGenerate×3 → Ensemble → Test → FixCode
    "MBPP": [
        {"name": "CodeGenerate", "prompt_domain": "CODE_GENERATE_PROMPT",
         "attr_path": "code_generate.prompt", "dependencies": []},
        {"name": "ScEnsemble", "prompt_domain": "SC_ENSEMBLE_PROMPT",
         "attr_path": "sc_ensemble.prompt", "dependencies": ["CodeGenerate"]},
        {"name": "Test", "prompt_domain": "REFLECTION_ON_PUBLIC_TEST_PROMPT",
         "attr_path": "test.prompt", "dependencies": ["ScEnsemble"]},
        {"name": "FixCode", "prompt_domain": "FIX_CODE_PROMPT",
         "attr_path": "fix_code.prompt", "dependencies": ["Test"]},
    ],
}

def get_workflow_topologies() -> Dict[str, List[Dict]]:
    """Get workflow topologies.

    Returns:
        Dictionary mapping dataset names to workflow topology configurations
    """
    return WORKFLOW_TOPOLOGIES


def get_workflow_topology(dataset: str) -> List[Dict]:
    """Get workflow topology for the given dataset.

    Args:
        dataset: Dataset name

    Returns:
        Workflow topology configuration for the specified dataset
    """
    topologies = get_workflow_topologies()
    if dataset in topologies:
        return topologies[dataset]
    # Default to DROP topology
    print(f"[Warning] No topology defined for {dataset}, using DROP topology")
    return topologies["DROP"]


def load_workflow_class(dataset: str):
    """Dynamically load the Workflow class for the given dataset.

    Args:
        dataset: Dataset name

    Returns:
        Workflow class for the specified dataset
    """
    import importlib
    module_path = f"workspace.{dataset}.workflows.round_1.graph"
    try:
        module = importlib.import_module(module_path)
        return module.Workflow
    except ImportError as e:
        raise ImportError(f"Cannot load Workflow for dataset '{dataset}': {e}")

