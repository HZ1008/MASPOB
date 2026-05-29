# GNN-based Prompt Optimization for MASPOB Workflows

# =============================================================================
# IMPORTS
# =============================================================================

# Filter noisy third-party warnings
import warnings
warnings.filterwarnings("ignore", message="antlr4.error.ErrorListener module is not installed")
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")

# Standard library
import argparse
import asyncio
import copy
import gc
import json
import os
import random
import ctypes
import sys
import time
from typing import Dict, List

# PyTorch
import torch
import torch.optim as optim

# Transformers (for sentence embeddings)
from transformers import AutoTokenizer, AutoModel

# Project modules
from data.download_data import download
from scripts.async_llm import LLMsConfig, create_llm_instance

# Configuration modules
from config import (
    ExperimentConfig,
    get_experiment_configs,
    EXPERIMENT_CONFIGS,
    DATASET_SAMPLE_CONFIGS,
    get_workflow_topologies,
    get_workflow_topology,
    load_workflow_class,
    PROMPT_CONFIGS,
    PROMPT_TYPES,
    PROMPT_NAMES,
)

from scripts.prompts.generator import build_prompt_domains
from scripts.utils.experiment import (
    print_progress,
    save_experiment_results,
)
from scripts.utils.training import train_with_early_stopping
from scripts.embeddings import (
    EMBEDDING_MODE,
    SENTENCE_MODEL_NAME,
    OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    get_sen_embedding,
)
from scripts.gnn_model import (
    WorkflowGAT,
    compute_gradient_feature,
    initialize_fisher,
    update_fisher,
    compute_prediction_and_uncertainty,
    build_combined_embedding,
    select_best_prompt_for_operator,
    inject_prompt_to_workflow,
)


# =============================================================================
# RESOURCE CLEANUP
# =============================================================================
def cleanup_resources():
    """Release memory and process resources after each evaluation round."""
    # 1. Python garbage collection
    gc.collect()

    # 2. PyTorch GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 3. asyncio cleanup
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            pending = asyncio.all_tasks(loop)
            for task in pending:
                if not task.done() and not task.cancelled():
                    pass
    except RuntimeError:
        pass

    # 4. Windows: trim working set
    if sys.platform == 'win32':
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
            )
        except Exception:
            pass

# =============================================================================
# CONFIGURATION
# =============================================================================

# PyTorch device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
TKWARGS = {"device": DEVICE, "dtype": DTYPE}

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DOMAIN_DIR = os.path.join(BASE_DIR, "prompt_domain")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GNN-based Prompt Optimization for MASPOB")

    # Dataset and experiment settings
    parser.add_argument("--dataset", type=str, choices=list(EXPERIMENT_CONFIGS.keys()),
                        default="HotpotQA", help="Dataset to use for optimization")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of samples to evaluate per round (default: use dataset-specific default from DATASET_SAMPLE_CONFIGS)")
    parser.add_argument("--max_rounds", type=int, default=45,
                        help="Maximum number of optimization rounds")
    parser.add_argument("--num_prompts", type=int, default=20,
                        help="Number of prompt variants to generate per operator")
    parser.add_argument("--concurrent_batch", type=int, default=50,
                        help="Number of concurrent API calls per batch (lower to avoid rate limits)")

    # Pretraining settings
    parser.add_argument("--pretrain_rounds", type=int, default=5,
                        help="Number of pretraining rounds with random prompt selection")
    parser.add_argument("--pretrain_epochs", type=int, default=800,
                        help="Number of epochs to train GNN on pretrain data")

    # Model settings
    parser.add_argument("--opt_model_name", type=str, default="gpt-4o-mini",
                        help="Model for prompt generation/optimization")
    parser.add_argument("--exec_model_name", type=str, default="gpt-4o-mini-exec",
                        help="Model for workflow execution")

    # GNN + Uncertainty settings
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension for GNN")
    parser.add_argument("--num_gnn_layers", type=int, default=1,
                        help="Number of GAT layers")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--ucb_alpha", type=float, default=0.2,
                        help="Exploration coefficient for UCB (higher = more exploration)")
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="Regularization coefficient for diagonal Fisher matrix")
    parser.add_argument("--fisher_coef", type=float, default=10,
                        help="Coefficient for Fisher matrix update: A = A + coef * x * x^T (higher = faster uncertainty decay)")
    parser.add_argument("--ucb_type", type=str, default="linear", choices=["neural", "linear", "greedy"],
                        help="Type of UCB: 'neural' (GNN gradients), 'linear' (prompt embeddings), 'greedy' (no exploration)")

    # Early stopping settings
    parser.add_argument("--patience", type=int, default=200,
                        help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-10,
                        help="Minimum improvement to reset patience")

    # Search strategy settings
    parser.add_argument("--search_strategy", type=str, default="coordinate",
                        choices=["coordinate", "exhaustive"],
                        help="Search strategy: 'coordinate' or 'exhaustive'")

    # Path settings
    parser.add_argument("--prompt_domain_dir", type=str, default=None,
                        help="Directory for prompt domain files (default: prompt_domain)")

    # Other settings
    parser.add_argument("--if_force_download", type=lambda x: x.lower() == "true", default=False,
                        help="Force dataset download")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Graph structure settings
    parser.add_argument("--bidirectional", type=lambda x: x.lower() == "true", default=True,
                        help="Use bidirectional edges in GNN (default: True, better for prediction tasks)")

    # Test phase settings
    parser.add_argument("--run_test", type=lambda x: x.lower() != "false", default=True,
                        help="Run test phase after training (default: True, use --run_test false to disable)")
    parser.add_argument("--test_samples", type=int, default=None,
                        help="Number of test samples to evaluate (default: use dataset-specific default from DATASET_SAMPLE_CONFIGS)")
    parser.add_argument("--test_repeats", type=int, default=3,
                        help="Number of times to repeat each test evaluation (default: 3)")

    return parser.parse_args()


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

# Default concurrency; overridden by --concurrent_batch argument
CONCURRENT_BATCH_SIZE = 10


async def evaluate_workflow(flow, dataset: str, data_path: str = None,
                           max_examples: int = None, concurrent_batch: int = None,
                           log_path: str = None):
    """
    Evaluate a workflow on a given dataset.

    Args:
        flow: The workflow to evaluate
        dataset: Dataset name (DROP, GSM8K, MATH, HotpotQA, etc.)
        data_path: Path to data file (default: auto-generated from dataset name)
        max_examples: Maximum number of examples to evaluate
        concurrent_batch: Number of concurrent tasks (default: CONCURRENT_BATCH_SIZE)
        log_path: Directory for evaluation logs (optional)

    Returns:
        (avg_score, avg_cost, cumulative_cost)
    """
    if concurrent_batch is None:
        concurrent_batch = CONCURRENT_BATCH_SIZE

    if data_path is None:
        data_path = os.path.join("data", "datasets", f"{dataset.lower()}_validate.jsonl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if log_path is None:
        log_path = os.path.join("result_CSV", "MASPO", dataset)

    os.makedirs(log_path, exist_ok=True)

    # Select benchmark class for the dataset
    from benchmarks.drop import DROPBenchmark
    from benchmarks.gsm8k import GSM8KBenchmark
    from benchmarks.hotpotqa import HotpotQABenchmark
    from benchmarks.math import MATHBenchmark
    from benchmarks.humaneval import HumanEvalBenchmark
    from benchmarks.mbpp import MBPPBenchmark

    benchmark_classes = {
        "DROP": DROPBenchmark,
        "GSM8K": GSM8KBenchmark,
        "MATH": MATHBenchmark,
        "HotpotQA": HotpotQABenchmark,
        "HumanEval": HumanEvalBenchmark,
        "MBPP": MBPPBenchmark,
    }

    if dataset not in benchmark_classes:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported: {list(benchmark_classes.keys())}")

    benchmark_class = benchmark_classes[dataset]
    benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=log_path)

    va_list = list(range(max_examples)) if max_examples else None

    try:
        avg_score, avg_cost_per_sample, _ = await benchmark.run_evaluation(
            agent=flow,
            va_list=va_list,
            max_concurrent_tasks=concurrent_batch,
        )
    except Exception as e:
        print(f"    [ERROR] Benchmark evaluation failed: {type(e).__name__}: {str(e)[:100]}")
        avg_score, avg_cost_per_sample = 0.0, 0.0

    cumulative_cost = flow.llm.get_usage_summary()["total_cost"]

    return avg_score, avg_cost_per_sample, cumulative_cost


async def run_pretrain_phase(
    gnn_model: WorkflowGAT,
    frozen_model: WorkflowGAT,
    optimizer: torch.optim.Optimizer,
    all_operator_embeddings: List[torch.Tensor],
    workflow_agent_prompt: List[List[str]],
    flow,
    topology: List[Dict],
    eval_func,
    eval_data_path: str,
    pretrain_rounds: int = 10,
    pretrain_epochs: int = 500,
    max_eval_examples: int = 10,
    lambda_reg: float = 1.0,
    ucb_type: str = "neural",
    patience: int = 50,
    min_delta: float = 1e-6,
    fisher_coef: float = 10,
):
    """
    Pretrain phase: random prompt selection to collect data, then train GNN.
    Accumulates the diagonal Fisher information matrix for Neural/Linear UCB.
    """
    num_operators = len(all_operator_embeddings)
    num_prompts_per_op = [emb.shape[0] for emb in all_operator_embeddings]

    print(f"\n{'='*60}")
    print(f"PRETRAIN PHASE: {pretrain_rounds} rounds with random prompts")
    print(f"  [UCB Type] {ucb_type.upper()}")
    print(f"{'='*60}\n")

    # Initialize Fisher information matrix
    embedding_dim = sum(emb.shape[1] for emb in all_operator_embeddings)
    device = all_operator_embeddings[0].device
    fisher_matrix = initialize_fisher(
        ucb_type, frozen_model=frozen_model, embedding_dim=embedding_dim,
        device=device, lambda_reg=lambda_reg
    )

    if ucb_type == "neural":
        feature_dim = fisher_matrix.shape[0]
        print(f"[Pretrain] Neural UCB: Initialized diagonal Fisher with λ = {lambda_reg}")
        print(f"[Pretrain] Neural UCB: Gradient feature dimension: {feature_dim}")
    elif ucb_type == "linear":
        feature_dim = fisher_matrix.shape[0]
        print(f"[Pretrain] Linear UCB: Initialized full Fisher matrix with λ = {lambda_reg}")
        print(f"[Pretrain] Linear UCB: Embedding dimension: {feature_dim}")
        print(f"[Pretrain] Linear UCB: Fisher matrix size: [{feature_dim} x {feature_dim}]")
    else:  # greedy
        feature_dim = embedding_dim
        print(f"[Pretrain] Greedy mode: No Fisher matrix needed")

    # Collect data with random prompt selections
    pretrain_data = []
    pretrain_history = {"rounds": [], "prompts": [], "scores": []}

    for round_idx in range(pretrain_rounds):
        # Random prompt selection for each operator
        random_indices = [random.randint(0, num_prompts_per_op[i] - 1) for i in range(num_operators)]
        print_progress(round_idx, pretrain_rounds, prefix="[Pretrain] ", suffix=f"prompts={random_indices}")

        # Build combined embedding
        combined = build_combined_embedding(all_operator_embeddings, random_indices)

        # Evaluate
        inject_prompt_to_workflow(flow, topology, random_indices, workflow_agent_prompt)
        actual_score, _, total_cost = await eval_func(flow, data_path=eval_data_path, max_examples=max_eval_examples)

        status = "[ZERO]" if actual_score == 0 else ""
        print(f"[PT{round_idx+1:>2}] prompts={random_indices} score={actual_score:.4f} cost=${total_cost:.4f} {status}")

        # Update Fisher matrix
        if ucb_type == "neural":
            feature_vector = compute_gradient_feature(frozen_model, combined)
        else:  # linear or greedy
            feature_vector = combined
        fisher_matrix = update_fisher(ucb_type, fisher_matrix, feature_vector, fisher_coef)

        # Store data (including prompt indices for one-hot encoding)
        pretrain_data.append((combined.clone(), actual_score, random_indices.copy()))
        pretrain_history["rounds"].append(round_idx + 1)
        pretrain_history["prompts"].append(random_indices.copy())
        pretrain_history["scores"].append(actual_score)

        cleanup_resources()

    # Train GNN on collected data
    print(f"\n{'='*60}")
    print(f"Training GNN on {len(pretrain_data)} samples...")
    print(f"{'='*60}\n")

    batch_embs = torch.stack([emb for emb, _, _ in pretrain_data])
    batch_targets = torch.tensor([score for _, score, _ in pretrain_data], dtype=torch.float32).to(batch_embs.device)

    # Scale targets to [0, 1] to amplify score differences and reduce gradient vanishing
    scaled_targets = gnn_model.scale_score(batch_targets)

    train_with_early_stopping(gnn_model, optimizer, batch_embs, scaled_targets,
                              max_epochs=pretrain_epochs, patience=patience, min_delta=min_delta, verbose=True)

    gnn_model.eval()
    with torch.no_grad():
        scaled_preds = gnn_model(batch_embs)
        preds = gnn_model.unscale_score(scaled_preds)

    pretrain_history["predicted_scores"] = [preds[i].item() for i in range(len(pretrain_data))]
    pretrain_history["errors"] = [abs(preds[i].item() - actual) for i, (_, actual, _) in enumerate(pretrain_data)]

    # Find the best prompt combination from pretrain phase
    best_pretrain_idx = pretrain_history["scores"].index(max(pretrain_history["scores"]))
    best_pretrain_prompts = pretrain_history["prompts"][best_pretrain_idx]
    best_pretrain_score = pretrain_history["scores"][best_pretrain_idx]

    print(f"\n[Pretrain] Trained GNN on {len(pretrain_data)} samples.")
    print(f"[Pretrain] Best prompts from pretrain: {best_pretrain_prompts} (score: {best_pretrain_score:.4f})")
    if ucb_type == "neural":
        print(f"[Pretrain] Diagonal Fisher (NEURAL) accumulated from {len(pretrain_data)} samples.")
    elif ucb_type == "linear":
        print(f"[Pretrain] Full Fisher matrix (LINEAR) accumulated from {len(pretrain_data)} samples.")
    else:  # greedy
        print(f"[Pretrain] Greedy mode: No Fisher matrix.")

    return pretrain_history, best_pretrain_prompts, pretrain_data, fisher_matrix


async def run_gnn_optimization(
    gnn_model: WorkflowGAT,
    frozen_model: WorkflowGAT,
    optimizer: torch.optim.Optimizer,
    all_operator_embeddings: List[torch.Tensor],
    workflow_agent_prompt: List[List[str]],
    flow,
    topology: List[Dict],
    eval_func,
    eval_data_path: str,
    num_rounds: int = 10,
    max_eval_examples: int = 10,
    pretrain_best_prompts: List[int] = None,
    pretrain_data: List[tuple] = None,
    pretrain_fisher_matrix: torch.Tensor = None,
    train_epochs: int = 50,
    ucb_alpha: float = 0.1,
    lambda_reg: float = 1.0,
    ucb_type: str = "neural",
    patience: int = 50,
    min_delta: float = 1e-6,
    search_strategy: str = "coordinate",
    initial_lr: float = 5e-3,  # reset to this LR each round
    fisher_coef: float = 10,
):
    """
    Prompt optimization with UCB (Neural, Linear, or Greedy).

    Search strategies:
    - 'coordinate': coordinate descent, optimizing one operator per step
    - 'exhaustive': exhaustive search over all prompt combinations

    UCB types:
    - Neural UCB: gradient features from frozen initial model, diagonal Fisher
    - Linear UCB: prompt embeddings as features, full Fisher matrix
    - Greedy: pure exploitation, no exploration term
    """
    num_operators = len(all_operator_embeddings)
    if len(topology) != num_operators:
        raise ValueError(f"Topology mismatch: {len(topology)} vs {num_operators}")

    ucb_type_upper = ucb_type.upper()

    # Initialize with pretrain best prompts
    if pretrain_best_prompts is not None:
        current_prompt_indices = list(pretrain_best_prompts)
        print(f"[{ucb_type_upper}] Initialized with pretrain best: {current_prompt_indices}")
    else:
        current_prompt_indices = [0] * num_operators
        print(f"[{ucb_type_upper}] Initialized with zeros: {current_prompt_indices}")

    # Inherit historical data from pretrain phase
    training_data = list(pretrain_data) if pretrain_data else []
    print(f"[{ucb_type_upper}] Starting with {len(training_data)} samples from pretrain")
    print(f"[{ucb_type_upper}] Exploration coefficient alpha = {ucb_alpha}")

    num_prompts_per_op = [len(emb) for emb in all_operator_embeddings]
    total_combinations = 1
    for n in num_prompts_per_op:
        total_combinations *= n
    strategy_name = "Exhaustive" if search_strategy == "exhaustive" else "Coordinate Descent"
    print(f"[{ucb_type_upper}] Search strategy: {strategy_name}")
    if search_strategy == "exhaustive":
        print(f"[{ucb_type_upper}] Total combinations to search per round: {total_combinations:,}")

    # Initialize or inherit Fisher information matrix
    embedding_dim = sum(emb.shape[1] for emb in all_operator_embeddings)
    device = all_operator_embeddings[0].device

    if pretrain_fisher_matrix is not None:
        fisher_matrix = pretrain_fisher_matrix.clone()
        print(f"[{ucb_type_upper}] Fisher matrix initialized from pretrain (samples: {len(training_data)})")
    else:
        fisher_matrix = initialize_fisher(
            ucb_type, frozen_model=frozen_model, embedding_dim=embedding_dim,
            device=device, lambda_reg=lambda_reg
        )
        print(f"[{ucb_type_upper}] Fisher matrix initialized with λ = {lambda_reg}")

    if ucb_type == "neural":
        feature_dim = fisher_matrix.shape[0]
        print(f"[NEURAL UCB] Gradient feature dimension: {feature_dim}")
    elif ucb_type == "linear":
        feature_dim = fisher_matrix.shape[0]
        print(f"[LINEAR UCB] Embedding feature dimension: {feature_dim}")
        print(f"[LINEAR UCB] Fisher matrix size: [{feature_dim} x {feature_dim}]")
    else:  # greedy
        feature_dim = embedding_dim
        print(f"[GREEDY] No exploration, pure exploitation")

    history = {
        "rounds": [], "predicted_scores": [], "uncertainties": [],
        "ucb_scores": [], "actual_scores": [], "selected_prompts": [],
        "losses": [], "topology": [t["name"] for t in topology],
        "ucb_type": ucb_type,
        "selection_times": [],
        "total_selection_time": 0.0,
        "search_strategy": search_strategy,
    }

    print(f"\n{'='*80}")
    print(f"{ucb_type_upper} OPTIMIZATION: {num_rounds} rounds, {num_operators} operators")
    print(f"  Training: {train_epochs} epochs per round on ALL historical data")
    print(f"{'='*80}\n")

    round_results = []
    total_selection_time = 0.0

    for round_idx in range(num_rounds):
        print_progress(round_idx, num_rounds, prefix="[Optimize] ")

        selection_start_time = time.time()

        if search_strategy == "exhaustive":
            import itertools
            num_prompts_per_op = [len(emb) for emb in all_operator_embeddings]
            best_ucb = -float('inf')
            best_combo = None

            all_combos = list(itertools.product(*[range(n) for n in num_prompts_per_op]))
            for combo in all_combos:
                combo_list = list(combo)
                combined_emb = build_combined_embedding(all_operator_embeddings, combo_list)
                pred, unc, _ = compute_prediction_and_uncertainty(
                    ucb_type, gnn_model, combined_emb, frozen_model=frozen_model, fisher_matrix=fisher_matrix
                )
                ucb = pred + ucb_alpha * unc
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_combo = combo_list

            current_prompt_indices = best_combo
        else:
            # Coordinate Descent: optimize one operator at a time
            for op_idx in range(num_operators):
                best_idx, _, _, _ = select_best_prompt_for_operator(
                    ucb_type, gnn_model, op_idx, current_prompt_indices,
                    all_operator_embeddings, frozen_model=frozen_model,
                    fisher_matrix=fisher_matrix, alpha=ucb_alpha
                )
                current_prompt_indices[op_idx] = best_idx

        # Build combined embedding for final evaluation
        combined = build_combined_embedding(all_operator_embeddings, current_prompt_indices)

        pred_score, final_uncertainty, feature_vector = compute_prediction_and_uncertainty(
            ucb_type, gnn_model, combined, frozen_model=frozen_model, fisher_matrix=fisher_matrix
        )
        ucb_score = pred_score + ucb_alpha * final_uncertainty

        selection_end_time = time.time()
        selection_time = selection_end_time - selection_start_time
        total_selection_time += selection_time

        # Evaluate
        inject_prompt_to_workflow(flow, topology, current_prompt_indices, workflow_agent_prompt)
        actual_score, _, total_cost = await eval_func(flow, data_path=eval_data_path, max_examples=max_eval_examples)

        training_data.append((combined.clone(), actual_score, current_prompt_indices.copy()))

        # Update Fisher information matrix
        if ucb_type == "neural":
            fisher_matrix = update_fisher(ucb_type, fisher_matrix, feature_vector, fisher_coef)
        elif ucb_type == "linear":
            fisher_matrix = update_fisher(ucb_type, fisher_matrix, combined, fisher_coef)

        # Retrain GNN from scratch each round to avoid accumulated bias
        gnn_model.reset_parameters()

        # Reset optimizer state (model weights were reset)
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
            for param in param_group['params']:
                if param in optimizer.state:
                    del optimizer.state[param]

        batch_embs = torch.stack([emb for emb, _, _ in training_data])
        batch_targets = torch.tensor([score for _, score, _ in training_data], dtype=torch.float32).to(batch_embs.device)
        scaled_targets = gnn_model.scale_score(batch_targets)

        print(f"\n  [Training] {len(training_data)} samples, max_epochs={train_epochs}, patience={patience}")
        final_loss = train_with_early_stopping(gnn_model, optimizer, batch_embs, scaled_targets,
                                               max_epochs=train_epochs, patience=patience, min_delta=min_delta,
                                               verbose=True)

        status = "[ZERO]" if actual_score == 0 else ""
        print(f"\r[R{round_idx+1:>2}] prompts={current_prompt_indices} score={actual_score:.3f} cost=${total_cost:.4f} sel_time={selection_time:.3f}s {status}")

        # Record
        history["rounds"].append(round_idx + 1)
        history["predicted_scores"].append(pred_score)
        history["uncertainties"].append(final_uncertainty)
        history["ucb_scores"].append(ucb_score)
        history["actual_scores"].append(actual_score)
        history["selected_prompts"].append(current_prompt_indices.copy())
        history["losses"].append(final_loss)
        history["selection_times"].append(selection_time)

        round_results.append({
            "round": round_idx + 1,
            "prompts": current_prompt_indices.copy(),
            "pred": pred_score,
            "uncertainty": final_uncertainty,
            "ucb": ucb_score,
            "actual": actual_score,
            "error": abs(pred_score - actual_score),
            "loss": final_loss,
            "selection_time": selection_time,
        })

        cleanup_resources()

    print()

    history["total_selection_time"] = total_selection_time

    if history['actual_scores']:
        best_score = max(history['actual_scores'])
        best_round = history['actual_scores'].index(best_score) + 1
        avg_selection_time = sum(r['selection_time'] for r in round_results) / len(round_results) if round_results else 0.0

        print(f"\n[Summary]")
        print(f"  Best Score: {best_score:.4f} (Round {best_round})")
        print(f"  Best Prompts: {history['selected_prompts'][best_round - 1]}")
        print(f"  Average Selection Time: {avg_selection_time:.4f}s")
        print(f"  Total Selection Time: {total_selection_time:.4f}s ({search_strategy})")
    else:
        print(f"\n[Summary] No optimization rounds were run (max_rounds=0)")

    print(f"{'='*115}\n")

    return history


# =============================================================================
# TEST PHASE
# =============================================================================

async def run_test_phase(
    gnn_model: WorkflowGAT,
    all_operator_embeddings: List[torch.Tensor],
    workflow_agent_prompt: List[List[str]],
    flow,
    topology: List[Dict],
    eval_func,
    test_data_path: str,
    best_prompts: List[int],
    test_samples: int = 100,
    ucb_type: str = "linear",
    best_val_score: float = 0.0,
    test_repeats: int = 3,
):
    """
    Evaluate the best prompts from validation on the test set.

    Repeats evaluation test_repeats times and averages the results.

    Args:
        gnn_model: trained GNN model
        all_operator_embeddings: prompt embeddings for each operator
        workflow_agent_prompt: prompt text lists for each operator
        flow: workflow instance
        topology: workflow topology
        eval_func: evaluation function
        test_data_path: path to test dataset
        best_prompts: best prompt indices from validation
        test_samples: number of test samples
        ucb_type: UCB type used
        best_val_score: validation score of best_prompts (for display)
        test_repeats: number of evaluation repetitions

    Returns:
        test_results: dict with test scores and metadata
    """
    print(f"\n{'='*100}")
    print(f"TEST PHASE: Evaluating best prompts on test set ({test_repeats} repeats)")
    print(f"{'='*100}")

    num_test_runs = test_repeats

    print(f"[TEST] Test samples: {test_samples}")
    print(f"[TEST] Test repeats: {test_repeats}")
    print(f"[TEST] Test data path: {test_data_path}")

    print(f"\n[TEST] ========== BEST PROMPTS EVALUATION ==========")
    print(f"\n[TEST] ----- Best Prompts: {best_prompts} (Val Score: {best_val_score:.4f}) -----")
    print(f"[TEST] Running {num_test_runs} times and averaging results...")

    inject_prompt_to_workflow(flow, topology, best_prompts, workflow_agent_prompt)

    gnn_model.eval()
    prompt_embedding = build_combined_embedding(all_operator_embeddings, best_prompts)
    with torch.no_grad():
        pred_score = gnn_model(prompt_embedding).item()

    optimized_scores = []
    optimized_costs = []
    cumulative_cost = 0.0
    for run_idx in range(num_test_runs):
        print(f"  Run {run_idx + 1}/{num_test_runs}...", end=" ")
        score, cost, cumulative_cost = await eval_func(
            flow=flow,
            data_path=test_data_path,
            max_examples=test_samples,
        )
        optimized_scores.append(score)
        optimized_costs.append(cost)
        print(f"score={score:.4f}, cost=${cumulative_cost:.4f} (cumulative)")

    final_optimized_score = sum(optimized_scores) / len(optimized_scores)
    final_optimized_std = (sum((s - final_optimized_score) ** 2 for s in optimized_scores) / len(optimized_scores)) ** 0.5
    optimized_cost = sum(optimized_costs) / len(optimized_costs)

    print(f"  Average: {final_optimized_score:.4f} ± {final_optimized_std:.4f}")

    print(f"\n{'='*100}")
    print(f"TEST RESULTS: BEST PROMPTS EVALUATION (tested {num_test_runs} times)")
    print(f"{'='*100}")

    print(f"\n  {'-'*100}")
    print(f"  {'FINAL RESULTS':<40} {'Optimized':<40}")
    print(f"  {'-'*100}")
    print(f"  {'Prompts':<40} {str(best_prompts):<40}")
    print(f"  {'Test Score':<40} {f'{final_optimized_score:.4f} ± {final_optimized_std:.4f}':<40}")
    print(f"  {'-'*100}")
    print(f"{'='*100}\n")

    test_results = {
        "best_prompts": best_prompts,
        "best_val_score": best_val_score,
        "test_repeats": num_test_runs,
        "test_scores": optimized_scores,
        "gnn_predicted_score": pred_score,
        "optimized_score": final_optimized_score,
        "optimized_std": final_optimized_std,
        "optimized_cost_per_sample": optimized_cost,
        "test_samples": test_samples,
        "cumulative_cost": cumulative_cost,
        "ucb_type": ucb_type,
    }

    return test_results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GNN-based Prompt Optimization for MASPOB")
    print("=" * 60)

    # Print device info
    print(f"\n[Device] PyTorch version: {torch.__version__}")
    print(f"[Device] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Device] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[Device] Using: cuda:0")
    else:
        print(f"[Device] Using: CPU")

    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"\n[Config] Random seed: {args.random_seed}")

    CONCURRENT_BATCH_SIZE = args.concurrent_batch

    # Auto-configure dataset-specific sample counts if not specified by user
    dataset_config = DATASET_SAMPLE_CONFIGS.get(args.dataset, {"validate": 100, "test": 800})
    sample_auto_configured = False
    test_samples_auto_configured = False

    if args.sample is None:
        args.sample = dataset_config["validate"]
        sample_auto_configured = True

    if args.test_samples is None:
        args.test_samples = dataset_config["test"]
        test_samples_auto_configured = True

    print(f"[Config] Dataset: {args.dataset}")
    print(f"[Config] Optimization model: {args.opt_model_name}")
    print(f"[Config] Execution model: {args.exec_model_name}")
    print(f"[Config] Max rounds: {args.max_rounds}, Samples per round: {args.sample}" +
          (" (dataset default)" if sample_auto_configured else " (user specified)"))
    print(f"[Config] Test samples: {args.test_samples}" +
          (" (dataset default)" if test_samples_auto_configured else " (user specified)"))
    print(f"[Config] Concurrent batch size: {args.concurrent_batch}")
    print(f"[Config] GNN: hidden_dim={args.hidden_dim}, lr={args.lr}")

    # Load experiment config
    experiment_configs = get_experiment_configs()
    config = experiment_configs[args.dataset]

    # Load LLM configurations
    models_config = LLMsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(f"Optimization model '{args.opt_model_name}' not found in config.")

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(f"Execution model '{args.exec_model_name}' not found in config.")

    # Download dataset if needed
    print("\n[Step 1] Downloading dataset...")
    download(["datasets"], force_download=args.if_force_download)

    # Determine prompt domain directory
    prompt_domain_dir = args.prompt_domain_dir or PROMPT_DOMAIN_DIR

    # Create LLM for prompt generation
    print("\n[Step 2] Building prompt domains...")
    meta_llm = create_llm_instance(opt_llm_config)

    prompt_domains = asyncio.run(
        build_prompt_domains(
            llm=meta_llm,
            prompt_configs=PROMPT_CONFIGS,
            num_prompts=args.num_prompts,
            domain_dir=prompt_domain_dir,
            dataset=args.dataset,
        )
    )
    print(f"[Step 2] Loaded {len(prompt_domains)} prompt domains")

    workflow_topology = get_workflow_topology(args.dataset)

    print(f"\n[Step 3] Workflow topology ({len(workflow_topology)} operators):")
    for i, topo in enumerate(workflow_topology):
        deps = topo.get("dependencies", [])
        deps_str = " → ".join(deps) + " → " if deps else ""
        print(f"  [{i}] {deps_str}{topo['name']}")

    # Extract prompt lists for each operator
    workflow_agent_prompt = [prompt_domains[topo["prompt_domain"]] for topo in workflow_topology]

    # Load sentence embedding model
    if EMBEDDING_MODE == "openrouter":
        print(f"\n[Step 4] Using OpenRouter embedding API: {OPENROUTER_EMBEDDING_MODEL}")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set!")
        sen_tokenizer = None
        sen_model = None
    else:
        print(f"\n[Step 4] Loading local sentence embedding model: {SENTENCE_MODEL_NAME}")
        sen_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_MODEL_NAME)
        sen_model = AutoModel.from_pretrained(SENTENCE_MODEL_NAME)

    # Compute embeddings for each operator's prompt domain
    print("\n[Step 5] Computing prompt embeddings...")
    all_operator_embeddings = []
    for i, prompts in enumerate(workflow_agent_prompt):
        embeddings = get_sen_embedding(sen_model, sen_tokenizer, prompts)
        embeddings = embeddings.to(**TKWARGS)
        all_operator_embeddings.append(embeddings)
        print(f"  Operator {i} ({workflow_topology[i]['name']}): {embeddings.shape[0]} prompts, dim={embeddings.shape[1]}")

    embedding_dim = all_operator_embeddings[0].shape[1]
    num_operators = len(workflow_topology)

    # Determine edge direction
    use_bidirectional = args.bidirectional

    # Initialize GAT model
    # GSM8K scores cluster near 1.0, so scale [0.8, 1.0] → [0, 1] to amplify differences
    if args.dataset == "GSM8K":
        score_min, score_max = 0.80, 1.0
    else:
        score_min, score_max = 0.0, 1.0

    use_sigmoid = True

    gnn_model = WorkflowGAT(
        embedding_dim=embedding_dim,
        num_operators=num_operators,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        topology=workflow_topology,
        bidirectional=use_bidirectional,
        use_sigmoid=use_sigmoid,
        score_min=score_min,
        score_max=score_max,
    ).to(**TKWARGS)

    # Frozen initial model for gradient features (Neural UCB only)
    if args.ucb_type == "neural":
        frozen_model = copy.deepcopy(gnn_model)
        for param in frozen_model.parameters():
            param.requires_grad_(True)  # gradients needed for feature computation, but weights are never updated
        frozen_model.eval()
    else:
        frozen_model = None

    # Optimizer with weight decay
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Dynamically load the Workflow class for the selected dataset
    WorkflowClass = load_workflow_class(args.dataset)
    flow = WorkflowClass(name=f"{args.dataset}_workflow", llm_config=exec_llm_config, dataset=config.dataset)
    eval_data_path = os.path.join("data", "datasets", f"{config.dataset.lower()}_validate.jsonl")

    from datetime import datetime
    experiment_name = (
        f"s{args.sample}_pt{args.pretrain_rounds}_r{args.max_rounds}_p{args.num_prompts}"
        f"_h{args.hidden_dim}_l{args.num_gnn_layers}_lr{args.lr}_wd{args.weight_decay}"
        f"_d{args.dropout}_{args.ucb_type}_seed{args.random_seed}"
    )
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(
        "result_CSV", "MASPO", args.dataset,
        f"{experiment_name}_{experiment_time}"
    )
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"\n[Step 6.5] Experiment results will be saved to: {experiment_folder}")

    from functools import partial
    eval_func = partial(evaluate_workflow, dataset=args.dataset, log_path=experiment_folder)

    # Phase 1: Pretrain GNN with random prompt selection
    print(f"\n[Step 7] Pretraining GNN ({args.pretrain_rounds} rounds, UCB type: {args.ucb_type})...")
    pretrain_history, pretrain_best_prompts, pretrain_data, pretrain_fisher_matrix = asyncio.run(
        run_pretrain_phase(
            gnn_model=gnn_model,
            frozen_model=frozen_model,
            optimizer=gnn_optimizer,
            all_operator_embeddings=all_operator_embeddings,
            workflow_agent_prompt=workflow_agent_prompt,
            flow=flow,
            topology=workflow_topology,
            eval_func=eval_func,
            eval_data_path=eval_data_path,
            pretrain_rounds=args.pretrain_rounds,
            pretrain_epochs=args.pretrain_epochs,
            max_eval_examples=args.sample,
            lambda_reg=args.lambda_reg,
            ucb_type=args.ucb_type,
            patience=args.patience,
            min_delta=args.min_delta,
            fisher_coef=args.fisher_coef,
        )
    )

    # Phase 2: Run UCB optimization with GNN
    print(f"\n[Step 8] Running {args.ucb_type.upper()} Optimization ({args.max_rounds} rounds, alpha={args.ucb_alpha})...")
    history = asyncio.run(
        run_gnn_optimization(
            gnn_model=gnn_model,
            frozen_model=frozen_model,
            optimizer=gnn_optimizer,
            all_operator_embeddings=all_operator_embeddings,
            workflow_agent_prompt=workflow_agent_prompt,
            flow=flow,
            topology=workflow_topology,
            eval_func=eval_func,
            eval_data_path=eval_data_path,
            num_rounds=args.max_rounds,
            max_eval_examples=args.sample,
            pretrain_best_prompts=pretrain_best_prompts,
            pretrain_data=pretrain_data,
            pretrain_fisher_matrix=pretrain_fisher_matrix,
            train_epochs=args.pretrain_epochs,
            ucb_alpha=args.ucb_alpha,
            lambda_reg=args.lambda_reg,
            ucb_type=args.ucb_type,
            patience=args.patience,
            min_delta=args.min_delta,
            search_strategy=args.search_strategy,
            initial_lr=args.lr,
            fisher_coef=args.fisher_coef,
        )
    )

    # Merge pretrain and optimization history
    history["pretrain"] = pretrain_history

    # Save results
    results_path = save_experiment_results(args, config, history, pretrain_history)

    # ==========================================================================
    # Phase 3: Test Phase
    # ==========================================================================
    if args.run_test:
        print(f"\n[Step 9] Running TEST PHASE on test set...")
        print(f"  Test Repeats: {args.test_repeats}")

        test_data_path = os.path.join("data", "datasets", f"{config.dataset.lower()}_test.jsonl")

        if not os.path.exists(test_data_path):
            print(f"  [WARNING] Test set not found: {test_data_path}")
            print(f"  [WARNING] Falling back to validation set for testing...")
            test_data_path = eval_data_path

        # Collect all prompt combinations and their validation scores
        all_prompt_scores = []

        # From pretrain history
        if pretrain_history and 'prompts' in pretrain_history and 'scores' in pretrain_history:
            for prompts, score in zip(pretrain_history['prompts'], pretrain_history['scores']):
                all_prompt_scores.append((prompts, score))

        # From optimization history
        if history['selected_prompts'] and history['actual_scores']:
            for prompts, score in zip(history['selected_prompts'], history['actual_scores']):
                all_prompt_scores.append((prompts, score))

        # Deduplicate: keep highest score per combination
        unique_prompts = {}
        for prompts, score in all_prompt_scores:
            prompts_key = tuple(prompts)
            if prompts_key not in unique_prompts or score > unique_prompts[prompts_key]:
                unique_prompts[prompts_key] = score

        # Sort by score descending
        sorted_prompts = sorted(unique_prompts.items(), key=lambda x: x[1], reverse=True)

        print(f"  Found {len(unique_prompts)} unique prompt combinations")

        if sorted_prompts:
            best_prompts = list(sorted_prompts[0][0])
            best_val_score = sorted_prompts[0][1]
            print(f"  Best prompts on validation set: {best_prompts} -> {best_val_score:.4f}")
        elif history['actual_scores']:
            best_score_idx = history['actual_scores'].index(max(history['actual_scores']))
            best_prompts = history['selected_prompts'][best_score_idx]
            best_val_score = history['actual_scores'][best_score_idx]
        else:
            best_prompts = pretrain_best_prompts
            best_val_score = 0.0

        test_eval_func = partial(evaluate_workflow, dataset=args.dataset, log_path=experiment_folder)

        test_results = asyncio.run(
            run_test_phase(
                gnn_model=gnn_model,
                all_operator_embeddings=all_operator_embeddings,
                workflow_agent_prompt=workflow_agent_prompt,
                flow=flow,
                topology=workflow_topology,
                eval_func=test_eval_func,
                test_data_path=test_data_path,
                best_prompts=best_prompts,
                test_samples=args.test_samples,
                ucb_type=args.ucb_type,
                best_val_score=best_val_score,
                test_repeats=args.test_repeats,
            )
        )

        test_results_path = os.path.join(experiment_folder, "test_results.json")
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"\n[Step 9] Test results saved to: {test_results_path}")

        # Also merge test results into the main results file
        if results_path and os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    main_results = json.load(f)
                main_results["test"] = test_results
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(main_results, f, indent=2, ensure_ascii=False)
                print(f"[Step 9] Test results also merged into: {results_path}")
            except Exception as e:
                print(f"[WARNING] Failed to merge test results into main file: {e}")

        print(f"\n{'='*70}")
        print(f"TEST PHASE FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Optimized Score':<40} {test_results['optimized_score']:.4f} ± {test_results['optimized_std']:.4f}")
        print(f"  {'-'*70}")
        print(f"  {'Best Prompts (from Validation)':<40} {test_results['best_prompts']}")
        print(f"  {'Validation Score':<40} {test_results['best_val_score']:.4f}")
        print(f"  {'-'*70}")
        print(f"  {'UCB Type':<40} {test_results['ucb_type']}")
        print(f"  {'Test Samples':<40} {test_results['test_samples']}")
        print(f"  {'Test Repeats':<40} {test_results['test_repeats']}")
        print(f"{'='*70}\n")
