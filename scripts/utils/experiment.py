import csv
import json
import os
from datetime import datetime


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", bar_len: int = 30):
    """Print a simple progress bar to stdout."""
    progress = (current + 1) / total
    filled = int(bar_len * progress)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}[{bar}] {current + 1}/{total} ({progress*100:.0f}%) {suffix}", end="", flush=True)
    if current == total - 1:
        print()  # newline at the end


def save_experiment_results(args, config, history, pretrain_history) -> str:
    """Save experiment results to a JSON file and a CSV log.

    Directory layout:
        results/MASPO/{dataset}/{filename}.json
        result_CSV/MASPO/{dataset}/{filename}.csv

    File naming:
        {model}_{ucb_type}_s{sample}_pt{pretrain_rounds}_r{max_rounds}_p{num_prompts}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize model name (replace special chars)
    model_name_safe = args.exec_model_name.replace("/", "_").replace(":", "_")

    filename_base = (
        f"{model_name_safe}_{args.ucb_type}_s{args.sample}_pt{args.pretrain_rounds}"
        f"_r{args.max_rounds}_p{args.num_prompts}_h{args.hidden_dim}"
        f"_alpha{args.ucb_alpha}_seed{args.random_seed}_{timestamp}"
    )

    # Save JSON
    results_dir = os.path.join("results", "MASPO", config.dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{filename_base}.json")

    # Attach config metadata to history
    history["config"] = {
        "dataset": config.dataset,
        "sample": args.sample,
        "pretrain_rounds": args.pretrain_rounds,
        "pretrain_epochs": args.pretrain_epochs,
        "max_rounds": args.max_rounds,
        "num_prompts": args.num_prompts,
        "hidden_dim": args.hidden_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "opt_model": args.opt_model_name,
        "exec_model": args.exec_model_name,
        "ucb_type": args.ucb_type,
        "ucb_alpha": args.ucb_alpha,
        "lambda_reg": args.lambda_reg,
        "random_seed": args.random_seed,
        "concurrent_batch": args.concurrent_batch,
        "timestamp": timestamp,
    }

    if pretrain_history:
        history["pretrain"] = pretrain_history

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n[Results] JSON saved to {results_path}")

    # Save CSV
    csv_dir = os.path.join("result_CSV", "MASPO", config.dataset)
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{filename_base}.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Write config as comment rows
        writer.writerow(["# MASPO Experiment Results"])
        writer.writerow([f"# Dataset: {config.dataset}"])
        writer.writerow([f"# Model: {args.exec_model_name}"])
        writer.writerow([f"# UCB Type: {args.ucb_type}"])
        writer.writerow([f"# Timestamp: {timestamp}"])
        writer.writerow([])

        # Pretrain results
        if pretrain_history and pretrain_history.get("scores"):
            writer.writerow(["## Pretrain Phase"])
            writer.writerow(["round", "prompts", "actual_score", "predicted_score", "error"])
            pt_scores = pretrain_history["scores"]
            pt_prompts = pretrain_history["prompts"]
            pt_preds = pretrain_history.get("predicted_scores", [])
            pt_errors = pretrain_history.get("errors", [])

            for i in range(len(pt_scores)):
                pred = pt_preds[i] if i < len(pt_preds) else 0.0
                error = pt_errors[i] if i < len(pt_errors) else 0.0
                writer.writerow([i + 1, str(pt_prompts[i]), pt_scores[i], pred, error])

            # Pretrain summary
            pt_best = max(pt_scores)
            pt_avg = sum(pt_scores) / len(pt_scores)
            writer.writerow(["PRETRAIN_SUMMARY", f"best={pt_best:.4f}", f"avg={pt_avg:.4f}",
                           f"count={len(pt_scores)}", ""])
            writer.writerow([])

        # Optimization results
        if history and history.get("actual_scores"):
            writer.writerow(["## Optimization Phase"])
            writer.writerow(["round", "prompts", "predicted", "uncertainty", "ucb", "actual", "error", "loss", "selection_time"])

            opt_scores = history["actual_scores"]
            predicted = history.get("predicted_scores", [])
            uncertainties = history.get("uncertainties", [])
            ucb_scores = history.get("ucb_scores", [])
            prompts_list = history.get("selected_prompts", [])
            losses = history.get("losses", [])
            selection_times = history.get("selection_times", [])

            for i in range(len(opt_scores)):
                prompt_str = str(prompts_list[i]) if i < len(prompts_list) else "N/A"
                pred = predicted[i] if i < len(predicted) else 0.0
                unc = uncertainties[i] if i < len(uncertainties) else 0.0
                ucb = ucb_scores[i] if i < len(ucb_scores) else 0.0
                actual = opt_scores[i]
                error = abs(pred - actual)
                loss = losses[i] if i < len(losses) else 0.0
                sel_time = selection_times[i] if i < len(selection_times) else 0.0
                writer.writerow([i + 1, prompt_str, pred, unc, ucb, actual, error, loss, sel_time])

            # Optimization summary
            best_score = max(opt_scores)
            best_idx = opt_scores.index(best_score)
            opt_avg = sum(opt_scores) / len(opt_scores)
            total_sel_time = history.get("total_selection_time", 0.0)
            search_strategy = history.get("search_strategy", "unknown")
            writer.writerow(["OPT_SUMMARY", f"best={best_score:.4f}", f"best_round={best_idx + 1}",
                           f"avg={opt_avg:.4f}", f"count={len(opt_scores)}",
                           f"total_sel_time={total_sel_time:.4f}s", f"strategy={search_strategy}", "", ""])

    print(f"[Results] CSV saved to {csv_path}")

    return results_path


