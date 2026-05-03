#!/usr/bin/env python
"""Find top 6 hyperparameter configurations by test accuracy or AUROC."""

import json
import math
from pathlib import Path
from typing import Any


def _coerce_metric(value: Any) -> float:
    """Convert metric values to float; map missing/non-finite values to -inf."""
    if value is None:
        return float("-inf")
    try:
        num = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return num if math.isfinite(num) else float("-inf")


def _format_metric(value: float) -> str:
    """Pretty print metric values while preserving missing values as N/A."""
    return "N/A" if value == float("-inf") else f"{value:.4f}"


def find_top_hyperparameters(
    checkpoint_dir: str = "results/checkpoints",
    metric: str = "final_accuracy",
    top_k: int = 6,
    model_type: str = "mlp",  # "mlp", "tf", or "baseline"
) -> list[dict[str, Any]]:
    """
    Scan all completed runs and return top K by specified metric.
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        metric: Metric to sort by ("final_accuracy" or "final_macro_auroc")
        top_k: Number of top configs to return
        model_type: Filter by model type ("mlp", "tf", "baseline")
    
    Returns:
        List of dicts with keys: config, metrics, accuracy, auroc, run_dir
    """
    metric_aliases = {
        "final_accuracy": "accuracy",
        "accuracy": "accuracy",
        "final_macro_auroc": "auroc",
        "macro_auroc": "auroc",
        "auroc": "auroc",
    }
    trial_prefix = {
        "baseline": "baseline",
        "mlp": "mlp",
        "mlp2": "mlp2",
        "tf": "tf",
    }.get(model_type, model_type)
    sort_key = metric_aliases.get(metric)
    if sort_key is None:
        allowed = ", ".join(sorted(metric_aliases))
        raise ValueError(f"Unsupported metric '{metric}'. Use one of: {allowed}")

    checkpoint_path = Path(checkpoint_dir)
    tune_dir = checkpoint_path / "tune"
    
    if not tune_dir.exists():
        print(f"No tuning directory found at {tune_dir}")
        return []
    
    results = []
    
    # Scan all trial directories
    for trial_dir in sorted(tune_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        
        # Filter by model type if specified. Match the token before the first underscore
        # so the secondary sweep (mlp2_...) does not get mixed into the original mlp run.
        if trial_prefix:
            trial_bucket = trial_dir.name.split("_", 1)[0]
            if trial_bucket != trial_prefix:
                continue
        
        # Tuning outputs are nested as:
        # tune/<trial_name>/<run_name>/metrics.json
        metric_candidates = sorted(trial_dir.rglob("metrics.json"))
        if not metric_candidates:
            continue
        metrics_file = metric_candidates[-1]
        
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Extract config from trial name (e.g., "mlp_s42_lr0.001_t0.07_b256_h256_e128_cd0.2")
            config = parse_trial_name(trial_dir.name)
            
            results.append({
                "trial_name": trial_dir.name,
                "run_dir": str(metrics_file.parent),
                "config": config,
                "accuracy": _coerce_metric(metrics.get("final_accuracy")),
                "auroc": _coerce_metric(metrics.get("final_macro_auroc")),
                "metrics": metrics,
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Skipping {metrics_file}: {e}")
            continue
    
    if not results:
        print(f"No completed runs found in {tune_dir}")
        return []
    
    # Sort by requested metric
    results.sort(key=lambda x: _coerce_metric(x.get(sort_key)), reverse=True)
    
    # Print results
    print(f"\n{'='*120}")
    print(f"Top {top_k} configurations by {metric}:")
    print(f"{'='*120}")
    
    for rank, result in enumerate(results[:top_k], 1):
        print(f"\nRank #{rank}: {result['trial_name']}")
        print(f"  Accuracy:      {_format_metric(result['accuracy'])}")
        print(f"  Macro AUROC:   {_format_metric(result['auroc'])}")
        print(f"  Config:        {result['config']}")
        print(f"  Run Dir:       {result['run_dir']}")
    
    print(f"\n{'='*120}\n")
    
    return results[:top_k]


def parse_trial_name(trial_name: str) -> dict[str, str]:
    """
    Parse trial name like:
      baseline_rna_seed42_20260414_120530
      mlp_s42_lr0.001_t0.07_b256_h256_e128_cd0.2
      tf_s42_lr0.001_t0.07_b256_d64_n4_f256_drop0.1_cd0.2
    """
    config = {}
    parts = trial_name.split("_")
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Handle seed
        if part.startswith("s") and part[1:].replace(".", "").isdigit():
            config["seed"] = part[1:]
            i += 1
        # Handle learning rate (lr)
        elif part.startswith("lr"):
            config["lr"] = part[2:]
            i += 1
        # Handle temperature (t)
        elif part.startswith("t") and i + 1 < len(parts) and not parts[i + 1].startswith(("b", "h", "e", "d", "n", "f", "c")):
            config["temperature"] = part[1:]
            i += 1
        # Handle batch size (b)
        elif part.startswith("b") and len(part) > 1 and part[1].isdigit():
            config["batch_size"] = part[1:]
            i += 1
        # Handle hidden dim (h)
        elif part.startswith("h") and part[1:].isdigit():
            config["hidden_dim"] = part[1:]
            i += 1
        # Handle embedding dim (e)
        elif part.startswith("e") and part[1:].isdigit():
            config["embedding_dim"] = part[1:]
            i += 1
        # Handle transformer d_model (d)
        elif part.startswith("d") and part[1:].isdigit():
            config["d_model"] = part[1:]
            i += 1
        # Handle transformer heads (n)
        elif part.startswith("n") and part[1:].isdigit():
            config["heads"] = part[1:]
            i += 1
        # Handle transformer ffn (f)
        elif part.startswith("f") and part[1:].isdigit():
            config["ffn_dim"] = part[1:]
            i += 1
        # Handle dropout (drop or cd)
        elif part.startswith("drop"):
            config["dropout"] = part[4:]
            i += 1
        elif part.startswith("cd"):
            config["classifier_dropout"] = part[2:]
            i += 1
        else:
            i += 1
    
    return config


if __name__ == "__main__":
    import sys
    
    metric = sys.argv[1] if len(sys.argv) > 1 else "final_accuracy"
    model_type = sys.argv[2] if len(sys.argv) > 2 else "mlp"
    checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else "results/checkpoints"
    
    results = find_top_hyperparameters(
        checkpoint_dir=checkpoint_dir,
        metric=metric,
        model_type=model_type,
    )
    
    if results:
        print("\nTop configuration details:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['trial_name']}")
            print(json.dumps(result['config'], indent=2))
