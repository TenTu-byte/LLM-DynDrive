#!/usr/bin/env python3
"""
W&B sweep runner for dynamic steering (conf) on the cluster.

This script:
  1) Defines a Bayesian sweep over q25, q75, low_val, tau near their defaults.
  2) Launches the existing entry script for each sweep run.
  3) Parses metrics written by the entry script and logs them to W&B.

It keeps the existing env-based interface intact so platform configs can still
set ${steer_layer}, ${max_tokens}, ${seed}, ${outputs}, ${models}, etc.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from typing import Dict, List, Tuple

import wandb

# --------------------------
# Sweep defaults (near base)
# --------------------------
DEFAULTS = {
    "q25": 0.75,
    "q75": 0.92,
    "low_val": -4.9,
    "tau": 0.0,
}

# Tight ranges around defaults (tune as needed).
RANGES = {
    "q25": (0.7, 0.8),
    "q75": (0.87, 0.97),
    "low_val": (-5.4, -4.4),
    "tau": (-0.3, 0.1),
}

# Constraint for token length.
TOKEN_BUDGET = 11200


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _setup_wandb_dirs(outputs_root: str) -> None:
    """
    Force all W&B artifacts/config/cache into outputs_root to match
    existing output conventions on the cluster.
    """
    outputs_root = os.path.abspath(outputs_root)
    wandb_dir = os.path.join(outputs_root, "wandb")
    wandb_cache = os.path.join(outputs_root, "wandb_cache")
    wandb_config = os.path.join(outputs_root, "wandb_config")
    wandb_artifacts = os.path.join(outputs_root, "wandb_artifacts")

    _ensure_dir(wandb_dir)
    _ensure_dir(wandb_cache)
    _ensure_dir(wandb_config)
    _ensure_dir(wandb_artifacts)

    # Only set defaults if the user/platform has not explicitly set them.
    os.environ.setdefault("WANDB_DIR", wandb_dir)
    os.environ.setdefault("WANDB_CACHE_DIR", wandb_cache)
    os.environ.setdefault("WANDB_CONFIG_DIR", wandb_config)
    os.environ.setdefault("WANDB_ARTIFACT_DIR", wandb_artifacts)


def _build_sweep_config() -> Dict:
    """
    Build a Bayesian sweep config focused around the default values.
    """
    return {
        "method": "bayes",
        "metric": {"name": "objective", "goal": "maximize"},
        "parameters": {
            "q25": {"min": RANGES["q25"][0], "max": RANGES["q25"][1]},
            "q75": {"min": RANGES["q75"][0], "max": RANGES["q75"][1]},
            "low_val": {"min": RANGES["low_val"][0], "max": RANGES["low_val"][1]},
            "tau": {"min": RANGES["tau"][0], "max": RANGES["tau"][1]},
        },
    }


def _required_env_check() -> None:
    """
    Ensure required env vars are present; keep it minimal to avoid
    breaking existing platform defaults.
    """
    missing = []
    for key in ("outputs", "models", "steer_layer", "max_tokens", "seed"):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        raise RuntimeError(
            "Missing required env vars for entry script: "
            + ", ".join(missing)
        )


def _run_entry_script(entry_script: str, env: Dict[str, str]) -> int:
    """
    Run the existing entry script. We use bash explicitly so the script
    does not need an executable bit.
    """
    cmd = ["bash", entry_script]
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _find_metrics_files(output_root: str, run_id: str, start_time: float) -> List[str]:
    """
    Collect metrics files created for this run_id after start_time.
    """
    pattern = os.path.join(output_root, "**", "*.metrics.json")
    candidates = []
    for path in glob.glob(pattern, recursive=True):
        base = os.path.basename(path)
        if run_id not in base:
            continue
        try:
            if os.path.getmtime(path) < start_time - 2:
                continue
        except OSError:
            continue
        candidates.append(path)
    return sorted(candidates)


def _aggregate_metrics(metric_paths: List[str]) -> Tuple[float, float, Dict[str, Dict]]:
    """
    Aggregate metrics across datasets and return:
      - avg accuracy (float)
      - avg full token count (float)
      - per-dataset metrics (for logging)
    """
    acc_vals = []
    token_vals = []
    per_dataset = {}

    for path in metric_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        dataset = data.get("dataset", os.path.basename(path))
        acc = data.get("accuracy")
        token_stats = data.get("token_stats", {})
        avg_full = token_stats.get("avg_full_token_count")

        if isinstance(acc, (int, float)):
            acc_vals.append(float(acc))
        if isinstance(avg_full, (int, float)):
            token_vals.append(float(avg_full))

        per_dataset[dataset] = {
            "accuracy": acc,
            "avg_full_token_count": avg_full,
            "path": path,
        }

    avg_acc = sum(acc_vals) / len(acc_vals) if acc_vals else 0.0
    avg_full_tokens = sum(token_vals) / len(token_vals) if token_vals else float("inf")
    return avg_acc, avg_full_tokens, per_dataset


def _objective(acc: float, avg_full_tokens: float, token_budget: int) -> float:
    """
    Constrained objective:
      - If token length <= budget, prioritize accuracy.
      - If above budget, penalize proportionally.
    """
    if not (isinstance(acc, float) and isinstance(avg_full_tokens, float)):
        return -1.0
    if avg_full_tokens <= token_budget:
        return acc
    # Penalty grows with budget overflow; keeps objective comparable to acc.
    return acc - (avg_full_tokens - token_budget) / float(token_budget)


def _format_float(val: float) -> str:
    return f"{val:.6f}"


def _sweep_run(entry_script: str, output_root: str, token_budget: int) -> None:
    """
    Single W&B sweep run:
      - read sweep config
      - call entry script
      - parse metrics and log to W&B
    """
    with wandb.init() as run:
        cfg = run.config

        # Extract params as floats.
        q25 = float(cfg.q25)
        q75 = float(cfg.q75)
        low_val = float(cfg.low_val)
        tau = float(cfg.tau)

        # Guard invalid configurations early.
        if q25 >= q75:
            wandb.log(
                {
                    "invalid_config": 1,
                    "objective": -1.0,
                    "acc": 0,
                    "avg_full_tokens": 999999,
                }
            )
            return

        # Use W&B run id to keep output filenames unique.
        run_id = f"sweep_{run.id}"
        run.name = run_id

        env = os.environ.copy()
        env["q25"] = _format_float(q25)
        env["q75"] = _format_float(q75)
        env["low_val"] = _format_float(low_val)
        env["tau"] = _format_float(tau)
        env["run_id"] = run_id

        start_time = time.time()
        rc = _run_entry_script(entry_script, env)
        if rc != 0:
            wandb.log(
                {
                    "run_failed": 1,
                    "exit_code": rc,
                    "objective": -1.0,
                    "acc": 0,
                    "avg_full_tokens": 999999,
                }
            )
            return

        metric_paths = _find_metrics_files(output_root, run_id, start_time)
        avg_acc, avg_full_tokens, per_dataset = _aggregate_metrics(metric_paths)

        # Convert to ints as requested (acc in percent, avg_full_tokens in tokens).
        acc_int = int(round(avg_acc * 100))
        avg_full_tokens_int = int(round(avg_full_tokens)) if avg_full_tokens != float("inf") else 999999

        objective = _objective(avg_acc, avg_full_tokens, token_budget)

        log_payload = {
            "acc": acc_int,
            "acc_raw": avg_acc,
            "avg_full_tokens": avg_full_tokens_int,
            "avg_full_tokens_raw": avg_full_tokens,
            "objective": objective,
            "metrics_files": len(metric_paths),
        }

        # Log per-dataset details with clear prefixes.
        for ds_name, ds_data in per_dataset.items():
            if isinstance(ds_data.get("accuracy"), (int, float)):
                log_payload[f"acc/{ds_name}"] = ds_data["accuracy"]
            if isinstance(ds_data.get("avg_full_token_count"), (int, float)):
                log_payload[f"avg_full_tokens/{ds_name}"] = ds_data["avg_full_token_count"]

        wandb.log(log_payload)
        run.summary["metrics_paths"] = metric_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run W&B sweep for dynamic steering (conf)."
    )
    parser.add_argument(
        "--entry_script",
        type=str,
        default="scripts_yulin/dynamic_steer_pangu_dist_conf.sh",
        help="Existing entry script used by the cluster.",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=os.path.join(
            os.environ.get("outputs"),
            "beta",
            "outputs_steer_dynamic_conf",
        ),
        help="Root output directory; W&B files will be written under this path.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "rebalance_pangu"),
        help="W&B project name.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "yulin_edu-hit"),
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=os.environ.get("WANDB_SWEEP_ID", ""),
        help="Existing sweep id. If empty, a new sweep is created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep runs to execute in this agent.",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=TOKEN_BUDGET,
        help="Constraint for avg_full_tokens.",
    )
    args = parser.parse_args()

    entry_script = os.path.abspath(args.entry_script)
    if not os.path.exists(entry_script):
        raise FileNotFoundError(f"Entry script not found: {entry_script}")

    _setup_wandb_dirs(args.outputs_root)
    _required_env_check()

    sweep_config = _build_sweep_config()
    sweep_id = args.sweep_id.strip()
    if not sweep_id:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project,
            entity=args.entity,
        )
        print(f"[W&B] Created sweep: {sweep_id}")

    # Run agent for N trials in this process.
    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: _sweep_run(entry_script, args.outputs_root, args.token_budget),
        count=args.count,
    )


if __name__ == "__main__":
    main()
