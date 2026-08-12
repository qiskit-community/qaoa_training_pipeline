#!/usr/bin/env python
"""Compare the *solution quality* of torch vs ONNX predicted QAOA angles.

Numeric angle parity (see tests) proves the graphs compute the same function.
This script answers the downstream question: do the ONNX angles solve QAOA as
well as the torch angles? For every model and operator it builds the p=1 QAOA
state with each predictor's angles, evaluates the exact expected cost with the
repo's StatevectorEvaluator, and reports the approximation ratio against the
brute-force optimum.

    python scripts/compare_quality.py
    python scripts/compare_quality.py --model graph_transformer
"""

from __future__ import annotations

import argparse
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_ops import BENCH_OPS  # noqa: E402
from qaoa_training_pipeline.inference.config_io import load_config  # noqa: E402

MODEL_CONFIGS_DIR = REPO_ROOT / "qaoa_training_pipeline" / "inference" / "model_configs"


def _checkpoint_present(config_path: Path) -> bool:
    cfg = load_config(config_path)
    ckpt = cfg.get("checkpoint")
    return bool(ckpt) and (config_path.parent / ckpt).resolve().is_file()


def _optimum(cost_op) -> tuple[float, float]:
    """Exact (min, max) eigenvalue of the diagonal cost operator by brute force.

    These operators are diagonal in the computational basis, so the spectrum is
    just the diagonal of the matrix — cheap for the small benchmark graphs.
    """
    diag = np.real(np.diag(cost_op.to_matrix()))
    return float(diag.min()), float(diag.max())


def _approx_ratio(expected_cost: float, opt_min: float, opt_max: float) -> float:
    """Approximation ratio in [0, 1]; 1.0 == reaches the minimizing optimum.

    QAOA here minimizes <H_C>. Normalize between the worst (max) and best (min)
    achievable cost so the metric is comparable across graphs of different scale.
    """
    if abs(opt_max - opt_min) < 1e-12:
        return 1.0
    return (opt_max - expected_cost) / (opt_max - opt_min)


def compare_model(model_name: str, evaluator) -> dict | None:
    from qaoa_training_pipeline.inference.torch_backend.lightweight_predictor import (
        LightweightQAOAPredictor,
    )
    from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

    config_path = MODEL_CONFIGS_DIR / model_name / "model_config.json"
    if not (MODEL_CONFIGS_DIR / model_name / "model.onnx").is_file():
        print(f"[{model_name}] SKIP: no model.onnx")
        return None
    if not _checkpoint_present(config_path):
        print(f"[{model_name}] SKIP: checkpoint not present")
        return None

    torch_pred = LightweightQAOAPredictor(config_path=config_path, device="cpu", strict=True)
    onnx_pred = OnnxQAOAPredictor(config_path=config_path, device="cpu")

    torch_ratios, onnx_ratios, ratio_gaps, cost_gaps = [], [], [], []
    for op in BENCH_OPS.values():
        opt_min, opt_max = _optimum(op)
        t_cost = evaluator.evaluate(op, torch_pred.predict(op))
        o_cost = evaluator.evaluate(op, onnx_pred.predict(op))
        t_ratio = _approx_ratio(t_cost, opt_min, opt_max)
        o_ratio = _approx_ratio(o_cost, opt_min, opt_max)
        torch_ratios.append(t_ratio)
        onnx_ratios.append(o_ratio)
        ratio_gaps.append(abs(t_ratio - o_ratio))
        cost_gaps.append(abs(t_cost - o_cost))

    result = {
        "model": model_name,
        "torch_mean_ratio": statistics.fmean(torch_ratios),
        "onnx_mean_ratio": statistics.fmean(onnx_ratios),
        "max_ratio_gap": max(ratio_gaps),
        "max_cost_gap": max(cost_gaps),
    }
    print(
        f"[{model_name:26s}] "
        f"approx-ratio torch {result['torch_mean_ratio']:.4f} | "
        f"onnx {result['onnx_mean_ratio']:.4f} | "
        f"max ratio gap {result['max_ratio_gap']:.2e} | "
        f"max cost gap {result['max_cost_gap']:.2e}"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare torch vs ONNX QAOA solution quality.")
    ap.add_argument("--model", default="all", help="model_config name, or 'all'")
    args = ap.parse_args()

    from qaoa_training_pipeline.evaluation import EVALUATORS

    evaluator = EVALUATORS["StatevectorEvaluator"]()

    if args.model == "all":
        names = sorted(
            p.name for p in MODEL_CONFIGS_DIR.iterdir() if (p / "model_config.json").exists()
        )
    else:
        names = [args.model]

    print(f"evaluator=StatevectorEvaluator ops={len(BENCH_OPS)}\n")
    results = [r for r in (compare_model(n, evaluator) for n in names) if r]

    if results:
        worst_gap = max(r["max_ratio_gap"] for r in results)
        print(
            f"\nworst approximation-ratio gap (torch vs onnx) across all models: "
            f"{worst_gap:.2e}"
        )


if __name__ == "__main__":
    main()
