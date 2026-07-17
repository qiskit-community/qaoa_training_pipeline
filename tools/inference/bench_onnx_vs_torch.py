#!/usr/bin/env python
"""Benchmark torch vs ONNX inference latency, and confirm they agree.

For each model architecture, times per-prediction latency of the torch
``LightweightQAOAPredictor`` and the torch-free ``OnnxQAOAPredictor`` over the
shared operator set in ``bench_ops.py``, reports median/mean/p95 and the
speedup, and prints the max angle difference between the two so a performance
run doubles as a correctness spot-check.

    python scripts/bench_onnx_vs_torch.py
    python scripts/bench_onnx_vs_torch.py --model gcn --repeats 200 --warmup 20
    python scripts/bench_onnx_vs_torch.py --json bench_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

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


def _time_predict(predict, ops, repeats, warmup):
    """Return per-call latencies in milliseconds (cycles through ops)."""
    op_list = list(ops.values())
    for i in range(warmup):
        predict(op_list[i % len(op_list)])
    latencies = []
    for i in range(repeats):
        op = op_list[i % len(op_list)]
        start = time.perf_counter()
        predict(op)
        latencies.append((time.perf_counter() - start) * 1e3)
    return latencies


def _stats(latencies):
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.fmean(latencies),
        "p95_ms": p95,
        "min_ms": min(latencies),
    }


def _max_angle_diff(a_predict, b_predict, ops):
    worst = 0.0
    for op in ops.values():
        a = a_predict(op)
        b = b_predict(op)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return worst


def bench_model(model_name, repeats, warmup):
    from qaoa_training_pipeline.inference.torch_backend.lightweight_predictor import (
        LightweightQAOAPredictor,
    )
    from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

    config_path = MODEL_CONFIGS_DIR / model_name / "model_config.json"
    onnx_file = MODEL_CONFIGS_DIR / model_name / "model.onnx"
    if not onnx_file.is_file():
        print(f"[{model_name}] SKIP: no model.onnx")
        return None
    if not _checkpoint_present(config_path):
        print(f"[{model_name}] SKIP: checkpoint not present (needed for torch baseline)")
        return None

    torch_pred = LightweightQAOAPredictor(config_path=config_path, device="cpu", strict=True)
    onnx_pred = OnnxQAOAPredictor(config_path=config_path, device="cpu")

    torch_stats = _stats(_time_predict(torch_pred.predict, BENCH_OPS, repeats, warmup))
    onnx_stats = _stats(_time_predict(onnx_pred.predict, BENCH_OPS, repeats, warmup))
    max_diff = _max_angle_diff(torch_pred.predict, onnx_pred.predict, BENCH_OPS)
    speedup = torch_stats["median_ms"] / onnx_stats["median_ms"]

    print(
        f"[{model_name:26s}] "
        f"torch {torch_stats['median_ms']:7.3f} ms | "
        f"onnx {onnx_stats['median_ms']:7.3f} ms | "
        f"speedup {speedup:5.2f}x | "
        f"max angle diff {max_diff:.2e}"
    )
    return {
        "model": model_name,
        "torch": torch_stats,
        "onnx": onnx_stats,
        "speedup_median": speedup,
        "max_angle_diff": max_diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark torch vs ONNX inference.")
    ap.add_argument("--model", default="all", help="model_config name, or 'all'")
    ap.add_argument("--repeats", type=int, default=100, help="timed predictions per predictor")
    ap.add_argument("--warmup", type=int, default=10, help="untimed warmup predictions")
    ap.add_argument("--json", type=str, default=None, help="write results to this JSON path")
    args = ap.parse_args()

    if args.model == "all":
        names = sorted(
            p.name for p in MODEL_CONFIGS_DIR.iterdir() if (p / "model_config.json").exists()
        )
    else:
        names = [args.model]

    print(f"repeats={args.repeats} warmup={args.warmup} device=cpu ops={len(BENCH_OPS)}\n")
    results = [r for r in (bench_model(n, args.repeats, args.warmup) for n in names) if r]

    if results:
        geo = statistics.geometric_mean([r["speedup_median"] for r in results])
        print(f"\ngeometric-mean median speedup (onnx vs torch): {geo:.2f}x")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
