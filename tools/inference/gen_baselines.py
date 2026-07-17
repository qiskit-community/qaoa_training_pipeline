#!/usr/bin/env python
"""Freeze torch predictions to JSON golden baselines (one-time / on retrain).

For every model architecture and every operator in ``scripts/bench_ops.py``,
this runs the *torch* ``LightweightQAOAPredictor`` and records the predicted
angles. The resulting ``tests/baselines/<model>.json`` files let the ONNX suite
regression-test against a fixed reference WITHOUT torch or the checkpoints being
present — which is the whole point of the port.

Regenerate only when the checkpoints or the model code intentionally change:

    python scripts/gen_baselines.py            # all models present on disk
    python scripts/gen_baselines.py --model gcn
"""

from __future__ import annotations

import argparse
import json
import sys
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
BASELINE_DIR = REPO_ROOT / "test" / "inference" / "baselines"


def _checkpoint_present(config_path: Path) -> bool:
    cfg = load_config(config_path)
    ckpt = cfg.get("checkpoint")
    return bool(ckpt) and (config_path.parent / ckpt).resolve().is_file()


def generate(model_name: str) -> Path | None:
    from qaoa_training_pipeline.inference.torch_backend.lightweight_predictor import (
        LightweightQAOAPredictor,
    )

    config_path = MODEL_CONFIGS_DIR / model_name / "model_config.json"
    if not _checkpoint_present(config_path):
        print(f"[{model_name}] SKIP: checkpoint not present")
        return None

    predictor = LightweightQAOAPredictor(config_path=config_path, device="cpu", strict=True)
    cases = {name: predictor.predict(op) for name, op in BENCH_OPS.items()}

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINE_DIR / f"{model_name}.json"
    payload = {
        "model": model_name,
        "predictor": "LightweightQAOAPredictor",
        "output_dim": predictor.output_dim,
        "cases": cases,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[{model_name}] wrote {out_path.relative_to(REPO_ROOT)} ({len(cases)} cases)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate torch prediction baselines.")
    ap.add_argument("--model", default="all", help="model_config name, or 'all'")
    args = ap.parse_args()

    if args.model == "all":
        names = sorted(
            p.name for p in MODEL_CONFIGS_DIR.iterdir() if (p / "model_config.json").exists()
        )
    else:
        names = [args.model]

    for name in names:
        generate(name)


if __name__ == "__main__":
    main()
