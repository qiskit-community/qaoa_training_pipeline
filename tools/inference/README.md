# AI inference tooling (torch / export path)

These scripts regenerate the artifacts that ship with the torch-free ONNX
inference runtime in `qaoa_training_pipeline/inference/`. They are **not**
imported at runtime and require the heavier `inference-torch` extra plus the
training checkpoints referenced by each `model_config.json`:

```bash
pip install -e ".[inference-torch]"    # torch + torch_geometric
```

Run from the repository root:

| Script | Purpose |
|---|---|
| `export_onnx.py` | Export a trained `.ckpt` to `model.onnx` (`--model all --check-parity`). |
| `gen_baselines.py` | Freeze torch predictions to `test/inference/baselines/<model>.json`. |
| `bench_ops.py` | Shared deterministic cost operators (imported by the others). |
| `bench_onnx_vs_torch.py` | Latency + correctness benchmark (`--json <path>`). |
| `compare_quality.py` | QAOA solution-quality equivalence (torch vs ONNX). |

Regeneration workflow on retrain / model change:

```bash
python tools/inference/export_onnx.py --model all --check-parity
python tools/inference/gen_baselines.py
python -m unittest discover -s test/inference -t .
python tools/inference/bench_onnx_vs_torch.py
python tools/inference/compare_quality.py
```
