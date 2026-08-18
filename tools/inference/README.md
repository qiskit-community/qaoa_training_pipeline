# AI inference tooling

Helper scripts for the torch-free ONNX inference runtime in
`qaoa_training_pipeline/inference/`. They are **not** imported at runtime and
need only the `inference` extra:

```bash
pip install -e ".[inference]"    # onnxruntime + huggingface_hub; no torch
```

Run from the repository root:

| Script | Purpose |
|---|---|
| `model_keys.py` | Shared discovery of `model_config` bundles under the nested layout. |
| `bench_ops.py` | Shared deterministic cost operators (fixed, no RNG). |
| `hf_manifest.py` | Generate / refresh the HuggingFace weight manifest (`model_configs/hf_manifest.json`). |
| `upload_to_hf.py` | Upload the ONNX weight bundles to the HuggingFace Hub and pin the manifest. |

The large `model.onnx` / `model.onnx.data` weights live on the HuggingFace Hub
and are lazily downloaded on first use (see
`qaoa_training_pipeline/inference/model_registry.py`); only the tiny
`model_config.json` files are kept in git.

Bundles live under `qaoa_training_pipeline/inference/model_configs/<model>/p<p>/`.
A **bundle key** is the relative directory `<model>/p<p>` (e.g. `gcn/p3`). The
MLP architecture ships in `mlp/` (its `model_type` inside the config is still
`agg_transformer`). Seven architectures × `p = 1..4` = 28 bundles, each exported
from the best-test-seed checkpoint.

> The torch training/export path (checkpoint → ONNX export, torch-vs-ONNX
> parity/benchmark) has been removed; this package is ONNX-only. The exported
> `.onnx` bundles and the frozen baselines in `test/inference/baselines/` remain
> the source of truth.
