# AI Parameter Inference

This package predicts QAOA angles (β, γ) **directly from a problem's cost
operator using pre-trained ML models**, instead of running a classical
optimizer. It acts as a learned warm-start / replacement for angle
optimization and plugs into the pipeline as a standard
`ProblemParamsProvider`.

Inference is **torch-free**: it runs an exported ONNX graph with `onnxruntime`
and numpy, and needs neither torch nor the original checkpoint.

## Architecture (three layers)

### 1. Entry point — `AIInference`

[`ai_inference.py`](ai_inference.py) is a `ProblemParamsProvider`, so it plugs
into the existing pipeline exactly like any other angle provider. You name a
model; it calls `provide_params(cost_op)` and returns a `ParamResult` of angles.

```python
from qaoa_training_pipeline.inference import AIInference

inference = AIInference(model="gcn/p3")      # downloaded from the Hub, then cached
result = inference.provide_params(cost_op)   # ParamResult of [beta_1..beta_p, gamma_1..gamma_p]
```

- Runs the exported `model.onnx` via `onnxruntime` + numpy — no torch, no
  checkpoint needed.
- An optional `rescale` hook that, when supplied, **replaces** the config's
  built-in denormalization (it passes `denormalize=False` down and applies the
  user's function to the raw output).

### 2. Predictor

- [`OnnxQAOAPredictor`](onnx_predictor.py) exposes `predict()` with `output_dim`
  validation.
- It loads the `.onnx` graph, runs it, and only feeds inputs the graph actually
  declares — so optional inputs like `edge_weights` do not break models that
  omit them.
- Model artifacts are **ingested from the HuggingFace Hub** (`resolve_bundle`,
  via `model_registry.py`) and cached locally.

### 3. Feature extraction

[`feature_extractor.py`](feature_extractor.py) turns a `SparsePauliOp` into
model inputs via a numpy path (`extract_np`/`pack_features_np`) that reproduces
the shaping used during training, so the ONNX runtime produces the same numbers.
It produces scalar features (num_nodes, degrees, …), a graph (edges,
edge_weights), and the `rescale_a` factor.

## Rescaling of the cost operator

Scale-invariance is handled in two coupled places:

- **On input**, the operator is normalized: `cost_op / rescaling_factor(cost_op)`
  where the factor is the RMS of per-Pauli-order coefficients
  (`datamodule_utils.rescaling_factor`). The model always sees a scale-normalized
  problem.
- **On output**, predictions are denormalized: `× π/2`, then the **gammas only**
  are divided back by `rescale_a` to map onto the original operator's scale.
  Betas are left untouched.
- `rescaling_factor` (input) and the gamma un-rescale (output) reproduce the
  scaling applied during training, which is what guarantees ONNX inference
  matches the trained model.

This is gated by `denormalize_output` (default `True`) in the config.

## Model zoo

Seven GNN/transformer architectures are released as exported ONNX bundles: GCN,
GIN, GNN, graph transformer, edge transformer, a DDPM transformer, plus MLP —
each at depths p = 1…5, for 35 bundles.
[`onnx_inputs.py`](onnx_inputs.py) holds a registry (`numpy_input_builders`)
mapping model type → how to build its numpy feed.

### How models are ingested

No model artifacts ship with this package. Each bundle is a **self-contained
HuggingFace repo** holding `model_config.json`, `model.onnx` and
`model.onnx.data` at its root. [`hf_setup.json`](huggingface/hf_setup.json) maps a
**bundle key** — `<model>/p<p>`, e.g. `gcn/p3`, the stable public identifier —
to that repo and the commit it is pinned to, so a download is reproducible.

The Hub is the only source of models, and a bundle key is the only way to
address one — there is no local-path loading and no raw-repo-id argument:

```python
AIInference(model="gcn/p3")              # or, lower level:
OnnxQAOAPredictor.from_bundle("gcn/p3")
```

Every model therefore resolves the same way: key → setup → repo at a pinned
revision. That is what makes a key a verified name (an unknown one fails, it
does not resolve to the wrong repo), a run reproducible, and `to_config()`
portable between machines.

To use a bundle outside the shipped zoo — a private export, a retrain — copy
`hf_setup.json`, give your bundle one of the verified keys, and point
`$QAOA_HF_SETUP` at your copy:

```bash
export QAOA_HF_SETUP=/path/to/my_hf_setup.json
```

The setup file in use is `$QAOA_HF_SETUP` when set, and the packaged
`hf_setup.json` otherwise — nothing else is picked up implicitly, so a
checkout always resolves models the same way.

`snapshot_download` caches under `~/.cache/huggingface`, so only the first use
touches the network. For an air-gapped run, warm the cache first with
`model_registry.prefetch_bundles()`. The repos are private while the models are
unreleased, so downloads need a token (`hf auth login`, or `HF_TOKEN`); a bundle
with no pinned revision in the setup file is not published yet and raises a
message saying so.

## Supporting tooling

[`tools/inference/`](../../tools/inference/) provides torch-free helpers:
`model_keys.py` (bundle discovery), `bench_ops.py` (deterministic cost
operators), `hf_setup.py` and `upload_to_hf.py` (bundle upload and setup
revision pinning). The frozen predictions in `test/inference/baselines/` guard
against regressions in the ONNX runtime.

## One-line summary

Given a cost operator, this package extracts graph/scalar features, runs a
pre-trained GNN/transformer exported to ONNX (torch-free), and returns
denormalized β/γ angles — dropping into the pipeline as a standard
`ProblemParamsProvider`. Scale-invariance is handled by normalizing the operator
on input and un-rescaling the gammas on output.
