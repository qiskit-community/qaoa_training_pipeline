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
into the existing pipeline exactly like any other angle provider. You give it a
`config_path`; it calls `provide_params(cost_op)` and returns a `ParamResult` of
angles.

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
- Model artifacts are **local-first with a lazy HuggingFace download fallback**
  (`ensure_onnx_local`, via `model_registry.py`).

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

Multiple GNN/transformer architectures ship as exported ONNX bundles: GCN, GIN,
graph transformer, edge transformer, a DDPM transformer, plus MLP — each with
per-depth (p = 1…4) configs under `model_configs/<model>/p<p>/`.
[`onnx_inputs.py`](onnx_inputs.py) holds a registry (`numpy_input_builders`)
mapping model type → how to build its numpy feed.

## Supporting tooling

[`tools/inference/`](../../tools/inference/) provides torch-free helpers:
`model_keys.py` (bundle discovery), `bench_ops.py` (deterministic cost
operators), `hf_manifest.py` and `upload_to_hf.py` (HuggingFace weight manifest
and upload). The frozen predictions in `test/inference/baselines/` guard against
regressions in the ONNX runtime.

## One-line summary

Given a cost operator, this package extracts graph/scalar features, runs a
pre-trained GNN/transformer exported to ONNX (torch-free), and returns
denormalized β/γ angles — dropping into the pipeline as a standard
`ProblemParamsProvider`. Scale-invariance is handled by normalizing the operator
on input and un-rescaling the gammas on output.
