# AI Parameter Inference

This package predicts QAOA angles (β, γ) **directly from a problem's cost
operator using pre-trained ML models**, instead of running a classical
optimizer. It acts as a learned warm-start / replacement for angle
optimization and plugs into the pipeline as a standard
`ProblemParamsProvider`.

## Architecture (three layers)

### 1. Entry point — `AIInference`

[`ai_inference.py`](ai_inference.py) is a `ProblemParamsProvider`, so it plugs
into the existing pipeline exactly like any other angle provider. You give it a
`config_path`; it calls `provide_params(cost_op)` and returns a `ParamResult` of
angles.

- **Two swappable backends** selected by `backend=`:
  - `"onnx"` (default) — torch-free, runs via `onnxruntime` + numpy. No torch,
    no checkpoint needed.
  - `"torch"` — the reference PyTorch predictor (needs torch + torch_geometric
    and the original checkpoint). Imported *lazily* so the default path never
    touches torch.
- An optional `rescale` hook that, when supplied, **replaces** the config's
  built-in denormalization (it passes `denormalize=False` down and applies the
  user's function to the raw output).

### 2. Predictors — the two backends share one contract

- [`OnnxQAOAPredictor`](onnx_predictor.py) and
  [`LightweightQAOAPredictor`](torch_backend/lightweight_predictor.py) have
  matching `predict()` signatures and `output_dim` validation, so they are
  interchangeable.
- The ONNX predictor loads the `.onnx` graph, runs it, and only feeds inputs
  the graph actually declares — so optional inputs like `edge_weights` do not
  break models that omit them.
- Model artifacts are **local-first with a lazy HuggingFace download fallback**
  (`ensure_onnx_local`, via `model_registry.py`).

### 3. Feature extraction

[`feature_extractor.py`](feature_extractor.py) turns a `SparsePauliOp` into
model inputs. It has **two mirrored paths** — a torch path
(`extract`/`pack_features`) and a torch-free numpy path
(`extract_np`/`pack_features_np`) — deliberately kept identical so the ONNX
runtime produces the same numbers as training. It produces scalar features
(num_nodes, degrees, …), a graph (edges, edge_weights), and the `rescale_a`
factor.

## Rescaling of the cost operator

Scale-invariance is handled in two coupled places:

- **On input**, the operator is normalized: `cost_op / rescaling_factor(cost_op)`
  where the factor is the RMS of per-Pauli-order coefficients
  (`datamodule_utils.rescaling_factor`). The model always sees a scale-normalized
  problem.
- **On output**, predictions are denormalized: `× π/2`, then the **gammas only**
  are divided back by `rescale_a` to map onto the original operator's scale.
  Betas are left untouched.
- The numpy/ONNX path replicates the torch training path exactly
  (`rescaling_factor` / `angles_to_target` / `undo_gamma_rescale`), which is what
  guarantees ONNX inference matches the trained model.

This is gated by `denormalize_output` (default `True`) in the config.

## Model zoo

Multiple GNN/transformer architectures are trained under
[`torch_backend/ml_models/`](torch_backend/ml_models/): GCN, GIN, graph
transformer, edge transformer, a DDPM transformer, plus MLP — each with
per-depth (p = 1…4) configs. [`onnx_inputs.py`](onnx_inputs.py) holds a registry
(`numpy_input_builders`) mapping model type → how to build its numpy feed.

## Supporting tooling

[`tools/inference/`](../../tools/inference/) provides the export + validation
harness: `export_onnx.py` (checkpoint → onnx), `bench_onnx_vs_torch.py`,
`compare_quality.py`, and `gen_baselines.py` — i.e. the tooling proving ONNX
matches torch in quality and speed.

## One-line summary

Given a cost operator, this package extracts graph/scalar features, runs a
pre-trained GNN/transformer exported to ONNX (torch-free by default, with a
torch reference backend), and returns denormalized β/γ angles — dropping into
the pipeline as a standard `ProblemParamsProvider`. Scale-invariance is handled
by normalizing the operator on input and un-rescaling the gammas on output.
