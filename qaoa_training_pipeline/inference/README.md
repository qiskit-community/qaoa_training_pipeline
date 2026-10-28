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
import networkx as nx
from qaoa_training_pipeline.inference import AIInference
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator

cost_op = graph_to_operator(nx.random_regular_graph(3, 10, seed=7), pre_factor=-0.5)
inference = AIInference(model="gcn/p3")      # downloaded from the Hub, then cached
result = inference.provide_params(cost_op)   # ParamResult of [beta_1..beta_p, gamma_1..gamma_p]
```

- **The cost operator must use the max-cut convention the models were trained
  on**, `H = -0.5 * sum_e w_e Z_i Z_j` — i.e. `pre_factor=-0.5`, *not* the
  `pre_factor=1.0` default of `graph_to_operator`. The models consume edge
  weights, which `extract_np` recovers by dividing the coefficients by that
  pre-factor (`MAX_CUT_PRE_FACTOR`). What has to match is the *sign*: the
  normalization described under *Rescaling of the cost operator* below divides
  the pre-factor's magnitude back out, so `-2.0` is the same normalized problem
  as `-0.5` and gets the gammas that are right for the operator it produced.
  The opposite sign does not divide out — it normalizes to the negated operator
  — so it gives finite but wrong gammas rather than an error. The one
  recognisable case, all-positive quadratic coefficients, raises a
  `UserWarning`; a graph with weights of both signs has mixed coefficients
  under either convention, so nothing warns and the convention is yours to get
  right.

- Runs the exported `model.onnx` via `onnxruntime` + numpy — no torch, no
  checkpoint needed.
- An optional `qaoa_angles_function` to post-process the predicted angles — a
  scaling, a change of basis. It is the framework's own mechanism and it
  serializes, so a `to_config()` / `from_config()` round trip keeps it. For the
  raw network output instead, call
  `OnnxQAOAPredictor.predict_normalized(..., denormalize=False)`.

### 2. Predictor

- [`OnnxQAOAPredictor`](onnx_predictor.py) exposes `predict_normalized()` with
  `output_dim` validation. It is the network step alone: the operator reaches it
  already normalized, and un-rescaling the gammas is the caller's step, so the
  scaling chain exists in one place only (see below).
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
edge_weights), and reports back the `rescale_a` it was normalized at — the
normalization itself belongs to [`scaling.py`](scaling.py), see below.

## Rescaling of the cost operator

Scale-invariance is a three-step chain, and `AIInference.provide_params` runs it
in the open:

```python
normalized_cost_op, rescale_a = normalize_cost_op(cost_op)          # 1. scale
raw_angles = predictor.predict_normalized(normalized_cost_op, rescale_a)  # 2. predict
angles = undo_gamma_rescale(raw_angles, p, rescale_a)               # 3. unscale
```

1. **Scale.** `normalize_cost_op` divides the operator by `rescaling_factor`,
   the RMS of its per-Pauli-order coefficients, so the network always sees the
   same scale-normalized problem.
2. **Predict.** `predict_normalized` builds the features from that operator
   (`rescale_a` is handed in, since some bundles consume it as a feature), runs
   the ONNX session, and denormalizes the output by the bundle's `output_scale`
   (`× π/2` by default).
3. **Unscale.** `undo_gamma_rescale` divides the **gammas only** by `rescale_a`,
   mapping them onto the operator that was actually passed. Betas are untouched.

Both scaling steps live in [`scaling.py`](scaling.py), which imports neither
`onnxruntime` nor a bundle, so the chain can be read and tested on its own
(`tests/inference/test_scaling.py`). They reproduce the scaling applied during
training, which is what makes ONNX inference match the trained model.

Step 2 is the only thing `OnnxQAOAPredictor` does, so `AIInference` is the way
to get angles for a cost operator; using the predictor directly means running
steps 1 and 3 yourself, as the snippet above does.
`denormalize_output` (default `True`) in the config governs step 2's output
scaling, and `predict_normalized(..., denormalize=False)` returns the raw
network output.

The invariance holds for a *positive* rescaling of the operator: `rescaling_factor`
is an RMS and so is sign-blind, and negating the cost operator describes a
different problem rather than the same one at another scale.

## Model zoo

Seven GNN/transformer architectures are registered as exported ONNX bundles,
listed here as *display name* (`bundle key architecture`): GCN (`gcn`), GIN
(`graph_isomorphism_network`), GNN (`graph_neural_network`), graph transformer
(`graph_transformer`), edge transformer (`edge_transformer`), diffusion
transformer (`diffusion_transformer`) and MLP (`mlp`) — each at depths
p = 1…5, for 35 bundle keys. Only the parenthesised name is addressable: the
repo ids abbreviate (`gin`, `gnn`, `gt`, `et`, `dit`) and those spellings are
rejected by `parse_bundle_key`.

All 35 are published in the
[model collection](https://huggingface.co/collections/ibm-research/angles-for-quantum-approximate-optimization)
and pinned to a commit in the setup file, so they download without a token.
`model_registry.published_bundles()` is the authoritative list of what can be
downloaded; an entry with `revision: null` is not published and raises a "not
published yet" error naming itself.
[`onnx_inputs.py`](onnx_inputs.py) holds a registry (`INPUT_BUILDERS`)
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
`model_registry.prefetch_bundles()`. The published repos are public, so no
token is needed; a bundle with no pinned revision in the setup file is not
published yet and raises a message saying so.

## One-line summary

Given a cost operator, this package extracts graph/scalar features, runs a
pre-trained GNN/transformer exported to ONNX (torch-free), and returns
denormalized β/γ angles — dropping into the pipeline as a standard
`ProblemParamsProvider`. Scale-invariance is handled by normalizing the operator
on input and un-rescaling the gammas on output.
