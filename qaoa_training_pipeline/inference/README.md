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

[`ai_inference.py`](ai_inference.py) is a `ProblemParamsProvider` and plugs
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

Users of the inference module should be aware of the following.

- The AI models consume a graph corresponding to the cost operator, see `feature_extractor.extract_np`.
Therefore, the cost operator is internally converted to a graph.
Each model declares in its bundle the pre-factor with which the cost operator is converted to a graph, see `hf_setup.json`.
- `AIInference` runs the exported `model.onnx` via `onnxruntime` + numpy. 
Torch checkpoints are note needed. 
To get the raw network output call
  `OnnxQAOAPredictor.predict_normalized(..., denormalize=False)`.
- Currently, all shipped bundles are pretrained on **max-cut** instances only, over the
graph families described in [arXiv:2610.23296](https://arxiv.org/abs/2610.23296). 
The cost operator must use the max-cut convention the models were trained on, 
`H = -0.5 * sum_e w_e Z_i Z_j` — i.e. `pre_factor=-0.5`, *not* the
`pre_factor=1.0` default of `graph_to_operator`, see `hf_setup.json`.
A different problem class — MIS, or any Hamiltonian that is not a weighted max-cut 
graph — is outside what they were trained for. 
`validate_cost_op` rejects the operators that cannot be a max-cut graph.
- The models provided in in [arXiv:2610.23296](https://arxiv.org/abs/2610.23296)
were trained directly on the QAOA angles corresponding to the setting where the `qaoa_angles_function` is the identity. 
- Internally, and before converting the cost operator to a graph, 
[`AIInterface`](ai_interface.py) normalizes the cost operator
by a constant.
This is crucial to get good angle inference over cost operators
with vary different coefficient scales.
This normalization is undone by [`AIInterface`](ai_interface.py) so that the returned angles
correspond to the angles for the unnormalized user provided cost operator.


### 2. Predictor

[`OnnxQAOAPredictor`](onnx_predictor.py) exposes the network step alone as `predict_normalized()`.
The cost operator reaches it already normalized, and un-rescaling the gammas is the caller's responsibility.
Crucially, some underlying models predict $\beta$ and $\gamma$ values
that have an additional scaling.
This scaling is different from the cost operator scaling discussed in [1. Entry point — `AIInference`](#1-entry-point--aiinference).
It is undone in by `OnnxQAOAPredictor`.
`OnnxQAOAPredictor` loads the `.onnx` model, runs it, and only feeds inputs the model declares.
Therefore, optional inputs like `edge_weights` do not break models that
  omit them.
Model artifacts are **ingested from the HuggingFace Hub** (`resolve_bundle`,
  via `model_registry.py`) and cached locally.

### 3. Feature extraction

[`feature_extractor.py`](feature_extractor.py) turns a `SparsePauliOp` into
model inputs via a numpy path (`extract_np`/`pack_features_np`) that reproduces
the shaping used during training, so the ONNX runtime produces the same numbers.
It produces scalar features (num_nodes, degrees, …), a graph (edges,
edge_weights), and reports back the `cost_op_scale` it was normalized at — the
normalization itself belongs to [`scaling.py`](scaling.py), see below.

Which operators are acceptable is [`conventions.py`](conventions.py):
`validate_cost_op` (rejects complex coefficients, local fields and terms above
second order) and `warn_on_unexpected_sign_convention`, which is checked
against the bundle's declared `prefactor`. That value comes from
`hf_setup.json` and nowhere else: this package holds no default convention, so
an entry without a `prefactor` is rejected by `validate_setup` rather than
silently treated as max-cut.
That last one has a blind spot by construction: a graph with weights of both
signs has mixed coefficients under either convention, so nothing warns and the
convention is the caller's to get right.

## Rescaling summary

Here we summarize the two independent normalizations discussed in the [Architecture](#architecture-three-layers).

| | normalizes | factor | applies to | owned by |
|---|---|---|---|---|
| **A** | the problem's scale | `cost_op_scale`, the operator's own RMS | $\gamma$ only | `AIInference` |
| **B** | the network's output units | `output_scale`, from the bundle config | $\beta$ and $\gamma$ | `OnnxQAOAPredictor` |

Since $\gamma$ multiplies the cost Hamiltonian scaling **A**, 
found in [`scaling.py`](scaling.py), is undone only for the $\gamma$ on the way out. 
The code flow is thus

1. **Scale (A, in).** `normalize_cost_op` divides the operator by
   `rescaling_factor`, the RMS of its per-Pauli-order coefficients.
2. **Predict (B).** `predict_normalized` builds the features (`cost_op_scale` is
   handed in, since some bundles consume it as a feature), runs the ONNX
   session, and multiplies the output by `output_scale` to get radians.
3. **Unscale (A, out).** `undo_cost_op_normalization` divides the **gammas
   only** by `cost_op_scale`, mapping them onto the operator that was passed. 

Therefore, `AIInference` is the way
to get angles for a cost operator.
Using the `OnnxQAOAPredictor` directly means running
steps 1 and 3 yourself.
`denormalize_output` (default `True`) in the config governs step 2's output
scaling, and `predict_normalized(..., denormalize=False)` returns the raw network output.

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
touches the network; for an air-gapped run, load the bundles you need once from
a networked run first. The published repos are public, so no token is needed; a
bundle with no pinned revision in the setup file is not published yet and raises
a message saying so.

## One-line summary

Given a cost operator, this package extracts graph/scalar features, runs a
pre-trained GNN/transformer exported to ONNX (torch-free), and returns
denormalized β/γ angles — dropping into the pipeline as a standard
`ProblemParamsProvider`. Scale-invariance is handled by normalizing the operator
on input and un-rescaling the gammas on output.
