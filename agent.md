# Agent Work Log

Repository: quantum_ai_parameter_prediction (inference pipeline)

## History (condensed)

Earlier sessions (2026-07-07 → 2026-07-15) brought the pipeline from blocked to
fully working. In brief:

- **Feature-extractor interface** harmonized with the QAOA training pipeline
  (`AIFeatureExtractor` inherits `BaseFeatureExtractor`; `features()`,
  `__call__`, `to_config`/`from_config`, extractor registry).
- **Environment / dependencies**: local `.venv` set up;
  `qaoa_training_pipeline` re-pinned to the framework-capable revision
  `8296bae` (public `github.com/qiskit-community/qaoa_training_pipeline`);
  `git-lfs` installed for checkpoints (pulled via SSH). `requirements.txt`
  authored and clean-install verified in an isolated venv.
- **Training pipeline unblocked E2E**: fixed a syntax/merge break in
  `training/qaoa_model.py`, made `graph_dir` optional, made `PPEvaluator`
  (Julia) lazy so p=1 needs no Julia, vendored a missing package data file.
  Dataset located at `git@github.ibm.com:DEG/2026_AI_QAOA_graphs.git` and
  placed at `../2026_AI_QAOA_graphs`. GNN p=1 training runs and learns.
- **Inference E2E PASS** for all architectures against the
  `multi_topology_checkpoints_new` p=1 checkpoints (the historical GNN
  "architecture drift" blocker no longer reproduces against these).
- **Test suite added** (`pytest.ini` + `tests/`): feature-extractor unit tests
  + E2E inference tests across all 7 architectures. `torch_geometric==2.8.0`
  added for the graph builders.

Full blow-by-blow of those sessions is in git history; the current state is
below.

---

# Status — ONNX Inference Port

_Last updated: 2026-07-17_

## Summary

The torch-free ONNX inference path is complete for **all 7 QAOA model
architectures**. Runtime (prediction) no longer requires torch or
torch_geometric — only `onnxruntime` + numpy. torch is now lazy-imported and
used only in the training/export path.

- Branch: **`onnx-port-graph-models-and-perf`** (based on
  `add-inference-pipeline-and-tests`), pushed to origin.
- **PR #2 → `main`:**
  https://github.ibm.com/IBM-Research-AI/-quantum_ai_parameter_inference/pull/2
  (targets `main` because the base branch `add-inference-pipeline-and-tests` is
  not on the remote; diff therefore includes the 2 pre-existing pipeline commits
  plus the 3 ONNX-port commits).

## Model status

| Model | ONNX exported | Angle parity vs torch | QAOA approx-ratio (torch → onnx) | Baseline test |
|---|---|---|---|---|
| mlp (agg_transformer) | ✅ | 5.96e-08 | 0.2674 → 0.2674 | ✅ |
| diffusion_transformer | ✅ | 0.00 | 0.2609 → 0.2609 | ✅ |
| edge_transformer | ✅ | 8.94e-08 | 0.2568 → 0.2568 | ✅ |
| gcn | ✅ (fixed) | 0.00 | 0.2683 → 0.2683 | ✅ |
| graph_isomorphism_network | ✅ | 1.49e-08 | 0.2689 → 0.2689 | ✅ |
| graph_neural_network | ✅ | 0.00 | 0.2620 → 0.2620 | ✅ |
| graph_transformer | ✅ (added) | ≤ 6e-4 | 0.2633 → 0.2633 | ✅ |

All exports use clean external-data sidecars: `model.onnx` + `model.onnx.data`.

**Update (best-seed, p=1..4):** the shipped set has since been regenerated from
the best-test-seed checkpoint per `(architecture, p)` and now covers QAOA depths
`p = 1, 2, 3, 4` for all 7 architectures (28 bundles) under
`model_configs/<model>/p<p>/` (MLP dir is `mlp/`). All 28 pass ONNX-vs-torch
parity (max abs diff ≤ 1.2e-07; graph_transformer under its looser Laplacian-PE
tolerance) and the regenerated `<model>_p<p>.json` baselines. The numbers in the
tables above are the original p=1 / seed-42 measurements.

## ML performance (solution quality) — preserved

Beyond numeric angle parity, the ONNX angles were evaluated on the actual QAOA
objective (exact expected cost via the repo's `StatevectorEvaluator`) and scored
as an approximation ratio against the brute-force optimum
(`scripts/compare_quality.py`).

- Approximation ratios are **identical to 4+ decimals** for every model.
- **Worst approximation-ratio gap across all 7 models: 5.57e-05** (graph_transformer;
  raw cost gap 6.68e-04). Everything else ≤ 2.75e-07.
- The tiny graph_transformer gap traces to the Laplacian-PE eigensolver sign
  ambiguity — far below anything that would change which solution QAOA finds.

Conclusion: **the ONNX port gives the same ML-based performance as torch.**

## Performance (CPU, 100 reps, `scripts/bench_onnx_vs_torch.py`)

| Model | torch | onnx | speedup |
|---|---|---|---|
| graph_transformer | 3.10 ms | 0.61 ms | **5.06x** |
| diffusion_transformer | 1.77 ms | 0.50 ms | 3.52x |
| gcn | 1.07 ms | 0.32 ms | 3.29x |
| edge_transformer | 1.57 ms | 0.52 ms | 3.01x |
| graph_isomorphism_network | 0.97 ms | 0.36 ms | 2.74x |
| graph_neural_network | 0.79 ms | 0.33 ms | 2.39x |
| mlp (agg_transformer) | 0.39 ms | 0.25 ms | 1.57x |

**Geo-mean speedup: 2.92x.** Max angle diff ≤ 5.8e-4 across all architectures.
(Numbers are machine-specific; regenerate locally.)

## Tests — 56 passing

- torch suite (`tests/test_inference.py`): 29 tests (unchanged).
- ONNX suite (`tests/test_onnx_inference.py`): 27 tests
  - 7× loads-and-predicts (all models, needs only `model.onnx`)
  - 2×3 behavioral (determinism, reacts-to-input, raw-vs-denorm)
  - 7× `test_onnx_matches_torch` (parity, needs checkpoint)
  - 7× `test_onnx_matches_baseline` (regression vs frozen golden values —
    **needs neither torch nor checkpoint**)

Run: `.venv/bin/python -m pytest tests/ -q`

## Key files

Runtime (torch-free):
- `inference_pipeline/inference_utils/onnx_predictor.py` — `OnnxQAOAPredictor`,
  drop-in analogue of `LightweightQAOAPredictor`.
- `inference_pipeline/inference_utils/onnx_inputs.py` — numpy input builders
  per model (incl. numpy Laplacian PE for graph_transformer).
- `inference_pipeline/inference_utils/config_io.py` — torch-free config load.

Export / tooling (torch path):
- `scripts/export_onnx.py` — one-time export with fixed-signature wrappers +
  `--check-parity`.
- `scripts/gen_baselines.py` — freeze torch predictions to
  `tests/baselines/<model>.json`.
- `scripts/bench_ops.py` — shared deterministic cost operators (8 topologies).
- `scripts/bench_onnx_vs_torch.py` — latency + correctness benchmark.
- `scripts/compare_quality.py` — QAOA solution-quality equivalence check
  (torch vs ONNX approximation ratio via StatevectorEvaluator).

## How the two hard models were solved

- **gcn**: the export dynamic-axes block only declared `edge_index`, so
  `torch.export` specialized the undeclared `edge_weight`/`node_x` to the
  example sizes and conflicted (GraphConv requires matching edge counts).
  Fix: declare every node-/edge-count axis with the same symbol.
- **graph_transformer**: two ONNX-hostile ops. (1) `torch.linalg.eigh` for the
  Laplacian PE — precomputed in numpy and passed as an input (`pos_enc`). Sign
  ambiguity is tolerated because the model was trained with random sign-flip
  augmentation, hence the looser 2e-3 test tolerance. (2)
  `edge_mask.fill_diagonal_` specialized `num_nodes` to a constant — the export
  wrapper re-expresses it as an `arange` index-put and reuses the trained
  submodules.

## Regeneration workflow (on retrain / model change)

1. `.venv/bin/python scripts/export_onnx.py --model all --check-parity`
2. `.venv/bin/python scripts/gen_baselines.py`
3. `.venv/bin/python -m pytest tests/ -q`
4. `.venv/bin/python scripts/bench_onnx_vs_torch.py`
5. `.venv/bin/python scripts/compare_quality.py`

## Commits (on branch, ahead of `add-inference-pipeline-and-tests`)

- `6bcd4df` Complete ONNX port: gcn + graph_transformer, clean external-data
  sidecars, tests
- `5b733e4` Add torch baselines, baseline regression test, and torch-vs-ONNX
  benchmark
- `69b9df9` Add QAOA solution-quality comparison + status doc

## Open items

- **PR #2 targets `main`, not `add-inference-pipeline-and-tests`** (base not on
  remote). If the pipeline commits land via their own PR first, rebase this
  branch afterward so the diff shows only the ONNX work.
- `tests/baselines/bench_results.json` is gitignored (machine-specific latency
  output from `--json`).
