# Session status — QAOA p=5 ONNX zoo (crash-recovery dump)

> Written 2026-09-04. This file is a complete brain-dump of the p=5 work so it
> survives a crash / disk failure. Everything below is either already committed
> or explained well enough to redo. Branch: `vibing`. Fork/backup remote:
> `origin` = `git@github.com:romeokienzler/qaoa_training_pipeline.git`
> (`upstream` = qiskit-community, do NOT push there).

## TL;DR — what got done

Task (from `P5_ONNX_HANDOFF.md`): add QAOA depth **p=5** (7 architectures) to the
torch-free ONNX model zoo, matching how p=1..4 shipped. **DONE and committed** as
`5748713` "Add QAOA p=5 to the ONNX model zoo (7 archs, 28→35 bundles)".

Zoo grew **28 → 35 bundles** (7 archs × p1..5). All 7 p5 bundles pass the
torch-free baseline regression.

## The one hard problem and how it was solved

`feature_normalization` (per-depth z-score stats for the 5 graph features) is
**not** in the checkpoints — they only carry remote cluster paths
(`/dccstor/eevdata/cmahlasi/2026_AI_QAOA_graphs/...`, not mounted locally). The
handoff explicitly warned: do NOT copy p4 stats.

**Source of truth found:** the private repo
`git@github.ibm.com:IBM-Research-AI/quantum_ai_parameter_prediction_checkpoints.git`,
local clone at `~/gitco/quantum_ai_parameter_prediction_checkpoints`, branch
**`results/visuals`**, file
`multi_topology_configs_final/p_5/graph_isomorphism_network/graph_isomorphism_network_p5_alltopo_seed42.yaml`
key `norm_stats` (format `feature: [mean, std]`).

Retrieve again with:
```bash
cd ~/gitco/quantum_ai_parameter_prediction_checkpoints
git fetch origin results/visuals
git show FETCH_HEAD:multi_topology_configs_final/p_5/graph_isomorphism_network/graph_isomorphism_network_p5_alltopo_seed42.yaml
```

**Key facts verified:**
- `norm_stats` are **dataset-level** — identical across all archs at a given
  depth (confirmed on p4: gin == agg_transformer == gcn). So the single gin p5
  yaml applies to all 7 archs.
- p5 is a **distinct data split** from p1..4. Clusters observed:
  - p1 ≡ p2: num_nodes mean 60.52 / std 30.66
  - p3 ≈ p4: num_nodes mean 62.19 / std 32.22
  - **p5: num_nodes mean 61.38 / std 32.33** — matches neither cluster.
  Copying p4 would have silently corrupted predictions.

**The p5 norm_stats actually written into all 7 configs:**
| feature | mean | std |
|---|---|---|
| num_nodes | 61.381141662597656 | 32.334877014160156 |
| num_edges | 110.3201675415039 | 90.3392333984375 |
| edges_per_node | 1.7744890451431274 | 1.0926597118377686 |
| mean_degree | 3.548978090286255 | 2.185319423675537 |
| std_degree | 0.5941612720489502 | 0.6956432461738586 |

Note: `feature_normalization` is applied at **runtime in numpy before** the ONNX
graph — it is NOT baked into the graph. So export + `--check-parity` pass
regardless of whether the stats are correct; wrong stats only corrupt runtime
predictions. The parity gate does NOT protect against bad stats.

## p5 checkpoint facts

Source: `~/gitco/quantum_ai_parameter_prediction_checkpoints_open_source/p_5/<arch>/<file>.ckpt`
(repo `quantum_ai_parameter_prediction_checkpoints_open_source`). All 7 present.
All: `input_dim=5`, `output_dim=10` (=2·p), `embed_dim=64`, one seed each, no
embedded norm stats. `mlp` checkpoint `model_type` is `"mlp"` → config uses
`agg_transformer`.

| dir | file |
|---|---|
| mlp | mlp-p5-20260721_144303-81-0.0005.ckpt |
| `edge_transformer ` (⚠️ trailing space in dir name) | edge_transformer-p5-20260721_145654-87-0.0004.ckpt |
| diffusion_transformer | diffusion_transformer-p5-20260721_144936-91-0.0004.ckpt |
| gcn | gcn-p5-20260721_145654-52-0.0006.ckpt |
| graph_neural_network | graph_neural_network-p5-20260721_150701-92-0.0002.ckpt |
| graph_isomorphism_network | graph_isomorphism_network-p5-20260721_150017-90-0.0001.ckpt |
| graph_transformer | graph_transformer-p5-20260721_151011-49-0.0006.ckpt |

Per-arch `model_init` dims cloned from the p4 configs (unchanged except
`output_dim` 8→10): edge/diffusion add `edge_embed_dim=32, dim_feedforward=256`
(diffusion also `timesteps=100`); gnn/gin add `edge_dim=1, node_input_dim=144`;
gcn uses `hidden_dim=64`; graph_transformer adds `num_heads=4, ff_dim=128,
pos_enc_dim=8, node_feature_dim=2`.

## IMPORTANT correction to the handoff: HF publish never happened

The handoff says "big weights live on the HF Hub, not git." **This is false for
the current repo state:**
- All 28 p1..4 `model.onnx` + `.onnx.data` are **committed directly in git**
  (commit `17aae7d`). No `.gitignore` excludes them.
- `hf_setup.json` `repo_id` is still `PLACEHOLDER_ORG/qaoa-training-pipeline-models`
  — the HF upload was never done.
- `huggingface_hub` is NOT installed in `.venv-export`; no HF auth configured.
- Runtime is **local-first** (`model_registry.py`), so it uses local weights and
  never needs to download; placeholder repo_id is fine.

**Decision (confirmed by user):** ship p5 the same way p1..4 actually shipped —
commit weights in git, keep the placeholder repo_id. NOT published to HF.
A real HF publish, if ever wanted, is a separate task that would also re-home
p1..4.

## Exactly what commit 5748713 contains (30 files)

- 7 × `qaoa_training_pipeline/inference/model_configs/<arch>/p5/model_config.json`
- 7 × `.../<arch>/p5/model.onnx` + 7 × `.../<arch>/p5/model.onnx.data`
- `qaoa_training_pipeline/inference/model_configs/hf_setup.json` (28→35 bundles,
  repo_id unchanged = placeholder)
- 7 × `test/inference/baselines/<arch>_p5.json` (8 cases each)
- `test/inference/test_onnx_inference.py` (`P_VALUES = [1,2,3,4]` → `[1,2,3,4,5]`)

archs = {mlp, edge_transformer, diffusion_transformer, gcn, graph_neural_network,
graph_isomorphism_network, graph_transformer}.

## Verification results (all PASS)

Torch-free ONNX runtime vs frozen torch baselines, worst |Δ| over 8 bench ops:
```
diffusion_transformer/p5   2.38e-07  (atol 1e-4) OK
edge_transformer/p5        2.98e-07  (atol 1e-4) OK
gcn/p5                     1.79e-07  (atol 1e-4) OK
graph_isomorphism_network/p5 1.79e-07 (atol 1e-4) OK
graph_neural_network/p5    1.19e-07  (atol 1e-4) OK
graph_transformer/p5       1.46e-03  (atol 2e-3) OK   # looser: Laplacian-PE eigvec sign ambiguity
mlp/p5                     4.77e-07  (atol 1e-4) OK
```
Each p5 bundle returns 10 angles. Export parity (torch vs ONNX on EXAMPLE_OP) was
≤1e-5 for all 7 (gate refuses to write above that).

## How to reproduce the export/baseline pipeline

Export env: `~/gitco/qaoa_training_pipeline/.venv-export/` (torch 2.13 + PyG 2.8 +
onnx 1.22 + onnxruntime 1.28). It has NO pytest and NO pip; runtime `python` also
resolves to this venv in the shell.

**One-time tooling is recovered but LEFT UNCOMMITTED** in the working tree (it was
deliberately deleted in `bd9d5a6` to keep the runtime torch-free; committing it
would reverse that). Recover with:
```bash
git checkout bd9d5a6^ -- tools/inference/export_onnx.py tools/inference/gen_baselines.py \
    qaoa_training_pipeline/inference/torch_backend
```

1. **Export** (norm-independent, so no re-export needed after fixing stats):
   ```bash
   for m in mlp edge_transformer diffusion_transformer gcn graph_neural_network \
            graph_isomorphism_network graph_transformer; do
     .venv-export/bin/python tools/inference/export_onnx.py --model $m/p5 --check-parity
   done
   ```
2. **Baselines** — the recovered torch `lightweight_predictor.py`/`feature_extractor`
   need methods (`undo_gamma_rescale`, `extract_and_pack`) that the ONNX-only
   refactor removed from the runtime modules. Temporarily overlay the old
   (superset) versions, generate, then RESTORE:
   ```bash
   DU=qaoa_training_pipeline/inference/datamodule_utils.py
   FE=qaoa_training_pipeline/inference/feature_extractor.py
   cp "$DU" /tmp/du.cur.py; cp "$FE" /tmp/fe.cur.py
   git show bd9d5a6^:$DU > "$DU"; git show bd9d5a6^:$FE > "$FE"
   for m in mlp edge_transformer diffusion_transformer gcn graph_neural_network \
            graph_isomorphism_network graph_transformer; do
     .venv-export/bin/python tools/inference/gen_baselines.py --model $m/p5
   done
   cp /tmp/du.cur.py "$DU"; cp /tmp/fe.cur.py "$FE"   # restore runtime (verify: git diff --stat empty)
   ```
   (bench ops are non-degenerate, so the old `rescaling_factor` lacking the
   `e8d57e1` zero-guard doesn't change baseline values.)
3. **Setup**: `.venv-export/bin/python tools/inference/hf_setup.py` (keeps
   existing placeholder repo_id, rescans all local weights → 35 bundles).

## Runtime architecture (torch-free, unchanged — do not touch)

`qaoa_training_pipeline/inference/`: `ai_inference.py` (AIInference /
ProblemParamsProvider), `onnx_predictor.py` (OnnxQAOAPredictor), `feature_extractor.py`
(numpy `extract_np`/`pack_features_np`), `onnx_inputs.py` (numpy_input_builders),
`model_registry.py` (local-first, lazy HF download), `config_io.py`.
Scale-invariance: operator normalized by `rescaling_factor(cost_op)` on input;
outputs × π/2, then gammas divided by `rescale_a`; gated by `denormalize_output`.

## Open items / decisions for the user

1. **Recovered torch tooling is uncommitted** (`tools/inference/export_onnx.py`,
   `gen_baselines.py`, `qaoa_training_pipeline/inference/torch_backend/**`). Decide:
   delete, or keep tracked somewhere. Not in commit 5748713.
2. **HF publish** not done (see correction above). Only if a real HF home is
   wanted for the whole zoo.
3. Unrelated pre-existing working-tree changes were left untouched and OUT of the
   commit: `Makefile`, `.github/workflows/main.yml`, `requirements-dev.txt`,
   `agent.md`, `P5_ONNX_HANDOFF.md`.
4. `~/.../checkpoints_open_source/p_5/edge_transformer ` still has the trailing
   space in the dir name; the config references the literal path with the space,
   which works. Normalize upstream if desired.
