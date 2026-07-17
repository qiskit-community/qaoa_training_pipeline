#!/usr/bin/env python
"""Export trained QAOA models to ONNX (one-time, torch/export-extra only).

This script is NEVER imported by the runtime. It loads a trained ``.ckpt`` via
the existing torch loader, wraps each architecture into a fixed-signature module
whose inputs match the numpy input builders in ``onnx_inputs.py``, exports to
``.onnx`` with torch's dynamo exporter, and (with ``--check-parity``) refuses to
write unless the ONNX output matches the torch output within tolerance.

Usage:
    python scripts/export_onnx.py --model agg_transformer --check-parity
    python scripts/export_onnx.py --model all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from qiskit.quantum_info import SparsePauliOp

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qaoa_training_pipeline.inference.config_io import (
    load_config,
    resolve_bundle_path,
)  # noqa: E402
from qaoa_training_pipeline.inference.torch_backend.model_loader import (
    load_model_from_config,
)  # noqa: E402
from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor  # noqa: E402
from qaoa_training_pipeline.inference.torch_backend.models import predictors  # noqa: E402
from qaoa_training_pipeline.inference.onnx_inputs import numpy_input_builders  # noqa: E402

MODEL_CONFIGS_DIR = REPO_ROOT / "qaoa_training_pipeline" / "inference" / "model_configs"
DEFAULT_ONNX_FILENAME = "model.onnx"

# A representative cost operator used to build export example inputs and to
# parity-check. Small, connected, weighted-ish.
EXAMPLE_OP = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])


# ---------- fixed-signature export wrappers --------------------------------


class MLPWrapper(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x):  # noqa: D401
        return self.core(x)


class EdgeTransformerWrapper(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x, edges, edge_weights):
        return self.core(x, edges, edge_weights)


class DiffusionWrapper(nn.Module):
    """Bakes eval t=0 (no-noise) into the graph so ``t`` is not an input."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x, edges, edge_weights):
        t = torch.zeros(x.shape[0], dtype=torch.long)
        return self.core(x, edges, t, edge_weights=edge_weights)


class GraphListWrapper(nn.Module):
    """Fixed-signature wrapper for GNNRegression / GINRegression, whose forward
    takes a Python list of (edge_index, edge_attr) tuples. At B=1 we reassemble
    that single-element list internally from plain tensor inputs so the module
    traces to ONNX."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x, edge_index, edge_attr, node_count):
        return self.core(x, [(edge_index, edge_attr)], node_count=node_count)


class GCNWrapper(nn.Module):
    """Fixed-signature wrapper for GCNModel. Reuses the trained conv/linear
    submodules but skips the PyG ``Batch``/``to_data_list`` path (which won't
    trace) — the per-node z-scored degree is precomputed in numpy (``node_x``)
    and ``global_mean_pool`` is a plain mean over nodes at B=1."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, node_x, edge_index, edge_weight, x):
        import torch.nn.functional as F

        h = self.core.conv1(node_x, edge_index, edge_weight)
        h = F.relu(h)
        # dropout is identity in eval mode
        h = self.core.conv2(h, edge_index, edge_weight)
        h = F.relu(h).to(torch.float32)
        pooled = h.mean(dim=0, keepdim=True)  # global_mean_pool, B=1
        z = torch.cat([pooled, x], dim=1).to(torch.float32)
        z = self.core.ln1(z)
        z = F.relu(z)
        z = self.core.ln2(z)
        return z


class GraphTransformerWrapper(nn.Module):
    """Fixed-signature wrapper for GraphTransformer at B=1. The two ONNX-hostile
    steps — degree node features and the ``torch.linalg.eigh`` Laplacian PE — are
    precomputed in numpy (``node_features``, ``pos_enc``) and passed in, so this
    wrapper only runs the trained embedding / transformer-layer / head stack over
    a single graph (no batch loop, no eigendecomposition)."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    @staticmethod
    def _attn(attn, x, edge_index, edge_weight):
        """Re-expression of GraphMultiHeadAttention.forward that trades the
        trace-hostile ``edge_mask.fill_diagonal_`` for an equivalent arange
        index-put (fill_diagonal_ specializes num_nodes to a constant). Reuses
        the trained projection/edge-bias submodules; eval-mode dropout is
        identity. edge_weight is always present for this model."""
        import torch.nn.functional as F

        batch_size, num_nodes, _ = x.shape
        q = (
            attn.q_proj(x)
            .view(batch_size, num_nodes, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )
        k = (
            attn.k_proj(x)
            .view(batch_size, num_nodes, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )
        v = (
            attn.v_proj(x)
            .view(batch_size, num_nodes, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scaling

        edge_mask = torch.zeros(num_nodes, num_nodes, device=x.device)
        edge_mask[edge_index[0], edge_index[1]] = 1.0
        diag = torch.arange(num_nodes, device=x.device)
        edge_mask[diag, diag] = 1.0  # == fill_diagonal_(1.0), trace-safe
        edge_mask = edge_mask.view(1, 1, num_nodes, num_nodes)
        attn_scores = attn_scores.masked_fill(edge_mask == 0, float("-inf"))

        ew = edge_weight.to(device=x.device, dtype=x.dtype).view(-1)
        encoded_weight = attn._signed_log1p(ew).unsqueeze(-1)
        per_head_bias = attn.edge_bias_mlp(encoded_weight).transpose(0, 1)
        dense_bias = torch.zeros(
            attn.num_heads, num_nodes, num_nodes, device=x.device, dtype=x.dtype
        )
        dense_bias[:, edge_index[0], edge_index[1]] = per_head_bias
        attn_scores = attn_scores + dense_bias.unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, attn.embed_dim)
        return attn.out_proj(attn_output)

    def _layer(self, layer, x, edge_index, edge_weight):
        x = layer.norm1(x + self._attn(layer.self_attn, x, edge_index, edge_weight))
        x = layer.norm2(x + layer.ff_net(x))
        return x

    def forward(self, x, node_features, pos_enc, edge_index, edge_weight):
        core = self.core
        x_i = core.node_embedding(node_features)
        x_i = x_i + core.pos_embedding(pos_enc)
        x_i = x_i.unsqueeze(0)  # [1, N, embed_dim]
        for layer in core.transformer_layers:
            x_i = self._layer(layer, x_i, edge_index, edge_weight)
        graph_emb = core.pool_node_features(x_i)  # [1, embed_dim]
        global_emb = core.global_proj(x)
        combined = torch.cat([graph_emb, global_emb], dim=1)
        return core.regression_head(combined)


def _build_export(model_type: str, core: nn.Module, features: dict):
    """Return (wrapped_module, example_args_tuple, input_names) for export."""
    x = torch.as_tensor(features["x"], dtype=torch.float32)
    edges = torch.as_tensor(features["edges"], dtype=torch.long)
    ew = torch.as_tensor(features["edge_weights"], dtype=torch.float32)

    if model_type == "agg_transformer":
        return MLPWrapper(core).eval(), (x,), ["x"]
    if model_type == "edge_transformer":
        return EdgeTransformerWrapper(core).eval(), (x, edges, ew), ["x", "edges", "edge_weights"]
    if model_type == "diffusion_transformer":
        return DiffusionWrapper(core).eval(), (x, edges, ew), ["x", "edges", "edge_weights"]
    if model_type in ("graph_neural_network", "graph_isomorphism_network"):
        feed = numpy_input_builders[model_type](features)
        edge_index = torch.as_tensor(feed["edge_index"], dtype=torch.long)
        edge_attr = torch.as_tensor(feed["edge_attr"], dtype=torch.float32)
        node_count = torch.as_tensor(feed["node_count"], dtype=torch.long)
        return (
            GraphListWrapper(core).eval(),
            (x, edge_index, edge_attr, node_count),
            ["x", "edge_index", "edge_attr", "node_count"],
        )
    if model_type == "gcn":
        feed = numpy_input_builders[model_type](features)
        node_x = torch.as_tensor(feed["node_x"], dtype=torch.float32)
        edge_index = torch.as_tensor(feed["edge_index"], dtype=torch.long)
        edge_weight = torch.as_tensor(feed["edge_weight"], dtype=torch.float32)
        return (
            GCNWrapper(core).eval(),
            (node_x, edge_index, edge_weight, x),
            ["node_x", "edge_index", "edge_weight", "x"],
        )
    if model_type == "graph_transformer":
        feed = numpy_input_builders[model_type](features)
        node_features = torch.as_tensor(feed["node_features"], dtype=torch.float32)
        pos_enc = torch.as_tensor(feed["pos_enc"], dtype=torch.float32)
        edge_index = torch.as_tensor(feed["edge_index"], dtype=torch.long)
        edge_weight = torch.as_tensor(feed["edge_weight"], dtype=torch.float32)
        return (
            GraphTransformerWrapper(core).eval(),
            (x, node_features, pos_enc, edge_index, edge_weight),
            ["x", "node_features", "pos_enc", "edge_index", "edge_weight"],
        )
    raise NotImplementedError(f"Export wrapper for {model_type!r} not implemented yet.")


def export_model(model_name: str, opset: int, check_parity: bool) -> Path:
    config_path = MODEL_CONFIGS_DIR / model_name / "model_config.json"
    config = load_config(config_path)
    model_type = str(config["model_init"]["model_type"]).lower()

    loaded = load_model_from_config(config, config_path, device="cpu", strict=True)
    core = loaded.core.eval()

    ext = AIFeatureExtractor(
        in_features=config["model_init"]["in_features"],
        norm_stats=config.get("feature_normalization")
        or config["model_init"].get("norm_stats")
        or {},
    )
    _, features_np = ext.extract_and_pack_np(EXAMPLE_OP)
    features_np["x"] = ext.pack_features_np(features_np)
    # Surface hyperparameters the numpy builders need (kept in sync with the
    # runtime predictor, which reads the same keys from model_init).
    if "pos_enc_dim" in config["model_init"]:
        features_np["pos_enc_dim"] = int(config["model_init"]["pos_enc_dim"])

    wrapper, args, input_names = _build_export(model_type, core, features_np)

    with torch.no_grad():
        torch_out = wrapper(*args).detach().cpu().numpy()

    out_path = resolve_bundle_path(config_path, config.get("onnx", DEFAULT_ONNX_FILENAME))
    tmp_path = out_path.with_suffix(".onnx.tmp")

    # Dynamic axes on node/edge counts; batch stays 1. Every input that carries
    # a node- or edge-count dimension must be declared here with the SAME symbol
    # name, or torch.export specializes the undeclared one to the example's
    # concrete size and then conflicts with the declared ones (GraphConv/GIN
    # require edge_index and edge_weight to share their edge count).
    dyn = {name: {} for name in input_names}
    if "edges" in input_names:
        dyn["edges"] = {1: "num_edges"}
    if "edge_weights" in input_names:
        dyn["edge_weights"] = {1: "num_edges"}
    if "edge_index" in input_names:
        dyn["edge_index"] = {1: "num_edges_sym"}
    if "edge_attr" in input_names:
        dyn["edge_attr"] = {0: "num_edges_sym"}
    if "edge_weight" in input_names:  # gcn: singular, 1-D (num_edges_sym,)
        dyn["edge_weight"] = {0: "num_edges_sym"}
    if "node_x" in input_names:  # gcn: per-node degree feature (num_nodes, 1)
        dyn["node_x"] = {0: "num_nodes"}
    if "node_features" in input_names:  # graph_transformer: (num_nodes, node_feature_dim)
        dyn["node_features"] = {0: "num_nodes"}
    if "pos_enc" in input_names:  # graph_transformer: (num_nodes, pos_enc_dim)
        dyn["pos_enc"] = {0: "num_nodes"}

    torch.onnx.export(
        wrapper,
        tuple(args),
        str(tmp_path),
        dynamo=True,
        opset_version=opset,
        input_names=input_names,
        output_names=["angles"],
        dynamic_axes=dyn,
    )

    if check_parity:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
        feed = numpy_input_builders[model_type](features_np)
        names = {i.name for i in sess.get_inputs()}
        feed = {k: v for k, v in feed.items() if k in names}
        onnx_out = np.asarray(sess.run(None, feed)[0], dtype=np.float32)
        max_diff = float(np.abs(onnx_out - torch_out).max())
        if not np.allclose(onnx_out, torch_out, atol=1e-5):
            tmp_path.unlink(missing_ok=True)
            raise SystemExit(
                f"PARITY FAILED for {model_name}: max abs diff {max_diff:.2e} > 1e-5. Not writing."
            )
        print(f"  parity OK (max abs diff {max_diff:.2e})")

    # Re-save with a clean, final external-data filename. Exporting to a temp
    # path makes torch name the sidecar ``model.onnx.tmp.data`` and bake that
    # name into the graph; load it back and rewrite so the committed artifact is
    # ``model.onnx`` + ``model.onnx.data`` with a matching internal reference.
    import onnx

    data_name = out_path.name + ".data"  # model.onnx.data
    model_proto = onnx.load(str(tmp_path))  # pulls in tmp sidecar via its ref
    for stale in (out_path, out_path.with_name(data_name)):
        stale.unlink(missing_ok=True)
    onnx.save_model(
        model_proto,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    # Remove the intermediate temp artifacts.
    tmp_path.unlink(missing_ok=True)
    tmp_path.with_name(tmp_path.name + ".data").unlink(missing_ok=True)

    print(f"  wrote {out_path.relative_to(REPO_ROOT)} (+ {data_name})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export QAOA models to ONNX.")
    ap.add_argument("--model", default="all", help="model_config name, or 'all'")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--check-parity", action="store_true")
    args = ap.parse_args()

    if args.model == "all":
        names = sorted(
            p.name for p in MODEL_CONFIGS_DIR.iterdir() if (p / "model_config.json").exists()
        )
    else:
        names = [args.model]

    for name in names:
        print(f"[{name}]")
        try:
            export_model(name, args.opset, args.check_parity)
        except NotImplementedError as e:
            print(f"  SKIP: {e}")


if __name__ == "__main__":
    main()
