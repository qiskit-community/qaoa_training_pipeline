"""Torch-free ONNX predictor for QAOA models.

``OnnxQAOAPredictor`` runs a pre-exported ``.onnx`` model with ``onnxruntime``
and numpy — no torch, no torch_geometric. It exposes ``predict`` with
``output_dim`` validation so ``AIInference`` and existing callers work
unchanged.

Models are ingested from the HuggingFace Hub: use :meth:`from_bundle` for a
bundle from the shipped zoo, :meth:`from_hf` for an arbitrary bundle repo, or
the constructor directly to point at a local export directory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.config_io import (
    CONFIG_FILENAME,
    load_config,
    resolve_bundle_path,
)
from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
from qaoa_training_pipeline.inference.model_registry import (
    bundle_entry,
    download_bundle,
    resolve_bundle,
)
from qaoa_training_pipeline.inference.onnx_inputs import numpy_input_builders

DEFAULT_ONNX_FILENAME = "model.onnx"


def denormalize_qaoa_params_np(
    qaoa_params_norm: np.ndarray, scale: float = math.pi / 2
) -> np.ndarray:
    """Denormalize predicted QAOA params by ``scale`` (default pi/2)."""
    return qaoa_params_norm * scale


def undo_gamma_rescale_np(angles: np.ndarray, p: int, rescale_a: float) -> np.ndarray:
    """Undo the gamma rescaling on the second half of ``angles`` (divide gammas)."""
    angles = np.array(angles, copy=True)
    angles[..., p:] = angles[..., p:] / float(rescale_a)
    return angles


class OnnxQAOAPredictor:
    """Torch-free predictor backed by an exported ONNX model.

    Example:
        predictor = OnnxQAOAPredictor.from_bundle("gcn/p1")
        angles = predictor.predict(cost_op)
    """

    def __init__(
        self,
        config_path: Path | str,
        device: str = "cpu",
        strict: bool = True,
        onnx_path: Path | str | None = None,
        bundle_key: str | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.device = str(device)
        self.strict = bool(strict)
        # Hub coordinates when this predictor came from from_bundle/from_hf.
        # They are what gets serialized, since config_path then points into a
        # machine-specific HF cache directory.
        self.bundle_key = bundle_key
        self.repo_id = repo_id
        self.revision = revision

        self.config = load_config(self.config_path)
        self.model_init = self.config.get("model_init", {})
        self.model_type = str(self.model_init.get("model_type", "")).lower()
        self.in_features = list(self.model_init.get("in_features", []))
        self.output_dim = int(self.model_init.get("output_dim", 0))

        if self.model_type not in numpy_input_builders:
            raise KeyError(
                f"No ONNX input builder registered for model type {self.model_type!r}. "
                f"Registered: {sorted(numpy_input_builders)}"
            )
        self._prepare = numpy_input_builders[self.model_type]

        # Resolve the .onnx artifact: explicit arg > config "onnx" key > default
        # filename. Without an explicit path it sits next to the config — true
        # both for a downloaded bundle snapshot and for a local export dir.
        if onnx_path is not None:
            resolved = Path(onnx_path)
        else:
            filename = self.config.get("onnx", DEFAULT_ONNX_FILENAME)
            resolved = resolve_bundle_path(self.config_path, filename)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {resolved}. Bundles are downloaded from the "
                "HuggingFace Hub — use OnnxQAOAPredictor.from_bundle('<model>/p<p>') "
                "or from_hf('<org>/<repo>') instead of a local path."
            )
        self.onnx_path = resolved

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self._input_names = {i.name for i in self.session.get_inputs()}

        norm_stats = (
            self.config.get("feature_normalization") or self.model_init.get("norm_stats") or {}
        )
        self.feature_extractor = AIFeatureExtractor(
            in_features=self.in_features,
            norm_stats=norm_stats,
        )

    @classmethod
    def from_bundle(cls, bundle_key: str, **kwargs: Any) -> "OnnxQAOAPredictor":
        """Load a bundle of the shipped zoo by its ``<model>/p<p>`` key.

        The manifest resolves the key to a HuggingFace repo at a pinned
        revision; the bundle is downloaded once and cached (see
        :mod:`~qaoa_training_pipeline.inference.model_registry`).
        """
        entry = bundle_entry(bundle_key)
        bundle = resolve_bundle(bundle_key)
        return cls(
            config_path=bundle / CONFIG_FILENAME,
            bundle_key=bundle_key,
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            **kwargs,
        )

    @classmethod
    def from_hf(cls, repo_id: str, revision: str = "main", **kwargs: Any) -> "OnnxQAOAPredictor":
        """Download a bundle repo from the Hub and wrap it in a predictor.

        For bundles outside the shipped manifest (a private export, a retrain).
        ``snapshot_download`` caches under ``~/.cache/huggingface``, so re-runs
        are offline.
        """
        bundle = download_bundle(repo_id, revision=revision)
        return cls(
            config_path=bundle / CONFIG_FILENAME,
            repo_id=repo_id,
            revision=revision,
            **kwargs,
        )

    def source(self) -> dict[str, str]:
        """Where this predictor's artifacts came from, in serializable form.

        Prefers the Hub coordinates; falls back to the local config path for a
        predictor built directly from a directory.
        """
        if self.bundle_key is not None:
            # The manifest owns the revision for a zoo bundle, so the key alone
            # is a complete, pin-stable address.
            return {"model": self.bundle_key}
        if self.repo_id is not None:
            return {"repo_id": self.repo_id, "revision": self.revision or "main"}
        return {"config_path": str(self.config_path)}

    def metadata(self) -> dict[str, Any]:
        """Return predictor metadata from the config."""
        metadata = dict(self.config)
        if "output_dim" not in metadata and "model_init" in metadata:
            metadata["output_dim"] = metadata["model_init"].get("output_dim")
        return metadata

    def predict(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit | None = None,  # pylint: disable=unused-argument
        ansatz_circuit: QuantumCircuit | None = None,  # pylint: disable=unused-argument
        initial_state: QuantumCircuit | None = None,  # pylint: disable=unused-argument
        denormalize: bool | None = None,
    ) -> list[float]:
        """Predict QAOA parameters from a cost operator (torch-free).

        ``mixer``/``ansatz_circuit``/``initial_state`` are accepted to match the
        provider signature but are currently unused.
        """
        x_vec, features = self.feature_extractor.extract_and_pack_np(cost_op)
        features["x"] = x_vec
        # Some builders need model hyperparameters (e.g. graph_transformer's
        # Laplacian PE width). Surface them from model_init so the numpy feed
        # matches what the graph was exported with.
        if "pos_enc_dim" in self.model_init:
            features["pos_enc_dim"] = int(self.model_init["pos_enc_dim"])

        feed = self._prepare(features)
        # Only pass inputs the exported graph actually declares (keeps optional
        # inputs like edge_weights from erroring when the graph omits them).
        feed = {k: v for k, v in feed.items() if k in self._input_names}
        prediction = self.session.run(None, feed)[0]
        prediction = np.asarray(prediction, dtype=np.float32)

        apply_denorm = (
            bool(self.config.get("denormalize_output", True))
            if denormalize is None
            else denormalize
        )
        if apply_denorm:
            scale = float(self.config.get("output_scale", math.pi / 2))
            prediction = denormalize_qaoa_params_np(prediction, scale=scale)
            p = self.output_dim // 2
            if p > 0:
                rescale_a = float(features["rescale_a"])
                prediction = undo_gamma_rescale_np(prediction, p=p, rescale_a=rescale_a)

        output = prediction.reshape(-1).tolist()

        if self.output_dim > 0 and len(output) != self.output_dim:
            raise ValueError(
                f"Predicted {len(output)} values, expected {self.output_dim}. "
                f"Model output shape: {prediction.shape}"
            )

        return [float(value) for value in output]
