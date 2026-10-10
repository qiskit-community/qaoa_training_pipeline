"""Torch-free ONNX predictor for QAOA models.

``OnnxQAOAPredictor`` runs a pre-exported ``.onnx`` model with ``onnxruntime``
and numpy — no torch, no torch_geometric. It exposes ``predict`` with
``output_dim`` validation so ``AIInference`` and existing callers work
unchanged.

Models are ingested from the HuggingFace Hub and nowhere else, through a
single entry point: :meth:`from_bundle`, which takes a ``<model>/p<p>`` bundle
key. The setup file resolves the key to a repo at a pinned revision; the bundle
is downloaded (cached under ``~/.cache/huggingface``) and its snapshot
directory handed to the constructor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
from qaoa_training_pipeline.inference.model_registry import (
    CONFIG_FILENAME,
    ONNX_FILENAME,
    bundle_entry,
    PRIVATE_CONFIG_FIELDS,
    expected_model_type,
    resolve_bundle,
)
from qaoa_training_pipeline.inference.onnx_inputs import INPUT_BUILDERS


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
        bundle_dir: Path | str,
        bundle_key: str,
        repo_id: str,
        revision: str,
        device: str = "cpu",
        strict: bool = True,
    ) -> None:
        """Wrap an already-downloaded bundle snapshot.

        Use :meth:`from_bundle` instead of calling this directly: it is what
        downloads the snapshot that ``bundle_dir`` points at.

        Args:
            bundle_dir: Directory of the downloaded snapshot, holding
                ``model_config.json`` next to the ONNX artifacts.
            bundle_key: ``<model>/p<p>`` key the bundle was resolved from.
            repo_id: HuggingFace repo the snapshot came from.
            revision: Revision (commit sha or branch) it was downloaded at.
            device: Device for inference ("cpu", "cuda", ...).
            strict: Reserved for parity with other providers; unused.
        """
        self.bundle_dir = Path(bundle_dir)
        self.device = str(device)
        self.strict = bool(strict)
        # Where the snapshot came from. This, not bundle_dir, is what gets
        # serialized: bundle_dir points into a machine-specific HF cache.
        self.bundle_key = bundle_key
        self.repo_id = repo_id
        self.revision = revision

        self.config = self._load_config()
        self.model_init = self.config.get("model_init", {})
        self.model_type = str(self.model_init.get("model_type", "")).lower()
        self.in_features = list(self.model_init.get("in_features", []))
        self.output_dim = int(self.model_init.get("output_dim", 0))

        # The key fixes which architecture the download must be: a mismatch
        # means the setup file points at the wrong repo, which would otherwise
        # surface only as quietly wrong angles.
        wanted = expected_model_type(self.bundle_key)
        if self.model_type != wanted:
            raise ValueError(
                f"Bundle {self.bundle_key!r} must declare model_type {wanted!r}, but "
                f"{self.repo_id} at revision {self.revision} declares "
                f"{self.model_type!r}. The setup file points at the wrong repo."
            )

        if self.model_type not in INPUT_BUILDERS:
            raise KeyError(
                f"No ONNX input builder registered for model type {self.model_type!r}. "
                f"Registered: {sorted(INPUT_BUILDERS)}"
            )
        self._prepare = INPUT_BUILDERS[self.model_type]

        # The .onnx sits next to the config under a name fixed by the bundle
        # contract; a snapshot that lacks it is an incomplete download.
        self.onnx_path = self.bundle_dir / ONNX_FILENAME
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"{ONNX_FILENAME} missing from bundle {self.repo_id} at {self.bundle_dir}. "
                "The snapshot looks incomplete; clear it from the HuggingFace cache "
                "and download it again."
            )

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
        """Load a model bundle by its ``<model>/p<p>`` key. The only way in.

        The setup file resolves the key to a HuggingFace repo at a pinned
        revision; the bundle is downloaded once and cached (see
        :mod:`~qaoa_training_pipeline.inference.model_registry`).
        """
        entry = bundle_entry(bundle_key)
        return cls(
            bundle_dir=resolve_bundle(bundle_key),
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            bundle_key=bundle_key,
            **kwargs,
        )

    def _load_config(self) -> dict[str, Any]:
        """Read ``model_config.json`` from the downloaded snapshot."""
        config_file = self.bundle_dir / CONFIG_FILENAME
        if not config_file.is_file():
            raise FileNotFoundError(
                f"{CONFIG_FILENAME} missing from bundle {self.repo_id} at {self.bundle_dir}. "
                "The snapshot looks incomplete; clear it from the HuggingFace cache "
                "and download it again."
            )
        with open(config_file, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        # Drop private training-environment fields the publisher may have left
        # in, so they cannot reach metadata() or a serialized config.
        for field in PRIVATE_CONFIG_FIELDS:
            config.pop(field, None)
        return config

    def source(self) -> dict[str, str]:
        """The Hub address this predictor's artifacts came from, serializable.

        Never the local snapshot directory: that path is specific to one
        machine's HuggingFace cache.
        """
        # The setup owns the revision, so the key alone is a complete,
        # pin-stable address.
        return {"model": self.bundle_key}

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
