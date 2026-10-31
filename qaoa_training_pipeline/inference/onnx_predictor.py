#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

import copy
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
from qaoa_training_pipeline.inference.model_registry import (
    CONFIG_FILENAME,
    ONNX_FILENAME,
    PRIVATE_CONFIG_FIELDS,
    bundle_entry,
    expected_model_type,
    resolve_bundle,
)
from qaoa_training_pipeline.inference.onnx_inputs import INPUT_BUILDERS

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Running the inference models requires 'onnxruntime'. Install the inference extra: "
        "pip install qaoa_training_pipeline[inference]."
    ) from exc

logger = logging.getLogger(__name__)


def _strip_private_fields(value: Any) -> Any:
    """Recursively drop :data:`PRIVATE_CONFIG_FIELDS` from a decoded config."""
    if isinstance(value, dict):
        return {
            key: _strip_private_fields(item)
            for key, item in value.items()
            if key not in PRIVATE_CONFIG_FIELDS
        }
    if isinstance(value, list):
        return [_strip_private_fields(item) for item in value]
    return value


class OnnxQAOAPredictor:
    """Torch-free predictor backed by an exported ONNX model.

    The network step alone: it applies the bundle's ``output_scale`` (output
    units -> radians) but does not scale the cost operator or un-rescale the
    gammas, which :class:`~qaoa_training_pipeline.inference.AIInference` owns.

    Example:
        predictor = OnnxQAOAPredictor.from_bundle("gcn/p1")
        normalized, rescale_a = normalize_cost_op(cost_op)
        raw = predictor.predict_normalized(normalized, rescale_a)
        angles = undo_cost_op_normalization(
            raw, p=predictor.output_dim // 2, rescale_a=rescale_a
        )
    """

    def __init__(
        self,
        bundle_dir: Path | str,
        bundle_key: str,
        repo_id: str,
        revision: str,
        device: str = "cpu",
        prefactor: float | None = None,
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
            prefactor: The cost-operator convention this bundle was trained
                under, from its setup entry's ``prefactor``. Required; there is
                no default, since the convention belongs to the bundle.
        """
        self.bundle_dir = Path(bundle_dir)
        self.device = self._normalize_device(device)
        # Where the snapshot came from. This, not bundle_dir, is what gets
        # serialized: bundle_dir points into a machine-specific HF cache.
        self.bundle_key = bundle_key
        self.repo_id = repo_id
        self.revision = revision

        self.config = self._load_config()
        self.model_init = self.config.get("model_init", {})
        self.model_type = str(self.model_init.get("model_type", "")).lower()
        self.in_features = list(self.model_init.get("in_features", []))
        self.output_dim = self._require_output_dim()

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

        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=self._execution_providers()
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        logger.debug(
            "Loaded bundle %s (%s at %s) from %s on %s",
            self.bundle_key,
            self.repo_id,
            self.revision,
            self.bundle_dir,
            self.session.get_providers(),
        )

        if prefactor is None:
            raise ValueError(
                f"No prefactor given for bundle {self.bundle_key!r}. It is the cost-operator "
                "convention the model was trained under and must come from the bundle's "
                "setup entry; this package has no default to fall back on."
            )
        self.feature_extractor = AIFeatureExtractor(
            in_features=self.in_features,
            norm_stats=self.config.get("feature_normalization") or {},
            prefactor=prefactor,
        )

    @staticmethod
    def _normalize_device(device: str) -> str:
        """Map a device string onto the spellings :meth:`_execution_providers` knows.

        ``"CUDA"`` and ``"gpu"`` both mean the same thing as ``"cuda"`` to a
        caller, but an unnormalized string would miss the ``startswith("cuda")``
        test and run on the CPU instead -- the exact silent fallback that
        :meth:`_execution_providers` exists to prevent.

        Raises:
            ValueError: If the device is not a recognized spelling.
        """
        normalized = str(device).strip().lower()
        if normalized == "gpu":
            normalized = "cuda"
        if normalized == "cpu" or normalized == "cuda" or normalized.startswith("cuda:"):
            return normalized
        raise ValueError(
            f"Unsupported device {device!r}. Expected 'cpu', 'cuda', 'cuda:<index>' or 'gpu'."
        )

    def _require_output_dim(self) -> int:
        """The angle count the bundle declares, which must be a positive even number.

        This is not a cosmetic check. ``output_dim`` is what splits a prediction
        into betas and gammas, and only the gamma half is divided by the
        operator's rescaling factor. A bundle that omits it would leave ``p = 0``,
        so no gamma would be rescaled and the angles would silently stop tracking
        the operator's scale -- finite, plausible, and wrong.

        Raises:
            ValueError: If ``output_dim`` is absent, not an integer, or not a
                positive even number.
        """
        raw = self.model_init.get("output_dim")
        if raw is None:
            raise ValueError(
                f"Bundle {self.bundle_key!r} ({self.repo_id} at revision {self.revision}) "
                "declares no model_init.output_dim. It is required: it is what splits a "
                "prediction into betas and gammas, so without it the gamma rescaling "
                "cannot be applied and the angles would be silently wrong."
            )
        try:
            output_dim = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Bundle {self.bundle_key!r} declares a non-integer "
                f"model_init.output_dim {raw!r}."
            ) from exc
        if output_dim <= 0 or output_dim % 2 != 0:
            raise ValueError(
                f"Bundle {self.bundle_key!r} declares model_init.output_dim={output_dim}, "
                "which cannot be a QAOA angle count: it must be a positive even number "
                "(p betas followed by p gammas)."
            )
        return output_dim

    def _execution_providers(self) -> list[str]:
        """The onnxruntime providers for ``self.device``, checked up front.

        Raises:
            RuntimeError: If a CUDA device was asked for but this onnxruntime
                build has no CUDA provider. onnxruntime would otherwise fall
                back to the CPU provider on its own, turning a GPU run into a
                silently much slower CPU run.
        """
        if not self.device.startswith("cuda"):
            return ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"device={self.device!r} needs onnxruntime's CUDAExecutionProvider, which "
                f"this build does not have (available: {', '.join(available)}). Install "
                "onnxruntime-gpu, or use device='cpu'."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @classmethod
    def from_bundle(cls, bundle_key: str, **kwargs: Any) -> "OnnxQAOAPredictor":
        """Load a model bundle by its ``<model>/p<p>`` key. The only way in.

        The setup file resolves the key to a HuggingFace repo at a pinned
        revision; the bundle is downloaded once and cached (see
        :mod:`~qaoa_training_pipeline.inference.model_registry`).
        """
        entry = bundle_entry(bundle_key)
        # The convention travels with the bundle, so a model trained under a
        # different one needs only its setup entry changed, not this code.
        kwargs.setdefault("prefactor", entry.get("prefactor"))
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
        # in, so they cannot reach metadata() or a serialized config. Nested,
        # because the fields appear inside model_init as well as at the top
        # level, and a top-level-only pop would publish the nested copy.
        return _strip_private_fields(config)

    def source(self) -> dict[str, str]:
        """The Hub address this predictor's artifacts came from, serializable.

        Never the local snapshot directory: that path is specific to one
        machine's HuggingFace cache.
        """
        # The setup owns the revision, so the key alone is a complete,
        # pin-stable address.
        return {"model": self.bundle_key}

    def metadata(self) -> dict[str, Any]:
        """Return predictor metadata from the config.

        A deep copy: the result is handed to callers and embedded in serialized
        configs, and a shallow one would let them mutate the nested dicts this
        predictor is still using to run inference.
        """
        metadata = copy.deepcopy(self.config)
        if "output_dim" not in metadata and "model_init" in metadata:
            metadata["output_dim"] = metadata["model_init"].get("output_dim")
        return metadata

    def denormalizes(self, denormalize: bool | None = None) -> bool:
        """Whether a ``denormalize`` argument of this value applies the config's scaling.

        ``False`` returns the raw network
        output, which is in the units the model was trained to emit.
        """
        if denormalize is None:
            return bool(self.config.get("denormalize_output", True))
        return bool(denormalize)

    def predict_normalized(
        self,
        cost_op: SparsePauliOp,
        rescale_a: float,
        mixer: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        denormalize: bool | None = None,
    ) -> list[float]:
        """Run the network on an already-normalized operator. Step two of the chain.

        Everything between the two scaling steps: features, the ONNX session,
        and the config's output denormalization. Neither scaling step is done
        here -- the operator arrives normalized, and the gammas are not
        un-rescaled.

        Args:
            cost_op: The max-cut cost operator divided by ``rescale_a``, as
                :func:`~qaoa_training_pipeline.inference.scaling.normalize_cost_op`
                returns it.
            rescale_a: The scale it was divided by. Not applied to the operator
                again; it is reported to the network as the ``rescale_a``
                feature, which some bundles consume.
            mixer: Accepted to match the provider signature but unused; a
                non-default circuit warns rather than being honoured silently.
            ansatz_circuit: Unused, as ``mixer``.
            initial_state: Unused, as ``mixer``.
            denormalize: Overrides the config's ``denormalize_output``. ``False``
                returns the raw network output.
        """
        ignored = [
            name
            for name, circuit in (
                ("mixer", mixer),
                ("ansatz_circuit", ansatz_circuit),
                ("initial_state", initial_state),
            )
            if circuit is not None
        ]
        if ignored:
            warnings.warn(
                f"{', '.join(ignored)} ignored: the inference models were trained on the "
                "standard max-cut QAOA ansatz with the X mixer and the |+> initial state, "
                "so the predicted angles do not apply to a different circuit.",
                UserWarning,
                stacklevel=2,
            )

        x_vec, features = self.feature_extractor.extract_and_pack_np(cost_op, rescale_a=rescale_a)
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

        if self.denormalizes(denormalize):
            # The network emits angles in normalized units; output_scale maps
            # them back to radians. Unrelated to the operator's scale, and it
            # applies to betas and gammas alike.
            prediction = prediction * float(self.config.get("output_scale", math.pi / 2))

        output = prediction.reshape(-1).tolist()

        if len(output) != self.output_dim:
            raise ValueError(
                f"Predicted {len(output)} values, expected {self.output_dim}. "
                f"Model output shape: {prediction.shape}"
            )

        return [float(value) for value in output]
