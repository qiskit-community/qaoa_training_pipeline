#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""AI Inference trainer implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.framework import ProblemParamsProvider
from qaoa_training_pipeline.framework import ParamResult
from qaoa_training_pipeline.training.functions import (
    BaseAnglesFunction,
    IdentityFunction,
)


# TODO: REMOVE EVERYTHING THAT RELATES TO LOCAL BUNDLE/CONFIG -- ONLY SUPPORT HUGGINGFACE. NO ADDITIONAL VALUE FROM LOCAL


class AIInference(ProblemParamsProvider):
    """AI-based inference for QAOA angle prediction.

    Loads a pre-trained AI model and predicts QAOA angles from a cost
    operator. The referenced model config declares which training setting
    the checkpoint was produced under.

    The model is addressed in one of three ways: ``model`` (a ``<model>/p<p>``
    bundle key from the shipped zoo), ``repo_id`` (any HuggingFace bundle repo)
    or ``config_path`` (a local export directory).

    Inference is torch-free: it runs an exported ``model.onnx`` with
    ``onnxruntime`` and numpy. Requires the optional ``onnxruntime`` dependency
    (``pip install qaoa_training_pipeline[inference]``) and needs neither torch
    nor the original checkpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        config_path: str | None = None,
        repo_id: str | None = None,
        revision: str = "main",
        device: str = "cpu",
        strict: bool = True,
        validate_input_operator: bool = True,
        rescale: Callable[[Sequence[float]], Sequence[float]] | None = None,
        qaoa_angles_function: BaseAnglesFunction | None = None,
    ) -> None:
        """Initialize the AI inference trainer.

        Exactly one of ``model``, ``config_path`` or ``repo_id`` must be given.

        Args:
            model: Bundle key of a model in the shipped zoo, ``"<model>/p<p>"``
                (e.g. ``"gcn/p3"``). Resolved to a HuggingFace repo at a pinned
                revision via ``inference/hf_setup.json`` and downloaded once
                into the local HF cache. See
                :func:`~qaoa_training_pipeline.inference.model_registry.available_bundles`.
            config_path: Path to a model_config.json file (or its enclosing
                directory) describing the model and inputs to load. For a local
                export; the ONNX artifacts must sit next to the config.
            repo_id: HuggingFace repo holding a bundle outside the manifest.
            revision: Revision to download when ``repo_id`` is used.
            device: Device for inference ("cpu", "cuda", ...).
            strict: Reserved for parity with other providers; unused by the
                ONNX runtime.
            validate_input_operator: If ``True``, cross-check the predicted
                angle count against the config's ``output_dim``.
            rescale: Optional post-processing hook applied to the predicted
                angles *after* the config's own denormalization. Receives the
                angle list and must return a same-length sequence. Use e.g.
                ``lambda a: [x * 2 for x in a]`` to rescale to ``[0, π]``, or
                pass ``None`` (default) to keep only the config's rescaling.
            qaoa_angles_function: Function transforming the predicted angles to
                a different basis before use. Defaults to
                :class:`IdentityFunction` (no transformation).
        """
        super().__init__(qaoa_angles_function=qaoa_angles_function or IdentityFunction())
        given = [
            name
            for name, val in (("model", model), ("config_path", config_path), ("repo_id", repo_id))
            if val is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "AIInference needs exactly one of 'model', 'config_path' or 'repo_id', "
                f"got {given or 'none'}."
            )
        self.model_key = model
        self.config_path = config_path
        self.repo_id = repo_id
        self.revision = str(revision)
        self.device = str(device)
        self.strict = bool(strict)
        self.validate_input_operator = bool(validate_input_operator)
        self.rescale = rescale
        self.model = None

        self.load_model()

    def provide_params(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
    ) -> ParamResult:
        """Return QAOA angles by running inference on the loaded model."""
        start = time()

        if self.model is None:
            raise RuntimeError("AI inference model config was not loaded.")

        # A user-supplied rescale hook replaces the config's rescaling —
        # skip the built-in denormalization so the hook sees the raw output.
        qaoa_angles = self.model.predict(
            cost_op,
            mixer=mixer,
            ansatz_circuit=ansatz_circuit,
            initial_state=initial_state,
            denormalize=None if self.rescale is None else False,
        )

        if self.rescale is not None:
            rescaled = list(self.rescale(qaoa_angles))
            if len(rescaled) != len(qaoa_angles):
                raise ValueError(
                    f"rescale hook changed the number of angles: "
                    f"{len(qaoa_angles)} -> {len(rescaled)}."
                )
            qaoa_angles = [float(value) for value in rescaled]

        if self.validate_input_operator:
            expected_dim = self.model.metadata().get("output_dim")
            if expected_dim is not None and len(qaoa_angles) != int(expected_dim):
                raise ValueError(
                    f"Predicted {len(qaoa_angles)} QAOA parameters but config expects "
                    f"output_dim={expected_dim}."
                )

        energy = None

        result = ParamResult(qaoa_angles, time() - start, self, energy)
        result["ai_inference"] = {
            **self.model.source(),
            "device": self.device,
            "strict": self.strict,
            "predictor_metadata": self.model.metadata(),
        }
        return result

    def features(self, cost_op):
        """Return the packed feature vector for ``cost_op`` (numpy path)."""
        return self.model.feature_extractor.extract_and_pack_np(cost_op)

    @classmethod
    def from_config(cls, config: dict) -> "AIInference":
        """Return an instance of the class based on a config."""
        config = dict(config)

        # Accept legacy keys (`model_bundle`, `model_path`) alongside the
        # current `config_path` — old call sites keep working.
        config_path = config.get(
            "config_path",
            config.get("model_bundle", config.get("model_path")),
        )
        model = config.get("model")
        repo_id = config.get("repo_id")
        if model is None and repo_id is None and config_path is None:
            raise ValueError(
                "AIInference requires one of 'model', 'repo_id' or 'config_path' in config."
            )

        # Rebuild the angles function from its config when serialized; default
        # to the identity transformation otherwise.
        angles_function = None
        if "qaoa_angles_function" in config:
            from qaoa_training_pipeline.training.functions import FUNCTIONS

            angles_function = FUNCTIONS[config["qaoa_angles_function"]](
                **config.get("qaoa_angles_function_init", {})
            )

        return cls(
            model=model,
            config_path=config_path,
            repo_id=repo_id,
            revision=str(config.get("revision", "main")),
            device=str(config.get("device", "cpu")),
            strict=bool(config.get("strict", True)),
            validate_input_operator=bool(config.get("validate_input_operator", True)),
            rescale=config.get("rescale"),
            qaoa_angles_function=angles_function,
        )

    def to_config(self) -> dict:
        """Create a serializable dictionary describing the instance."""
        # Serialize the model's address, not self.config_path: for a Hub-ingested
        # bundle that path points into a machine-specific HF cache and would not
        # round-trip elsewhere.
        config = {
            **(self.model.source() if self.model is not None else self._address()),
            "device": self.device,
            "strict": self.strict,
            "validate_input_operator": self.validate_input_operator,
            "qaoa_angles_function": self.qaoa_angles_function.__class__.__name__,
        }

        if self.model is not None:
            config["predictor_metadata"] = self.model.metadata()

        return config

    def _address(self) -> dict:
        """The model address as given to the constructor (pre-load fallback)."""
        if self.model_key is not None:
            return {"model": self.model_key}
        if self.repo_id is not None:
            return {"repo_id": self.repo_id, "revision": self.revision}
        return {"config_path": str(self.config_path)}

    def parse_train_kwargs(self, args_str: str | None = None) -> dict:
        """Extract supported runtime keyword arguments from a string."""
        train_kwargs = {}
        for key, val in self.extract_train_kwargs(args_str).items():
            if key == "device":
                train_kwargs[key] = str(val)
            elif key in {"strict", "validate_input_operator"}:
                train_kwargs[key] = val.lower() == "true"
            else:
                raise ValueError(f"Unknown key {key!r} in provided train_kwargs.")
        return train_kwargs

    def load_model(self) -> None:
        """Load the ONNX predictor from whichever model address was given."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        shared = {"device": self.device, "strict": self.strict}

        if self.model_key is not None:
            self.model = OnnxQAOAPredictor.from_bundle(self.model_key, **shared)
        elif self.repo_id is not None:
            self.model = OnnxQAOAPredictor.from_hf(self.repo_id, revision=self.revision, **shared)
        else:
            self.model = OnnxQAOAPredictor(config_path=Path(self.config_path), **shared)
