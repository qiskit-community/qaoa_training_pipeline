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
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.framework import ProblemParamsProvider
from qaoa_training_pipeline.framework import ParamResult
from qaoa_training_pipeline.inference.model_registry import (
    available_bundles,
    parse_bundle_key,
)
from qaoa_training_pipeline.training.functions import (
    BaseAnglesFunction,
    IdentityFunction,
)


class AIInference(ProblemParamsProvider):
    """AI-based inference for QAOA angle prediction.

    Loads a pre-trained AI model and predicts QAOA angles from a cost
    operator. The referenced model config declares which training setting
    the checkpoint was produced under.

    There is one way to address a model: ``model``, a ``<model>/p<p>`` bundle
    key. The setup file resolves it to a HuggingFace repo at a pinned revision,
    so every run is reproducible and every key is a verified name. A bundle
    outside the shipped zoo (a private export, a retrain) is reached by
    listing it in your own setup file and pointing ``$QAOA_HF_SETUP`` at it —
    not by a second constructor argument.

    Inference is torch-free: it runs an exported ``model.onnx`` with
    ``onnxruntime`` and numpy. Requires the optional ``onnxruntime`` dependency
    (``pip install qaoa_training_pipeline[inference]``) and needs neither torch
    nor the original checkpoint.
    """

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        strict: bool = True,
        validate_input_operator: bool = True,
        rescale: Callable[[Sequence[float]], Sequence[float]] | None = None,
        qaoa_angles_function: BaseAnglesFunction | None = None,
    ) -> None:
        """Initialize the AI inference trainer.

        Args:
            model: Bundle key of the model, ``"<model>/p<p>"`` (e.g.
                ``"gcn/p3"``). Resolved to a HuggingFace repo at a pinned
                revision via the setup file and downloaded once into the local
                HF cache. See
                :func:`~qaoa_training_pipeline.inference.model_registry.available_bundles`.
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
        # Fail here, not at download time: an unverified name must not get as
        # far as resolving to some repo.
        parse_bundle_key(model)

        self.model_key = model
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

        if "model" not in config:
            raise ValueError(
                "AIInference requires 'model' in config: a bundle key such as 'gcn/p3'. "
                f"Available: {', '.join(available_bundles())}."
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
            model=str(config["model"]),
            device=str(config.get("device", "cpu")),
            strict=bool(config.get("strict", True)),
            validate_input_operator=bool(config.get("validate_input_operator", True)),
            rescale=config.get("rescale"),
            qaoa_angles_function=angles_function,
        )

    def to_config(self) -> dict:
        """Create a serializable dictionary describing the instance."""
        # Serialize the model's Hub address, never the local snapshot
        # directory: that path is specific to one machine's HF cache and would
        # not round-trip elsewhere.
        config = {
            **self.model.source(),
            "device": self.device,
            "strict": self.strict,
            "validate_input_operator": self.validate_input_operator,
            "qaoa_angles_function": self.qaoa_angles_function.__class__.__name__,
            "predictor_metadata": self.model.metadata(),
        }

        return config

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
        """Download the bundle from the Hub and wrap it in an ONNX predictor."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        self.model = OnnxQAOAPredictor.from_bundle(
            self.model_key, device=self.device, strict=self.strict
        )
