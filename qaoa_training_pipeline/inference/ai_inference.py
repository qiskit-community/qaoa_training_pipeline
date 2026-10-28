#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""AI Inference trainer implementation."""

from __future__ import annotations

import importlib.util
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.framework.param_result import ParamResult
from qaoa_training_pipeline.framework.problem_params_provider import (
    ProblemParamsProvider,
)
from qaoa_training_pipeline.inference.model_registry import (
    available_bundles,
    parse_bundle_key,
)
from qaoa_training_pipeline.inference.scaling import (
    normalize_cost_op,
    undo_gamma_rescale,
)
from qaoa_training_pipeline.training.functions import IdentityFunction


# Whether the optional inference dependencies are installed, following the
# HAS_QUIMB convention of qaoa_training_pipeline.evaluation.mps_evaluator.
HAS_INFERENCE = all(
    importlib.util.find_spec(module) is not None for module in ("onnxruntime", "huggingface_hub")
)


class AIInference(ProblemParamsProvider):
    """AI-based inference for QAOA angle prediction.

    Loads a pre-trained model from the HuggingFace Hub by its ``<model>/p<p>``
    bundle key and predicts the QAOA angles for a cost operator in one forward
    pass, with no circuit evaluations.

    **The scaling round trip.** The models are scale-invariant, and
    :meth:`provide_params` is where that is arranged, in three steps:

    1. **Scale.** ``normalize_cost_op(cost_op)`` divides the operator by the
       RMS of its coefficients, so the network sees the same normalized problem
       whatever the operator's scale.
    2. **Predict.** ``predict_normalized`` runs the network on it and
       denormalizes the output by the bundle's own ``output_scale``.
    3. **Unscale.** ``undo_gamma_rescale`` divides the gammas -- and only the
       gammas -- by the same factor, undoing step 1 for the operator the caller
       actually passed.

    Both scaling steps are in :mod:`~qaoa_training_pipeline.inference.scaling`.

    Inference is torch-free: it runs an exported ``model.onnx`` with
    ``onnxruntime`` and numpy. Requires the optional ``onnxruntime`` dependency
    (``pip install qaoa_training_pipeline[inference]``) and needs neither torch
    nor the original checkpoint.
    """

    def __init__(
        self,
        model: str,
        device: str = "cpu",
    ) -> None:
        """Initialize the AI inference trainer.

        Args:
            model: Bundle key of the model, ``"<model>/p<p>"`` (e.g.
                ``"gcn/p3"``). Resolved to a HuggingFace repo at a pinned
                revision via the setup file and downloaded once into the local
                HF cache. See
                :func:`~qaoa_training_pipeline.inference.model_registry.available_bundles`.
            device: Device for inference ("cpu", "cuda", "cuda:<index>" or
                "gpu"). An unknown device is rejected rather than quietly
                falling back to the CPU.
        """
        if not HAS_INFERENCE:
            raise ImportError(
                f"{self.__class__.__name__} requires onnxruntime and huggingface_hub. "
                "Please install the optional inference dependencies with "
                "`pip install qaoa_training_pipeline[inference]`."
            )

        # The angles function defines the basis a trainer *optimizes* in; this
        # provider runs no optimization, so it is the identity.
        super().__init__(qaoa_angles_function=IdentityFunction())
        parse_bundle_key(model)

        self.model_key = model
        self.device = str(device)
        self.model = None

        self.load_model()

    def provide_params(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
    ) -> ParamResult:
        """Return QAOA angles by running inference on the loaded model.

        ``cost_op`` must be in the max-cut convention described on the class.
        """
        start = time()

        if self.model is None:
            raise RuntimeError("AI inference model config was not loaded.")

        # Scale -> predict -> unscale
        normalized_cost_op, rescale_a = normalize_cost_op(cost_op)

        raw_angles = self.model.predict_normalized(
            normalized_cost_op,
            rescale_a,
            mixer=mixer,
            ansatz_circuit=ansatz_circuit,
            initial_state=initial_state,
        )

        qaoa_angles = undo_gamma_rescale(
            raw_angles, p=self.model.output_dim // 2, rescale_a=rescale_a
        )

        energy = None

        result = ParamResult(qaoa_angles, time() - start, self, energy)
        result["ai_inference"] = {
            **self.model.source(),
            "device": self.device,
            "predictor_metadata": self.model.metadata(),
        }
        return result

    @classmethod
    def from_config(cls, config: dict) -> "AIInference":
        """Return an instance of the class based on a config."""
        config = dict(config)

        if "model" not in config:
            raise ValueError(
                "AIInference requires 'model' in config: a bundle key such as 'gcn/p3'. "
                f"Available: {', '.join(available_bundles())}."
            )

        return cls(
            model=str(config["model"]),
            device=str(config.get("device", "cpu")),
        )

    def to_config(self) -> dict:
        """Create a serializable dictionary describing the instance."""
        # The model's Hub address, never the local snapshot directory: that
        # path is specific to one machine's HF cache.
        return {
            **self.model.source(),
            "device": self.device,
            "predictor_metadata": self.model.metadata(),
        }

    def load_model(self) -> None:
        """Download the bundle from the Hub and wrap it in an ONNX predictor."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        self.model = OnnxQAOAPredictor.from_bundle(self.model_key, device=self.device)
