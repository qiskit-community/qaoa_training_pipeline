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

from qaoa_training_pipeline.framework import ParamResult, ProblemParamsProvider
from qaoa_training_pipeline.inference.model_registry import (
    available_bundles,
    parse_bundle_key,
)
from qaoa_training_pipeline.inference.scaling import (
    normalize_cost_op,
    undo_gamma_rescale,
)
from qaoa_training_pipeline.training.functions import (
    BaseAnglesFunction,
    IdentityFunction,
)


# Whether the optional inference dependencies are installed, following the
# HAS_QUIMB convention of qaoa_training_pipeline.evaluation.mps_evaluator.
HAS_INFERENCE = all(
    importlib.util.find_spec(module) is not None for module in ("onnxruntime", "huggingface_hub")
)


class AIInference(ProblemParamsProvider):
    """AI-based inference for QAOA angle prediction.

    Loads a pre-trained AI model and predicts QAOA angles from a cost
    operator. The referenced model config declares which training setting
    the checkpoint was produced under.

    **The scaling round trip.** The models are scale-invariant, and
    :meth:`provide_params` is where that is arranged, in three steps:

    1. **Scale.** ``normalize_cost_op(cost_op)`` divides the operator by the
       RMS of its coefficients, so the network sees the same normalized problem
       whatever the operator's scale.
    2. **Predict.** ``predict_normalized`` runs the network on it and
       denormalizes the output by the bundle's own ``output_scale``.
    3. **Unscale.** ``undo_gamma_rescale`` divides the gammas -- and only the
       gammas -- by the same factor, undoing step 1 for the operator the caller
       actually passed. This step is unconditional: a provider's job is angles
       for the operator it was handed. A bundle that set
       ``denormalize_output: false`` would therefore get its raw output
       un-rescaled here, which is meaningless -- but no published bundle sets
       it, and raw output is reached through
       :meth:`~qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor.predict_normalized`
       with ``denormalize=False``, which skips step 2's scaling.

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
        qaoa_angles_function: BaseAnglesFunction | None = None,
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
            qaoa_angles_function: Function transforming the predicted angles to
                a different basis before use. Defaults to
                :class:`IdentityFunction` (no transformation). This is the way
                to post-process predicted angles -- a scaling, a change of
                basis -- because unlike a bare callable it serializes, so
                :meth:`to_config` / :meth:`from_config` round-trips keep it.
                For the raw network output instead, call
                :meth:`~qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor.predict_normalized`
                with ``denormalize=False``.
        """
        if not HAS_INFERENCE:
            raise ImportError(
                f"{self.__class__.__name__} requires onnxruntime and huggingface_hub. "
                "Please install the optional inference dependencies with "
                "`pip install qaoa_training_pipeline[inference]`."
            )

        super().__init__(qaoa_angles_function=qaoa_angles_function or IdentityFunction())
        # Fail here, not at download time: an unverified name must not get as
        # far as resolving to some repo.
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
            qaoa_angles_function=angles_function,
        )

    def to_config(self) -> dict:
        """Create a serializable dictionary describing the instance."""
        # Serialize the model's Hub address, never the local snapshot
        # directory: that path is specific to one machine's HF cache and would
        # not round-trip elsewhere.
        # The function's own config carries its init arguments (e.g. a Fourier
        # depth); from_config feeds them back as keyword arguments, so dropping
        # them would rebuild a differently-configured function.
        angles_function_config = dict(self.qaoa_angles_function.to_config())
        angles_function_config.pop("function_name", None)

        config = {
            **self.model.source(),
            "device": self.device,
            "qaoa_angles_function": self.qaoa_angles_function.__class__.__name__,
            "qaoa_angles_function_init": angles_function_config,
            "predictor_metadata": self.model.metadata(),
        }

        return config

    def load_model(self) -> None:
        """Download the bundle from the Hub and wrap it in an ONNX predictor."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        self.model = OnnxQAOAPredictor.from_bundle(self.model_key, device=self.device)
