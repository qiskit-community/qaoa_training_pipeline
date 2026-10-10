#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""AI Inference trainer implementation."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.framework import ParamResult, ProblemParamsProvider
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

    The cost operator passed to :meth:`provide_params` must use the max-cut
    convention the models were trained on, ``H = -0.5 * sum_e w_e Z_i Z_j``
    (``graph_to_operator(graph, pre_factor=-0.5)``). See
    :data:`~qaoa_training_pipeline.inference.feature_extractor.MAX_CUT_PRE_FACTOR`:
    an operator built with another pre-factor is not an error, it just predicts
    the wrong gammas.

    Inference is torch-free: it runs an exported ``model.onnx`` with
    ``onnxruntime`` and numpy. Requires the optional ``onnxruntime`` dependency
    (``pip install qaoa_training_pipeline[inference]``) and needs neither torch
    nor the original checkpoint.
    """

    def __init__(
        self,
        model: str,
        device: str = "cpu",
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
            device: Device for inference ("cpu", "cuda", "cuda:<index>" or
                "gpu"). An unknown device is rejected rather than quietly
                falling back to the CPU.
            rescale: Optional post-processing hook applied to the predicted
                angles *after* the config's own denormalization, which always
                runs. Receives the denormalized angle list and must return a
                same-length sequence. Use e.g. ``lambda a: [x * 2 for x in a]``
                to map betas from ``[0, π/2]`` onto ``[0, π]``, or pass ``None``
                (default) for the config's scaling alone. To get the raw network
                output instead, call
                :meth:`~qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor.predict`
                with ``denormalize=False``.
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
        """Return QAOA angles by running inference on the loaded model.

        ``cost_op`` must be in the max-cut convention described on the class.
        """
        start = time()

        if self.model is None:
            raise RuntimeError("AI inference model config was not loaded.")

        # The rescale hook runs *after* the config's denormalization, never
        # instead of it: denormalization undoes the gamma rescaling that makes
        # the model scale-invariant, so skipping it would hand the hook raw
        # network output and quietly produce gammas on the wrong scale.
        qaoa_angles = self.model.predict(
            cost_op,
            mixer=mixer,
            ansatz_circuit=ansatz_circuit,
            initial_state=initial_state,
        )

        if self.rescale is not None:
            rescaled = list(self.rescale(qaoa_angles))
            if len(rescaled) != len(qaoa_angles):
                raise ValueError(
                    f"rescale hook changed the number of angles: "
                    f"{len(qaoa_angles)} -> {len(rescaled)}."
                )
            qaoa_angles = [float(value) for value in rescaled]

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

        if config.get("rescale_hook_dropped") and config.get("rescale") is None:
            warnings.warn(
                "This config was serialized from a provider with a rescale hook, which is a "
                "callable and could not be stored. The rebuilt provider has no hook, so its "
                "angles will differ from the original's. Pass rescale= yourself, or fold the "
                "scaling into qaoa_angles_function, which does serialize.",
                UserWarning,
                stacklevel=2,
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
            rescale=config.get("rescale"),
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

        if self.rescale is not None:
            # A callable has no JSON form, so the hook cannot be carried. Record
            # that it was dropped rather than emitting a config that looks
            # complete; from_config warns when it reads this back, which is the
            # point at which the angles actually change.
            config["rescale_hook_dropped"] = True

        return config

    def load_model(self) -> None:
        """Download the bundle from the Hub and wrap it in an ONNX predictor."""
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        self.model = OnnxQAOAPredictor.from_bundle(self.model_key, device=self.device)
