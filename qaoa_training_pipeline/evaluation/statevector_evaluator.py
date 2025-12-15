#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Statevector-based QAOA evaluator."""

from typing import Dict, Optional

from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qaoa_training_pipeline.evaluation.aer_interface import AerEvaluator


class StatevectorEvaluator(AerEvaluator):
    """Evaluates the energy of a QAOA circuit with Qiskit's StatevectorSimulator.

    This evaluator naturally does not scale to large problem sizes but is useful
    when working with small-scale problem instances.
    """

    def __init__(
        self, statevector_init_args: Optional[Dict] = None, device: Optional[str] = None
    ) -> None:
        """Initialize the statevector evaluator.

        Args:
            statevector_init_args: The arguments to initialize the StatevectorSimulator with.
                                   Can include "device": "GPU" for GPU acceleration.
            device: The device to use for the simulation. If provided, it overrides the device
                    in statevector_init_args.
        """
        self._init_args = statevector_init_args or {}
        if device:
            self._init_args["device"] = device

        device = self._init_args.get("device")
        if device is not None and device != "GPU":
            raise ValueError(
                f"Invalid device '{device}'. Only 'GPU' is supported for device parameter, "
                "or None/omit for CPU."
            )

        estimator = AerEstimator(
            options={
                "backend_options": {
                    "method": "statevector",
                    **({"device": device} if device else {}),
                },
                **{k: v for k, v in self._init_args.items() if k != "device"},
            }
        )

        super().__init__(estimator=estimator)

    @classmethod
    def parse_init_kwargs(cls, init_kwargs: Optional[str] = None) -> dict:
        """Parse initialization kwargs.

        If you are using the GPU, the input string should be "GPU".
        Any other string will be ignored.

        Args:
            init_kwargs: The initialization arguments as a string.

        Returns:
            dict: The parsed initialization arguments.
        """
        if init_kwargs is None:
            return dict()

        if init_kwargs.upper() == "GPU":
            return {"device": "GPU"}

        return {}

    def to_config(self) -> dict:
        config = super().to_config()
        config["statevector_init_args"] = self._init_args

        return config

    @classmethod
    def from_config(cls, config: dict) -> "StatevectorEvaluator":
        """Initialize the evaluator from a configuration dictionary."""
        return cls(**config)
