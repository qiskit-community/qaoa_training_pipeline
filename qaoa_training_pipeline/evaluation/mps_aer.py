#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluator based on Qiskit Aer's MPS."""

from typing import Dict, Optional

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2

from qaoa_training_pipeline.evaluation.aer_interface import AerEvaluator


class MPSAerEvaluator(AerEvaluator):
    """Evaluates the energy of a QAOA circuit with Qiskit Aer's MPS evaluator.

    This evaluator can scale to large problem sizes by changing the bond dimension.
    """

    def __init__(self, mps_init_args: Optional[Dict] = None) -> None:
        """Initialize the MPS evaluator.

        Args:
            mps_init_args: The arguments to initialize the MPS with. If the needed
                arguments are missing we assume a default bond dimension of 64 and
                a MPS truncation threshold of 1e-5.
        """

        self._init_args = mps_init_args or {}

        if "matrix_product_state_max_bond_dimension" not in self._init_args:
            self._init_args["matrix_product_state_max_bond_dimension"] = 64

        if "matrix_product_state_truncation_threshold" not in self._init_args:
            self._init_args["matrix_product_state_truncation_threshold"] = 1e-5

        if "method" not in self._init_args:
            self._init_args["method"] = "matrix_product_state"

        estimator = EstimatorV2(mode=AerSimulator(**self._init_args))
        super().__init__(estimator)

    def to_config(self) -> dict:
        config = super().to_config()
        config["mps_init_args"] = self._init_args

        return config

    @classmethod
    def from_config(cls, config: dict) -> "MPSAerEvaluator":
        """Initialize the evaluator from a configuration dictionary."""
        return cls(**config)
