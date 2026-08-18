#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Classes to evaluate the energy of a QAOA circuit."""

from .efficient_depth_one import EfficientDepthOneEvaluator
from .light_cone import LightConeEvaluator
from .mps_aer import MPSAerEvaluator
from .mps_evaluator import MPSEvaluator
from .mps_sample_evaluator import MPSSampleEvaluator
from .pauli_propagation import PPEvaluator
from .qpu_sample_evaluator import QPUSampleEvaluator
from .statevector_evaluator import StatevectorEvaluator

__all__ = [
    "EfficientDepthOneEvaluator",
    "LightConeEvaluator",
    "MPSAerEvaluator",
    "MPSEvaluator",
    "PPEvaluator",
    "StatevectorEvaluator",
    "EVALUATORS",
    "QPUSampleEvaluator",
    "MPSSampleEvaluator",
]

EVALUATORS = {
    "EfficientDepthOneEvaluator": EfficientDepthOneEvaluator,
    "LightConeEvaluator": LightConeEvaluator,
    "MPSEvaluator": MPSEvaluator,
    "StatevectorEvaluator": StatevectorEvaluator,
    "PPEvaluator": PPEvaluator,
    "MPSAerEvaluator": MPSAerEvaluator,
    "MPSSampleEvaluator": MPSSampleEvaluator,
    "QPUSampleEvaluator": QPUSampleEvaluator,
}
