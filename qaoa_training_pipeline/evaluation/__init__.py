#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module collects all the functions to evaluate the energy of a QAOA circuit."""

from .efficient_depth_one import EfficientDepthOneEvaluator
from .light_cone import LightConeEvaluator
from .mps_evaluator import MPSEvaluator
from .pauli_propagation import PPEvaluator
from .statevector_evaluator import StatevectorEvaluator


EVALUATORS = {
    "EfficientDepthOneEvaluator": EfficientDepthOneEvaluator,
    "LightConeEvaluator": LightConeEvaluator,
    "MPSEvaluator": MPSEvaluator,
    "StatevectorEvaluator": StatevectorEvaluator,
    "PPEvaluator": PPEvaluator,
}
