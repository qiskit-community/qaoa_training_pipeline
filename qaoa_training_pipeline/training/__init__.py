#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module collects all the methods to train the parameters of a QAOA circuit."""
from .lrqaoa_trainer import LRQAOATrainer
from .models.random_regular_fit import RandomRegularDepthOneFit
from .optimized_parameter_loader import OptimizedParametersLoader
from .parameter_scanner import DepthOneGammaScanTrainer, DepthOneScanTrainer
from .qaoa_pca import QAOAPCA
from .random_point import RandomPoint
from .recursion import RecursionTrainer
from .recursive_transition_states import RecursiveTransitionStates
from .reweighting import ReweightingTrainer
from .scipy_trainer import ScipyTrainer
from .tqa_trainer import TQATrainer
from .transfer_trainer import TransferTrainer
from .transition_states import TransitionStatesTrainer

__all__ = [
    "RandomRegularDepthOneFit",
    "OptimizedParametersLoader",
    "DepthOneScanTrainer",
    "DepthOneGammaScanTrainer",
    "QAOAPCA",
    "RandomPoint",
    "RecursionTrainer",
    "RecursiveTransitionStates",
    "ReweightingTrainer",
    "ScipyTrainer",
    "TQATrainer",
    "LRQAOATrainer",
    "TransferTrainer",
    "TransitionStatesTrainer",
]

PARAMS_PROVIDERS = {
    "OptimizedParametersLoader": OptimizedParametersLoader,
    "RandomPoint": RandomPoint,
    "TransferTrainer": TransferTrainer,
    "RandomRegularDepthOneFit": RandomRegularDepthOneFit,
    "DepthOneGammaScanTrainer": DepthOneGammaScanTrainer,
}

PIPELINE_COMPONENTS = {
    "RecursionTrainer": RecursionTrainer,
    "RecursiveTransitionStates": RecursiveTransitionStates,
    "ReweightingTrainer": ReweightingTrainer,
    "ScipyTrainer": ScipyTrainer,
    "TransitionStatesTrainer": TransitionStatesTrainer,
    "QAOAPCA": QAOAPCA,
    "TQATrainer": TQATrainer,
    "LRQAOATrainer": LRQAOATrainer,
}

PROBLEM_PARAMS_PROVIDERS = {
    "DepthOneScanTrainer": DepthOneScanTrainer,
    "TransferTrainer": TransferTrainer,
}
