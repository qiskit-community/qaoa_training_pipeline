#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module collects all the methods to train the parameters of a QAOA circuit."""
from .models.random_regular_fit import RandomRegularDepthOneFit
from .optimized_parameter_loader import OptimizedParametersLoader
from .parameter_scanner import DepthOneScanTrainer, DepthOneGammaScanTrainer
from .qaoa_pca import QAOAPCA
from .random_point import RandomPoint
from .recursion import RecursionTrainer
from .recursive_transition_states import RecursiveTransitionStates
from .reweighting import ReweightingTrainer
from .scipy_trainer import ScipyTrainer
from .transfer_trainer import TransferTrainer
from .transition_states import TransitionStatesTrainer


PARAMS_PROVIDERS = {
    "OptimizedParametersLoader": OptimizedParametersLoader,
    "RandomPoint": RandomPoint,
    "TransferTrainer": TransferTrainer,
    "RandomRegularDepthOneFit": RandomRegularDepthOneFit,
    "ScipyTrainer": ScipyTrainer,
    "DepthOneGammaScanTrainer": DepthOneGammaScanTrainer,
}

PIPELINE_COMPONENTS = {
    "RecursionTrainer": RecursionTrainer,
    "RecursiveTransitionStates": RecursiveTransitionStates,
    "ReweightingTrainer": ReweightingTrainer,
    "ScipyTrainer": ScipyTrainer,
    "TransitionStatesTrainer": TransitionStatesTrainer,
    "QAOAPCA": QAOAPCA,
}

PROBLEM_PARAMS_PROVIDERS = {
    "DepthOneScanTrainer": DepthOneScanTrainer,
    "TransferTrainer": TransferTrainer,
}
