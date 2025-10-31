#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module collects all the methods to train the parameters of a QAOA circuit."""

from .functions import IdentityFunction
from .functions import FourierFunction
from .functions import FUNCTIONS

from .fixed_angle_conjecture import FixedAngleConjecture
from .models.random_regular_fit import RandomRegularDepthOneFit
from .optimized_parameter_loader import OptimizedParametersLoader
from .parameter_scanner import DepthOneScanTrainer
from .qaoa_pca import QAOAPCA
from .random_point import RandomPoint
from .recursion import RecursionTrainer
from .recursive_transition_states import RecursiveTransitionStates
from .reweighting import ReweightingTrainer
from .scipy_trainer import ScipyTrainer
from .tqa_trainer import TQATrainer
from .transfer_trainer import TransferTrainer
from .transition_states import TransitionStatesTrainer


TRAINERS = {
    "DepthOneScanTrainer": DepthOneScanTrainer,
    "FixedAngleConjecture": FixedAngleConjecture,
    "OptimizedParametersLoader": OptimizedParametersLoader,
    "QAOAPCA": QAOAPCA,
    "ScipyTrainer": ScipyTrainer,
    "RandomPoint": RandomPoint,
    "RandomRegularDepthOneFit": RandomRegularDepthOneFit,
    "RecursionTrainer": RecursionTrainer,
    "RecursiveTransitionStates": RecursiveTransitionStates,
    "ReweightingTrainer": ReweightingTrainer,
    "TQATrainer": TQATrainer,
    "TransferTrainer": TransferTrainer,
    "TransitionStatesTrainer": TransitionStatesTrainer,    
}
