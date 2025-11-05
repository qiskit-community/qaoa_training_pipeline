#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base trainer interface for trainers that rely on a data base."""

from abc import abstractmethod
from time import time
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.pre_processing.feature_extraction import BaseFeatureExtractor
from qaoa_training_pipeline.pre_processing.feature_matching import (
    BaseFeatureMatcher,
    TrivialFeatureMatcher,
)
from qaoa_training_pipeline.pre_processing.angle_aggregation import (
    BaseAngleAggregator,
    TrivialAngleAggregator,
)
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.data_loading import BaseDataLoader
from qaoa_training_pipeline.training.param_result import ParamResult


class TransferTrainer(BaseTrainer):
    """A class to transfer parameters from a collection of parameters.

    The workflow allows for the following steps:

    1. Compute a set of features of the given problem instance.
    2. Match these features to keys in a database of QAOA angles. Here, there might be
       multiple angles in the database that match with the features.
    3. Aggregate the selected angles into a set of single QAOA angles.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        data_loader: BaseDataLoader,
        feature_extractor: BaseFeatureExtractor,
        feature_matcher: BaseFeatureMatcher = None,
        angle_aggregator: BaseAngleAggregator = None,
        evaluator: Optional[BaseEvaluator] = None,
    ):
        """Setup a class to train based on existing data.

        Args:
            data_loader: A callable that populates the data that the trainer relies on to set
                the QAOA parameters.
            feature_extractor: A method to extract features from a problem instance. This instance
                is callable and should also be what is used to generate the keys in the data
                loaded by the `data_loader`.
            feature_matcher: An object responsible for matching the extracted features to a set
                of QAOA angles stored in the keys of the data.
            angle_aggregator: Aggregates the angles.
            evaluator: An energy evaluator. Note that not all sub-classes may require an
                energy evaluator.
        """
        super().__init__(evaluator)

        # The data should be a dictionary in which the keys are features and the
        # values are QAOA angles corresponding to those features. The values may
        # be a list.
        self._db_extractor = data_loader
        self._data: dict = data_loader()
        self.validate_data()

        # Extracts features of cost operators.
        self._feature_extractor = feature_extractor

        # Matches the features in the data of known angles.
        if feature_matcher is not None:
            self._feature_matcher = feature_matcher
        else:
            self._feature_matcher = TrivialFeatureMatcher()

        # Aggregates the matched angles in the data if data have more than one set of angles.
        if angle_aggregator is not None:
            self._angle_aggregator = angle_aggregator
        else:
            self._angle_aggregator = TrivialAngleAggregator()

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    @abstractmethod
    def train(
        self,
        cost_op: SparsePauliOp,
        qaoa_depth: int,
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        **kwargs,
    ) -> ParamResult:
        """Performs the training."""

        start = time()

        # 1. Extract features of the cost operator.
        features = self._feature_extractor(cost_op, qaoa_depth)

        # 2. Match these features to keys in the data of good angles.
        if self._feature_matcher is not None:
            data_key = self._feature_matcher(features, self._data)
        else:
            data_key = features

        # 3. The values under self._data[data_key] may be a 2D array of known
        # angles that correspond to the features. One dimension is the dimension
        # of the angles while the second one can correspond to multiple cost operators
        # that match to the same feature key.
        qaoa_angles = self._angle_aggregator(self._data[data_key])

        if self._evaluator is not None:
            energy = self._evaluator.evaluate(cost_op, qaoa_angles)
        else:
            energy = "NA"

        return ParamResult(qaoa_angles, time() - start, self, energy)

    @abstractmethod
    def validate_data(self):
        """A hook to validate the format of self._data."""
        raise NotImplementedError("Sub-classes must implement `validate_data`.")

    def to_config(self):
        config = super().to_config()
        config["db_extractor"] = self._db_extractor.to_config()

        return config
