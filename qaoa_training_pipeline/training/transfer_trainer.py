#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base trainer interface for trainers that rely on a data base."""

from time import time
from typing import Dict, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.pre_processing.feature_extraction import (
    BaseFeatureExtractor,
    FEATURE_EXTRACTORS,
)
from qaoa_training_pipeline.pre_processing.feature_matching import (
    BaseFeatureMatcher,
    TrivialFeatureMatcher,
    FEATURE_MATCHERS,
)
from qaoa_training_pipeline.pre_processing.angle_aggregation import (
    BaseAngleAggregator,
    TrivialAngleAggregator,
    ANGLE_AGGREGATORS,
)
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.data_loading import BaseDataLoader, DATA_LOADERS
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
        self._data_loader = data_loader
        self._data: dict = self._data_loader()
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

    @property
    def minimization(self):
        """Raises a warning as a transfer neither minimizes nor maximizes."""
        raise ValueError(f"{self.__class__.__name__} neither minimizes nor maximizes.")

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
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
            data_key = self._feature_matcher(features, set(self._data.keys()))
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

    def validate_data(self):
        """Validation that the input data is a dict."""
        if not isinstance(self._data, dict):
            raise TypeError(f"{self.__class__.__name__} needs data as a dict.")

    def parse_train_kwargs(self, args_str: Optional[str] = None) -> dict:
        """Extract training key word arguments from a string.

        The input args are only the number of repetitions. The input should be of the form
        `reps:value`.
        """
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key in ["reps"]:
                train_kwargs[key] = int(val)
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs

    def to_config(self):
        """Produce a config for the transfer trainer."""
        config = {"trainer_name": self.__class__.__name__}
        config["data_loader"] = self._data_loader.__class__.__name__
        config["data_loader_init"] = self._data_loader.to_config()

        config["feature_extractor"] = self._feature_extractor.__class__.__name__
        config["feature_extractor_init"] = self._feature_extractor.to_config()

        config["feature_matcher"] = self._feature_matcher.__class__.__name__
        config["feature_matcher_init"] = self._feature_matcher.to_config()

        config["angle_aggregator"] = self._angle_aggregator.__class__.__name__
        config["angle_aggregator_init"] = self._angle_aggregator.to_config()

        config["evaluator"] = "None"
        if self._evaluator is not None:
            config["evaluator"] = self._evaluator.__class__.__name__
            config["evaluator_init"] = self._evaluator.to_config()

            if "name" in config["evaluator_init"]:
                del config["evaluator_init"]["name"]

        return config

    @classmethod
    def from_config(cls, config: Dict) -> "TransferTrainer":
        """Creat a class from a config."""
        data_loader_cls = DATA_LOADERS[config["data_loader"]]
        feature_extractor_cls = FEATURE_EXTRACTORS[config["feature_extractor"]]
        feature_matcher_cls = FEATURE_MATCHERS[config["feature_matcher"]]
        angle_aggregator_cls = ANGLE_AGGREGATORS[config["angle_aggregator"]]

        evaluator = None
        if config["evaluator"] in EVALUATORS:
            evaluator_cls = EVALUATORS[config["evaluator"]]
            evaluator = evaluator_cls.from_config(config["evaluator_init"])

        return cls(
            data_loader=data_loader_cls.from_config(config["data_loader_init"]),
            feature_extractor=feature_extractor_cls.from_config(config["feature_extractor_init"]),
            feature_matcher=feature_matcher_cls.from_config(config["feature_matcher_init"]),
            angle_aggregator=angle_aggregator_cls.from_config(config["angle_aggregator_init"]),
            evaluator=evaluator,
        )
