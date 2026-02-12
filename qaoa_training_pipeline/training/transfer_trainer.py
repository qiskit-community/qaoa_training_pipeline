#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base trainer interface for trainers that rely on a data base."""

from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.pre_processing.angle_aggregation import (
    ANGLE_AGGREGATORS,
    BaseAngleAggregator,
    TrivialAngleAggregator,
)
from qaoa_training_pipeline.pre_processing.feature_extraction import (
    FEATURE_EXTRACTORS,
    GraphFeatureExtractor,
)
from qaoa_training_pipeline.pre_processing.feature_matching import (
    FEATURE_MATCHERS,
    BaseFeatureMatcher,
    TrivialFeatureMatcher,
)
from qaoa_training_pipeline.training.base_trainer import BaseTrainer
from qaoa_training_pipeline.training.data_loading import DATA_LOADERS, BaseDataLoader
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
        feature_extractor: GraphFeatureExtractor,
        feature_matcher: BaseFeatureMatcher | None = None,
        angle_aggregator: BaseAngleAggregator | None = None,
        evaluator: BaseEvaluator | None = None,
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
        # values are QAOA angles corresponding to those features. The values are
        # lists that should be valid inputs to angle aggregator functions. Therefore,
        # the values may be a list of QAOA angles (requires trivial aggregation) or
        # a list of lists of QAOA angles (requires aggregation such as `AverageAngleAggregator`).
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

    # pylint: disable=too-many-positional-arguments
    def train(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
        params0: list[float] | None = None,
        qaoa_depth: int | None = None,
    ) -> ParamResult:
        """Performs the training."""

        if qaoa_depth is None:
            raise ValueError(
                f"class {self.__class__.__name__} requires parameter qaoa_depth to be specified."
            )

        if mixer is not None:
            raise NotImplementedError("Custom mixers are not yet supported.")

        if initial_state is not None:
            raise NotImplementedError("Custom initial states are not yet supported.")

        if ansatz_circuit is not None:
            raise NotImplementedError("Custom Ansatze are not yet supported.")

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
        qaoa_angles = self._angle_aggregator(self._data[data_key]["qaoa_angles"])

        if self._evaluator is not None:
            energy = self._evaluator.evaluate(
                cost_op=cost_op,
                params=qaoa_angles,
            )
        else:
            energy = None

        if len(qaoa_angles) // 2 != qaoa_depth:
            raise ValueError(
                f"Data in {self.__class__.__name__} returned angles for the wrong QAOA depth."
                "Check the underlying data used in the transfer."
            )

        result = ParamResult(qaoa_angles, time() - start, self, energy)
        result["data_key"] = data_key

        return result

    def validate_data(self):
        """Validation that the input data is a dict."""
        if not isinstance(self._data, dict):
            raise TypeError(f"{self.__class__.__name__} needs data as a dict.")

    def parse_train_kwargs(self, args_str: str | None = None) -> dict:
        """Extract training key word arguments from a string.

        The input args are only the number of repetitions. The input should be of the form
        `reps:value`.
        """
        train_kwargs = dict()
        for key, val in self.extract_train_kwargs(args_str).items():
            if key in ["qaoa_depth"]:
                train_kwargs[key] = int(val)
            else:
                raise ValueError("Unknown key in provided train_kwargs.")

        return train_kwargs

    def to_config(self):
        """Produce a config for the transfer trainer."""
        config: dict[str, str | dict] = {"trainer_name": self.__class__.__name__}
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
    def from_config(cls, config: dict) -> "TransferTrainer":
        """Create a class from a config."""
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
