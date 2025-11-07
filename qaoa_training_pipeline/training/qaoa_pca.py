#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trainer that leverages PCA to reduce the dimensionality of QAOA parameter optimization."""

from typing import Any, Dict, Optional
import numpy as np

from qaoa_training_pipeline.evaluation import EVALUATORS
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.data_loading import BaseDataLoader
from qaoa_training_pipeline.training.functions import PCAFunction
from qaoa_training_pipeline.training.data_loading import DATA_LOADERS
from qaoa_training_pipeline.training.scipy_trainer import ScipyTrainer


class QAOAPCA(ScipyTrainer):
    """Reduce the dimensionality of QAOA parameters with PCA.

    Example usage:
    ```
    trainer = QAOAPCA(LoadFromJson("my_data.json"), 2, StatevectorEvaluator())

    params0 = trainer.qaoa_angles_function.transform(qaoa_angles)
    result = trainer.train(cost_operator, params0)
    ```
    In the example, above the `qaoa_angles` are good angles that come from a similar graph.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        data_loader: BaseDataLoader,
        num_components: int,
        evaluator: BaseEvaluator,
        minimize_args: Optional[Dict[str, Any]] = None,
        energy_minimization: bool = False,
    ):
        """Setup the QAOAPCA trainer.

        The fitting of the principal components and the data loading happen at instance
        initialization. The `train` method is simply the train method of the ScipyTrainer.
        Therefore the `params0` must be given in principal component form.

        Args:
            data_loader: A callable method that will populate the data that the trainer
                relies on to set the QAOA parameters.
            num_components: The number of principal components to use.
            evaluator: An instance of `BaseEvaluator` which will evaluate the enrgy
                of the QAOA circuit.
            minimize_args: Arguments that will be passed to SciPy's `minimize`.
            energy_minimization: Allows us to switch between minimizing the energy or maximizing
                the energy. The default and assumed convention in this repository is to
                maximize the energy.

        """
        self._data_loader = data_loader
        self._data = self._data_loader()
        self.validate_data()

        # Now that data is loaded we can create and fit a PCA trainer.
        self._num_components = num_components
        qaoa_angles_function = PCAFunction(num_components)
        qaoa_angles_function.fit(self._data)

        # Setup the rest of the SciPy trainer.
        super().__init__(
            evaluator=evaluator,
            minimize_args=minimize_args,
            energy_minimization=energy_minimization,
            qaoa_angles_function=qaoa_angles_function,
        )

    def validate_data(self):
        """Data for QAOA PCA must be in the form of a 2D."""
        if not isinstance(self._data, np.ndarray):
            self._data = np.array(self._data)

        if len(self._data.shape) != 2:
            raise ValueError("Data for {self.__class__.__name__} must be a 2D numpy array.")

    def to_config(self):
        """Create a config file from a traininer instance."""
        config = super().to_config()
        if "name" in config["evaluator_init"]:
            del config["evaluator_init"]["name"]

        config["num_components"] = self._num_components

        config["data_loader"] = self._data_loader.__class__.__name__
        config["data_loader_init"] = self._data_loader.to_config()

        return config

    @classmethod
    def from_config(cls, config):
        """Create a PCA trainer based on a config."""
        evaluator_cls = EVALUATORS[config["evaluator"]]
        data_loader_cls = DATA_LOADERS[config["data_loader"]]

        return cls(
            data_loader=data_loader_cls.from_config(config["data_loader_init"]),
            num_components=config["num_components"],
            evaluator=evaluator_cls.from_config(config["evaluator_init"]),
            minimize_args=config.get("minimize_args", None),
            energy_minimization=config.get("energy_minimization", None),
        )
