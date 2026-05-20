#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import glob
import json
from abc import ABC, abstractmethod

import numpy as np

from qaoa_training_pipeline.pipeline_component import PipelineComponent 
from qaoa_training_pipeline.training.param_result import ParamResult


class ParamsProvider(PipelineComponent):
    """An interface that parameter providers need to follow.
    """

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def provide_params(
        self,
        folder: str | None = None,
        file_pattern: str | None = None,
    ) -> ParamResult:
        """Return a ParamResult object containing the parameters.
        """
        raise NotImplementedError("Sub-classes must implement `provide_params`.")

    
