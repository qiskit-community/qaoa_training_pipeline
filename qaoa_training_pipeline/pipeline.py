#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""QAOA training pipeline orchestration and execution.

This module defines the Pipeline class, which orchestrates the execution of an initial
ParamsProvider and multiple PipelineComponents to generate QAOA angles. It allows
multi-stage angle generation strategies. 
"""


from qaoa_training_pipeline.qaoa_training_pipeline.params_provider import ParamsProvider
from qaoa_training_pipeline.qaoa_training_pipeline.pipeline_component import PipelineComponent


class Pipeline:
    """A pipeline is a class that is formed by a list of PipelineComponents and
    one ParamsProvider, and is responsible for receiving and processing user input on what
    PipelineComponents and ParamsProvider to use to retrieve the desired QAOA angles
    """

    def __init__(
        self,
        pipeline_components: list[PipelineComponent] | None = None,
        params_provider: ParamsProvider | None = None,
    ):
        self._pipeline_components = pipeline_components or []
        self._params_provider = params_provider
