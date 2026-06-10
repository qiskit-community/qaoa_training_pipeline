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

    def extract_runtime_kwargs(self, kwargs_str: str | None = None) -> dict:
        """Method to parse keyword arguments passed when using the pipeline
        from the command line.

        The kwarg string is given, e.g., in form `k1:v1:k2:v2`. If the value is
        a list then the values in the list must be spaced by a `/`, for example,
        `params0:1.234/4.56`.
        """
        if kwargs_str is None:
            return dict()

        items = kwargs_str.split(":")

        if len(items) % 2 != 0:
            raise ValueError(
                f"Malformed keyword arguments {kwargs_str}: should be k1:v1:k2:v2_...."
            )

        return {items[idx]: items[idx + 1] for idx in range(0, len(items), 2)}
