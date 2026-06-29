#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Framework module: core abstract base classes for the QAOA training pipeline."""

from .param_result import ParamResult
from .params_provider import ParamsProvider
from .problem_params_provider import ProblemParamsProvider
from .pipeline_component import PipelineComponent
