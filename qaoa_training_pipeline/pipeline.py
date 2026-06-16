#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""QAOA training pipeline orchestration and execution.

This module defines the Pipeline class, which orchestrates the execution of QAOA parameter
generation and optimization workflows via a combination of an initial ParamsProvider with
a sequence of PipelineComponents to allow the creation of complex, multi-stage QAOA angles 
generation strategies. The class is flexible and allows users to compose QAOA training 
pipelines that include different optimization and heuristic strategies tailored to specific 
problem requirements.

Example:
    1. ParamsProvider provides initial QAOA angles
    2. PipelineComponent 1 applies a heuristic
    3. PipelineComponent 2 refines the QAOA angles
    4. Final optimized angles are returned and the energy is evaluated
"""


from collections import defaultdict
from qaoa_training_pipeline.qaoa_training_pipeline.training import PIPELINE_COMPONENTS, PARAMS_PROVIDERS
from qaoa_training_pipeline.qaoa_training_pipeline.pipeline_component import PipelineComponent
from qaoa_training_pipeline.qaoa_training_pipeline.params_provider import ParamsProvider

class Pipeline:
    """Orchestrates the QAOA angles training pipeline. The Pipeline class manages the training 
    of QAOA angles by combining an initial ParamsProvider with a sequence of PipelineComponents.
    It provides a flexible framework for building complex angles optimization pipelines 
    through the stacking of PipelineComponents that carry out different optimization methods.
    
    The pipeline executes in stages:
        1. Initial parameter generation via ParamsProvider
        2. Sequential refinement through each PipelineComponent
        3. Each component receives the output of the previous stage
    
    Attributes:
        _pipeline_components: List of PipelineComponent instances that sequentially
        optimize angles. Components are executed in order.

        _params_provider: Initial ParamsProvider that generates starting angles.
    """

    def __init__(
        self,
        pipeline_components: list[PipelineComponent] | None = None,
        params_provider: ParamsProvider | None = None,
    ):
        """Initialize the QAOA training pipeline.
        
        Args:
            pipeline_components: Optional list of PipelineComponent instances to execute
            sequentially. Each component refines the angles from the previous stage.
            If None, creates an empty list.

            params_provider: Optional ParamsProvider for generating initial parameters.
            If None, initial parameters must be provided when executing
            the pipeline.
        
        Note:
            At least one of pipeline_components or params_provider should be provided
            for the pipeline to be useful. An empty pipeline with no components and
            no provider will not perform any operations.
        """
        self._pipeline_components = pipeline_components or []
        self._params_provider = params_provider

    @classmethod
    def from_config(cls, config: dict, input_problem, args):
        """Create a Pipeline from a configuration dictionary."""

        params_provider = None
        provider_args = {}
        if "params_provider" in config:
            provider_config = config["params_provider"]
            provider_cls =  PARAMS_PROVIDERS[provider_config["provider_name"]]
            params_provider = provider_cls.from_config(provider_config["provider_init"])
            if any(x in provider_config["provider_name"] for x in ["DepthOneScanTrainer","TransferTrainer","RandomRegularDepthOneFit"]):
                provider_args["cost_op"] = input_problem
            if hasattr(args, f"provider_kwargs"):
                provider_args_str = getattr(args, f"provider_kwargs")
                cmd_provider_kwargs = params_provider.parse_runtime_kwargs(provider_args_str)
                provider_args.update(cmd_provider_kwargs)
        pipeline_components = []
        components_args = defaultdict(dict)
        if "pipeline_components" in config:
            for component_idx, component_config in enumerate(config["pipeline_components"]):
                component_cls = PIPELINE_COMPONENTS[component_config["component_name"]]
                component = component_cls.from_config(component_config["component_init"])
                pipeline_components.append(
                    component
                )
                if hasattr(args, f"component_kwargs{component_idx}"):
                    train_args_str = getattr(args, f"component_kwargs{component_idx}")
                    cmd_train_kwargs = component.parse_runtime_kwargs(train_args_str)
                    components_args[component_idx].update(cmd_train_kwargs)
                components_args[component_idx].update({"cost_op": input_problem})
        return cls(pipeline_components, params_provider), provider_args, components_args

    def execute(self, provider_args: dict, components_args: dict, results_logger: dict):
        """Executes the pipeline"""
        params = self._params_provider.provide_params(**provider_args)
        results_logger["params_provider"] = params
        for component_idx, component in enumerate(self._pipeline_components):
            components_args[component_idx].update(params0=params["optimized_params"])
            params = component.provide_params(**components_args[component_idx])
            results_logger[component_idx] = params
        return params