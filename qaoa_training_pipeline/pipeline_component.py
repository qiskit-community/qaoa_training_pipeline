#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from abc import abstractmethod

from qaoa_training_pipeline.qaoa_training_pipeline.params_provider import ParamsProvider

class PipelineComponent(ParamsProvider):
    """A pipeline component is a class that is responsible for providing or receiving 
    QAOA angles according to a method defined in subclasses inheriting from 
    PipelineComponent."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def run(
        self, 
        mixer, 
        initial_state, 
        ansatz_circuit, 
        params0
    ):
        """Runs the PipelineComponent method to provide angles"""
        raise NotImplementedError("Sub-classes must implement `run`.")

    
