#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pipeline components for QAOA angles training.

This module defines the PipelineComponent abstract class, which extends
ParamsProvider to support active parameter generation through methods found in the literature. 
Pipeline components are the building blocks of QAOA training pipelines, where each PipelineComponent can 
receive QAOA angles and improve them to generate optimized QAOA angles.

"""

from abc import abstractmethod

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.qaoa_training_pipeline.params_provider import ParamsProvider
from qaoa_training_pipeline.training.param_result import ParamResult


class PipelineComponent(ParamsProvider):
    """A pipeline component is a class that is responsible for providing or receiving
    QAOA angles according to a method defined in subclasses inheriting from
    PipelineComponent."""

    @abstractmethod
    def run(
        self,
        *,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit,
        initial_state: QuantumCircuit,
        ansatz_circuit: QuantumCircuit,
        params0: list[float],
    ):
        """Runs the PipelineComponent method to provide angles (which may involve training
        or optimization)"""
        raise NotImplementedError("Sub-classes must implement `run`.")

    def provide_params(self, **kwargs) -> ParamResult:
        """Provides QAOA angles by receiving initial angles and running a method to improve them"""
        return self.run(**kwargs)
