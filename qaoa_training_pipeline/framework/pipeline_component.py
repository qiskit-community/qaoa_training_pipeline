#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pipeline components for QAOA angles training.

This module defines the PipelineComponent abstract class, which extends ProblemParamsProvider to
support QAOA angles optimization. Pipeline components are the building blocks of QAOA 
training pipelines. PipelineComponents can receive QAOA angles and improve them
through various optimization or training methods found in the literature.

Multiple instances of PipelineComponent can be chained together in a Pipeline to create 
multi-stage optimization workflows, where each component refines the parameters produced by
the previous stage.
"""

from abc import abstractmethod

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.framework.problem_params_provider import (
    ProblemParamsProvider,
)
from qaoa_training_pipeline.training.functions import BaseAnglesFunction
from qaoa_training_pipeline.framework.param_result import ParamResult


class PipelineComponent(ProblemParamsProvider):
    """Abstract base class for pipeline components that optimize QAOA angles.

    PipelineComponent extends ProblemParamsProvider to add training and optimization capabilities.
    It receives initial QAOA angles and applies methods found in the literature to improve them.
    Such methods require PipelineComponent to assess the energy of a given objective through an
    energy evaluator, and to either minimize or maximize such objective.

    Attributes:
        _evaluator: Evaluator for assessing angles quality during optimization.
        _qaoa_angles_function: Inherited from ParamsProvider; transforms angles before use.

    Abstract methods that sub-classes implement:
        - provide_params: Defines the optimization method.
        - minimization: Indicates whether this component minimizes or maximizes the objective.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator | None = None,
        *,
        qaoa_angles_function: BaseAnglesFunction | None = None,
    ):
        """Initialize the pipeline component.

        Args:
            evaluator: evaluator for assessing parameter quality during optimization.
            qaoa_angles_function: Optional function to transform QAOA angles. If None,
                uses IdentityFunction (no transformation).
        """

        super().__init__(qaoa_angles_function=qaoa_angles_function)
        self._evaluator = evaluator

    @property
    @abstractmethod
    def minimization(self) -> bool:
        """Indicate whether this component minimizes or maximizes the objective.

        Returns:
            True if the component performs energy minimization (typical for QAOA),
            False if it performs maximization.
        """

    @abstractmethod
    # pylint: disable=too-many-positional-arguments,arguments-differ
    def provide_params(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit,
        initial_state: QuantumCircuit,
        ansatz_circuit: QuantumCircuit,
        params0: list[float],
    ) -> ParamResult:
        """Execute the optimization method to improve QAOA angles.

        This abstract method must be implemented by subclasses to define their specific
        optimization or training strategy. It receives the problem definition and initial
        parameters, then applies an optimization method to improve them.

        Args:
            cost_op: The cost Hamiltonian as a sparse Pauli operator, defining the
            optimization problem.
            mixer: Quantum circuit representing the mixer Hamiltonian (typically X mixer).
            initial_state: Quantum circuit preparing the initial state (typically |+âŸ©^n).
            ansatz_circuit: Parameterized quantum circuit representing the QAOA ansatz.
            params0: Initial QAOA angles to be optimized. They are expected to be in the format
                [Î²1, ..., Î²p, Î³1, ..., Î³p].

        Returns:
            ParamResult object containing the optimized angles and associated metadata
            (e.g., optimization history, final energy,...).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Sub-classes must implement `provide_params`.")

    @property
    def evaluator(self) -> BaseEvaluator:
        """Get the evaluator used for the QAOA angles energy evaluation.

        Returns:
            The BaseEvaluator instance used to evaluate parameter quality during
            optimization.

        Raises:
            ValueError: If the evaluator was not defined during initialization and
            is accessed before being set.
        """
        if self._evaluator is None:
            raise ValueError("The evaluator must be defined before accessing it")
        return self._evaluator

    def to_config(self) -> dict:
        """Serialize the PipelineComponent to a configuration dictionary.

        Creates a serializable dictionary representation of the component's configuration,
        including the component name, evaluator settings, and angle transformation function.
        This is used for tracking how results are generated and for logging purposes.

        Returns:
            Dictionary containing the PipelineComponent's configuration, i.e., its name,
            evaluator, and angle transformation function.
        """
        return {
            "pipeline_component_name": self.__class__.__name__,
            "evaluator": self._evaluator.to_config() if self._evaluator else None,
            "qaoa_angles_function": self._qaoa_angles_function.to_config(),
        }

