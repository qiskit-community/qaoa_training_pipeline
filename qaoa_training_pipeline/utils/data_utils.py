#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These functions help to process the data."""

import json
from collections import defaultdict
from typing import Dict

from qiskit_optimization import QuadraticProgram

from qiskit.quantum_info import SparsePauliOp


def samples_to_objective_values(
    samples: Dict[str, float],
    program: QuadraticProgram,
) -> Dict[float, float]:
    """Convert the samples to values of the objective function.

    Args:
        samples: A dict of samples where the key is bitstring and the value is the
            number of times that bitstring has been sampled.
        program: The quadratic program that will evaluate the quality of the samples.

    Returns:
        A dictionary of function values as keys and probabilities of sampling that
        value as dictionary values.
    """
    objective_values = defaultdict(float)
    for bit_str, prob in samples.items():
        candidate_sol = [int(bit) for bit in bit_str[::-1]]
        f_value = program.objective.evaluate(candidate_sol)
        objective_values[f_value] += prob

    return objective_values


def standardize_scipy_result(result, params0, train_duration, sign) -> dict:
    """Standardizes results from SciPy such that it can be serialized."""
    result = dict(result)
    result["optimized_params"] = result.pop("x").tolist()
    result["energy"] = sign * result.pop("fun")
    result["x0"] = params0
    result["train_duration"] = train_duration

    # Serialize the success bool to avoid json crashing
    if "success" in result:
        success = result["success"]
        result["success"] = f"{success}"

    return result


def input_to_operator(input_data: dict, pre_factor: float = 1.0) -> SparsePauliOp:
    """Create a cost operator from a dict.

    This function supports both QUBOs and higher order problems.

    Args:
        input_data: A dictionary (typically loaded from json). The data that specifies
            the cost operator is stored under the `edge list` key. The value is a list
            of dictionaries where each entry corresponds to a term in the cost
            operator.
        pre_factor: A multiplicative pre-factor applied to every term. This defaults
            to 1.0. For example, if the input is a graph then we can create a maximum
            cut cost operator by setting this prefactor to `-0.5`.

    Returns:
        A cost operator in SparsePauliOp form. This is a diagonal Hamiltonian built
        from Pauli Z's.
    """
    pauli_list = []

    # First, find the maximum number of variables.
    n_vars = 0
    for term in input_data["edge list"]:
        n_vars = max([n_vars] + term["nodes"])

    for term in input_data["edge list"]:
        paulis = ["I"] * (n_vars + 1)

        for variable in term["nodes"]:
            paulis[variable] = "Z"

        pauli_list.append(("".join(paulis)[::-1], pre_factor * term.get("weight", 1.0)))

    return SparsePauliOp.from_list(pauli_list)


def load_input(file_name: str) -> dict:
    """Load data in an input file from a json file."""
    with open(file_name, "r") as fin:
        data = json.load(fin)

    return data
