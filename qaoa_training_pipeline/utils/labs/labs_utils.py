# Using many functions from https://github.com/jpmorganchase/QOKit
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

from __future__ import annotations
import sys
from collections.abc import Sequence, Iterable
import numpy as np
from itertools import combinations
from operator import mul
from functools import reduce
from numba import njit
from pathlib import Path
from time import time

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import qaoa_ansatz


# approximate optimal merit factor and energy for small Ns
# from Table 1 of https://arxiv.org/abs/1512.02475
true_optimal_mf = {
    3: 4.500,
    4: 4.000,
    5: 6.250,
    6: 2.571,
    7: 8.167,
    8: 4.000,
    9: 3.375,
    10: 3.846,
    11: 12.100,
    12: 7.200,
    13: 14.083,
    14: 5.158,
    15: 7.500,
    16: 5.333,
    17: 4.516,
    18: 6.480,
    19: 6.224,
    20: 7.692,
    21: 8.481,
    22: 6.205,
    23: 5.628,
    24: 8.000,
    25: 8.681,
    26: 7.511,
    27: 9.851,
    28: 7.840,
    29: 6.782,
    30: 7.627,
    31: 7.172,
    32: 8.000,
    33: 8.508,
    34: 8.892,
    35: 8.390,
}

true_optimal_energy = {
    3: 1,
    4: 2,
    5: 2,
    6: 7,
    7: 3,
    8: 8,
    9: 12,
    10: 13,
    11: 5,
    12: 10,
    13: 6,
    14: 19,
    15: 15,
    16: 24,
    17: 32,
    18: 25,
    19: 29,
    20: 26,
    21: 26,
    22: 39,
    23: 47,
    24: 36,
    25: 36,
    26: 45,
    27: 37,
    28: 50,
    29: 62,
    30: 59,
    31: 67,
    32: 64,
    33: 64,
    34: 65,
    35: 73,
}

# -----------------------------------------------------------------------------
# LABS training pipeline helpers (CLI / config overrides)
# -----------------------------------------------------------------------------


def _replace_value_recursive(conf, old_value: str, new_value: str):
    """Recursively replace exact string values in nested dict/list configs."""
    if isinstance(conf, dict):
        for k, v in conf.items():
            if isinstance(v, str) and v == old_value:
                conf[k] = new_value
            else:
                _replace_value_recursive(v, old_value, new_value)
    elif isinstance(conf, list):
        for item in conf:
            _replace_value_recursive(item, old_value, new_value)


def _set_energy_minimization_default_recursive(trainer_conf, energy_minimization: bool):
    """Recursively set default energy_minimization for ScipyTrainer if not specified."""
    if not isinstance(trainer_conf, dict):
        return

    if trainer_conf.get("trainer") == "ScipyTrainer":
        trainer_init = trainer_conf.get("trainer_init")
        if isinstance(trainer_init, dict) and "energy_minimization" not in trainer_init:
            trainer_init["energy_minimization"] = energy_minimization

    for val in trainer_conf.values():
        if isinstance(val, dict):
            _set_energy_minimization_default_recursive(val, energy_minimization)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    _set_energy_minimization_default_recursive(item, energy_minimization)


def apply_labs_training_config_overrides(trainer_chain_config):
    """Apply LABS-specific in-memory overrides to a trainer_chain config.

    This lets wrappers (like scripts/run_methods.sh) avoid mutating method JSON files.
    """
    # LABS has higher-than-quadratic terms; EfficientDepthOneEvaluator is invalid.
    _replace_value_recursive(
        trainer_chain_config, "EfficientDepthOneEvaluator", "StatevectorEvaluator"
    )
    # LABS convention in this repo: minimize energy by default in SciPy trainers,
    # but only if the method JSON didn't explicitly specify it.
    _set_energy_minimization_default_recursive(trainer_chain_config, True)


def is_labs_problem(cost_op, tolerance: float = 1e-10) -> bool:
    """Check if a SparsePauliOp represents a LABS problem.

    This function verifies if the cost operator matches the structure of a LABS problem
    by checking:
    1. All terms are either quadratic (2 Z operators) or quartic (4 Z operators)
    2. Quartic terms have weight 4.0 (LABS-specific)
    3. The problem has at least some quartic terms (distinguishes from pure quadratic problems)

    Note: This is a heuristic check. Other non-quadratic problems might also pass this test
    if they have the same structure. For a definitive check, you would need to verify
    the exact term structure matches what get_terms_offset() would produce.

    Parameters
    ----------
    cost_op : SparsePauliOp
        The cost operator to check
    tolerance : float
        Numerical tolerance for weight comparison

    Returns
    -------
    bool
        True if the cost operator appears to be a LABS problem, False otherwise
    """

    has_quartic_terms = False
    has_invalid_terms = False

    for pauli, coeff in cost_op.to_list():
        # Count Z operators in the Pauli string
        z_count = pauli.count("Z")

        # LABS problems only have quadratic (2 Z) or quartic (4 Z) terms
        if z_count != 2 and z_count != 4:
            # If there are terms with odd number of Z operators or > 4, it's not LABS
            if z_count > 0:  # Ignore identity terms
                has_invalid_terms = True
                break

        # Check quartic terms specifically - they must have weight 4 in LABS
        if z_count == 4:
            has_quartic_terms = True
            abs_coeff = abs(coeff)
            # Quartic terms in LABS have weight 4
            if abs(abs_coeff - 4.0) > tolerance:
                has_invalid_terms = True
                break

    # LABS problems must:
    # 1. Have quartic terms (distinguishes from pure quadratic problems like MaxCut)
    # 2. Not have invalid terms (odd order or > 4 Z operators)
    return has_quartic_terms and not has_invalid_terms


def get_terms_offset(N: int):
    """Return terms with indices and offset of Pauli Zs in the LABS problem definition

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : list of tuples
        List of tuples, where each tuple defines a summand
        and contains weight and indices of the Pauli Zs in the product
        e.g. if terms = [(2, (0,1)), (4, (0,1,2,3)), (2, (1,2))]
        the Hamiltonian is Z0Z1 + Z0Z1Z2Z3 + Z1Z2

    offset : int
        energy offset required due to constant factors (identity terms)
        not included in the Hamiltonian
    """
    # return the indices of Pauli Zs in products
    all_terms = []
    offset = 0
    # sum_{k=1}^{N-1} (\sum_{i=1}^{N-k} s_i s_{i+k})^2
    for k in range(1, N):
        offset += N - k  # quadratic terms go to one
        for i, j in combinations(range(1, N - k + 1), 2):
            # -1 because we index from 1
            # Drop duplicate terms, e.g. Z1Z2Z2Z3 should be just Z1Z3
            if i + k == j:
                all_terms.append((2, tuple(sorted((i - 1, j + k - 1)))))
            else:
                all_terms.append((4, tuple(sorted((i - 1, i + k - 1, j - 1, j + k - 1)))))
    return list(set(all_terms)), offset


@njit()
def energy_vals(s: Sequence, N: int | None = None) -> float:
    """Compute LABS energy values from a string of spins

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {+1, -1}
    N : int
        Number of spins

    Returns
    -------
    energy_vals : float
        energy_vals of s
    """
    if N is None:
        N = len(s)
    E_s = 0
    for k in range(1, N):
        C_k = 0
        for i in range(1, N - k + 1):
            C_k += s[i - 1] * s[i + k - 1]
        E_s += C_k**2
    return E_s


def energy_vals_from_bitstring(x, N: int | None = None) -> float:
    """Convenience function
    Useful to get the LABS energy values for bitstrings which are {0,1}^N

    Parameters
    ----------
    s : list-like
        sequence of bits for which to compute the merit factor
        set(s) \in {0, 1}
    N : int
        Number of spins


    Returns
    -------
    energy_Vals : float
        energy_vals of s
    """
    return energy_vals(1 - 2 * x, N=N)


def energy_vals_general(
    s: Sequence,
    terms: Iterable | None = None,
    offset: float | None = None,
    check_parameters: bool = True,
) -> float:
    """Compute energy values from a string of spins
    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {+1, -1}
    terms : list-like, default None
    offset : float, default None
        precomputed output of get_terms_offset
        terms, offset = get_terms_offset(N)
    check_parameters : bool, default True
        if set to False, no input validation is performed
    Returns
    -------
    energy_vals : float
        energy_vals of s
    """

    if check_parameters:
        assert isinstance(s, Iterable)
        assert set(s).issubset(set([-1, 1]))
    N = len(s)
    if terms is None or offset is None:
        terms, offset = get_terms_offset(N)
    E_s = offset
    for term in terms:
        len_term, val_term = term
        E_s += len_term * reduce(mul, [s[idx] for idx in val_term])
    return E_s


def energy_vals_from_bitstring_general(
    x, terms: Sequence | None = None, offset: float | None = None, check_parameters: bool = False
) -> float:
    """Convenience func
    Useful to get the energy values for bitstrings which are {0,1}^N
    Parameters
    ----------
    s : list-like
        sequence of bits for which to compute the merit factor
        set(s) \in {0, 1}
    terms : list-like, default None
    offset : float, default None
        precomputed output of get_terms_offset
        terms, offset = get_terms_offset(N)
    check_parameters : bool, default True
        if set to False, no input validation is performed
    Returns
    -------
    energy_Vals : float
        energy_vals of s
    """
    return energy_vals_general(1 - 2 * x, terms=terms, offset=offset, check_parameters=check_parameters)  # type: ignore


def slow_merit_factor(
    s: Sequence,
    terms: Iterable | None = None,
    offset: float | None = None,
    check_parameters: bool = True,
) -> float:
    """Compute merit factor from a string of spins

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {-1, +1}
    terms : list-like, default None
    offset : float, default None
        precomputed output of get_terms_offset
        terms, offset = get_terms_offset(N)
    check_parameters : bool, default True
        if set to False, no input validation is performed


    Returns
    -------
    merit_factor : float
        merit factor of s
    """
    if check_parameters:
        assert isinstance(s, Iterable)
        assert set(s).issubset(set([-1, 1]))
    N = len(s)
    if terms is None or offset is None:
        terms, offset = get_terms_offset(N)
    E_s = offset
    for term in terms:
        len_term, val_term = term
        E_s += len_term * reduce(mul, [s[idx] for idx in val_term])
    return N**2 / (2 * E_s)


def merit_factor(s: Sequence, N: int | None = None) -> float:
    """Compute merit factor from a string of spins
    Faster implementation that does not rely on terms computation

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {-1, +1}
    N : int
        Number of spins

    Returns
    -------
    merit_factor : float
        merit factor of s
    """
    if N is None:
        N = len(s)
    E_s = energy_vals(s, N=N)
    return N**2 / (2 * E_s)


def negative_merit_factor_from_bitstring(x, N: int | None = None) -> float:
    """Convenience function
    Useful for e.g. QAOA since convention is to minimize, and we want a maximum merit factor

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {-1, +1}
    N : int
        Number of spins

    Returns
    -------
    merit_factor : float
        negative merit factor of s
    """
    return -merit_factor(1 - 2 * x, N=N)


def get_depth_optimized_terms(N: int) -> list:
    """Return indices of Pauli Zs in the LABS problem definition. The terms in the returned list are ordered to attempt to compress
    the circuit depth and increase parallelism.

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : list of tuples
        List of ordered tuples, where each tuple defines a summand
        and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0,1), (0,1,2,3), (1,2)]
        the Hamiltonian is Z0Z1 + Z0Z1Z2Z3 + Z1Z2
    """

    done = []  # contains gates that have already been applied
    layers = []

    # prioritize ZZZZ terms
    for pivot in range(N - 3, 0, -1):
        for t in range(1, int((N - pivot - 1) / 2) + 1):
            for k in range(t + 1, N - pivot - t + 1):
                interactions: list[tuple[int, int, int, int] | tuple[int, int]] = [
                    (pivot, pivot + t, pivot + k, pivot + t + k)
                ]
                if set(interactions[0]) in done:
                    continue
                idx_used = list(interactions[0])

                done.append(set(interactions[0]))
                # greedily apply ZZZZ terms to free qubits
                stack = sorted(list(set(range(1, N + 1)) - set(idx_used)))
                try:
                    while stack:
                        f = stack.pop()
                        for s in stack:
                            if f in idx_used:
                                break
                            for th in filter(lambda x: x < s, stack):
                                if s in idx_used:
                                    break
                                a = th - (f - s)
                                if a > 0 and not (a in idx_used) and set([a, th, s, f]) not in done:
                                    stack.remove(a)
                                    stack.remove(s)
                                    stack.remove(th)
                                    idx_used += [a, th, s, f]
                                    done.append(set([a, th, s, f]))
                                    interactions.append((a, th, s, f))
                except IndexError:
                    pass
                # greedily apply ZZ terms to free qubits
                try:
                    stack = sorted(list(set(range(1, N + 1)) - set(idx_used)))

                    while stack:
                        f = stack.pop()
                        for k in range(1, int((f - 1) / 2) + 1):
                            a = f - 2 * k
                            if a > 0 and not (a in idx_used) and set([a, f]) not in done:
                                stack.remove(a)
                                idx_used += [a, f]
                                done.append(set([a, f]))
                                interactions.append((a, f))
                except IndexError:
                    pass
                layers.append(set(j - 1 for j in i) for i in interactions)

    # add any missing ZZ terms not covered by the above. Typically isn't used at high N.
    for pivot in range(N - 2, 0, -1):
        for t in range(1, int((N - pivot) / 2) + 1):
            interactions = [(pivot, pivot + 2 * t)]
            if set(interactions[0]) in done:
                continue
            idx_used = list(interactions[0])
            done.append(set(interactions[0]))
            # greedily apply ZZ terms to free qubits
            try:
                stack = sorted(list(set(range(1, N + 1)) - set(idx_used)))

                while stack:
                    f = stack.pop()
                    for k in range(1, int((f - 1) / 2) + 1):
                        a = f - 2 * k
                        if a > 0 and not (a in idx_used) and set([a, f]) not in done:
                            stack.remove(a)
                            idx_used += [a, f]
                            done.append(set([a, f]))
                            interactions.append((a, f))
            except IndexError:
                pass
            layers.append(set(j - 1 for j in i) for i in interactions)

    # linearize list of cost operator layers
    terms = []
    for layer in layers:
        terms += [tuple(sorted(l)) for l in layer]
    return terms


def get_gate_optimized_terms_naive(N: int, number_of_gate_zones: int = 4):
    """
    Try to naively line up terms to encourage many CNOT cancellations
    """
    terms = []
    for i in range(1, N - 3 + 1):
        for t in range(1, int((N - i - 1) / 2) + 1):
            for k in range(t + 1, N - i - t + 1):
                terms.append((i - 1, i + t - 1, i + k - 1, i + t + k - 1))
    for i in range(1, N - 2 + 1):
        for t in range(1, int((N - i) / 2) + 1):
            terms.append((i - 1, i + 2 * t - 1))

    if number_of_gate_zones:
        k = 0
        while k < len(terms):
            num_zones_left = number_of_gate_zones - len(terms[k]) // 2
            j = k + 1
            swapped = 1
            while j < len(terms) and num_zones_left != 0:
                new_term = terms[j]
                if not set(terms[k]).intersection(set(new_term)):
                    terms.remove(new_term)
                    terms.insert(k + 1, new_term)
                    num_zones_left -= len(new_term) // 2
                    swapped += 1
                j += 1
            k += swapped

    return terms


def get_gate_optimized_terms_greedy(N: int, number_of_gate_zones: int = 4, seed: int | None = None):
    """
    Try to greedly cancel CNOTs for RZZZZ terms
    """

    four_body_by_k = []
    for k in range(1, N - 1):
        terms = []
        for i in range(0, N - k):
            for j in range(i + 1, N - k):
                if i + k < j:
                    terms.append((i, i + k, j, j + k))
        if terms:
            four_body_by_k.append(terms)

    two_body = []
    for i in range(0, N - 2):
        for k in range(1, int((N - i + 1) // 2)):
            two_body.append((i, i + 2 * k))

    circuit = []

    seed = seed if seed else np.random.randint(np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # greedly align four body terms to cancel CNOTs
    for terms in four_body_by_k:
        i = rng.choice(len(terms))
        first_term = terms.pop(i)
        circuit.append(first_term)
        open_tops = [first_term[:2]]
        open_bottoms = [first_term[2:]]

        while terms:
            terms_with_scores = []
            for t in terms:
                sc = [0, 0, t]

                cancels_top = False
                cancels_bottom = False
                if t[:2] in open_tops:
                    sc[0] += 1
                    sc[1] += 1
                    cancels_top = True
                if t[2:] in open_bottoms:
                    sc[0] += 1
                    sc[1] += 1
                    cancels_bottom = True

                qubits_top = set(sum(map(lambda x: list(x), open_tops), []))
                if set(qubits_top).intersection(set(t[2:])):
                    sc[0] -= 1

                qubits_bottom = set(sum(map(lambda x: list(x), open_bottoms), []))
                if set(qubits_bottom).intersection(set(t[:2])):
                    sc[0] -= 1

                if not cancels_top:
                    if set(qubits_top).intersection(set(t[:2])):
                        sc[0] -= 1
                if not cancels_bottom:
                    if set(qubits_bottom).intersection(set(t[2:])):
                        sc[0] -= 1

                terms_with_scores.append(sc)

            score, num_cancels, new_term = tuple(sorted(terms_with_scores, key=lambda x: -x[0])[0])

            terms.remove(new_term)
            circuit.append(new_term)
            open_tops.append(new_term[:2])
            open_bottoms.append(new_term[2:])

    # squeeze two body inbetween aligned four-body terms
    for b in two_body:
        found = False
        for i, term in enumerate(circuit):
            if b == term[:2] or b == term[2:]:
                circuit.insert(i + 1, b)
                found = True
                break
        if not found:
            circuit.append(b)

    if number_of_gate_zones:
        # move terms back to accomodate gate zone parallization
        k = 0
        while k < len(circuit):
            num_zones_left = number_of_gate_zones - len(circuit[k]) // 2
            j = k + 1
            swapped = 1
            while j < len(circuit) and num_zones_left != 0:
                new_term = circuit[j]
                if not set(circuit[k]).intersection(set(new_term)):
                    circuit.remove(new_term)
                    circuit.insert(k + 1, new_term)
                    num_zones_left -= len(new_term) // 2
                    swapped += 1
                j += 1
            k += swapped

    return circuit


def load_ground_state_bitstrings(num_qubits: int) -> set:
    """Load precomputed ground state bitstrings for a given number of qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the problem

    Returns
    -------
    set
        Set of ground state bitstrings as strings (e.g., {"0101", "1010"})
    """

    precomputed_path = Path(__file__).parent / "precomputed_bitstrings"
    gs_file = precomputed_path / f"precomputed_bitstrings_{num_qubits}.npy"

    if not gs_file.exists():
        raise FileNotFoundError(f"Ground state bitstrings file not found: {gs_file}")

    gs_bitstrings_array = np.load(gs_file)
    # Convert numpy array bitstrings to string format
    gs_bitstrings = {"".join(map(str, bs)) for bs in gs_bitstrings_array}
    return gs_bitstrings


def compute_overlap_with_gs_subspace(probabilities: dict, num_qubits: int) -> float:
    """Compute the overlap of a wavefunction with the ground state subspace.

    This computes the sum of probabilities over all ground state bitstrings.

    Parameters
    ----------
    probabilities : dict
        Dictionary mapping bitstrings (as strings) to their probabilities
    num_qubits : int
        Number of qubits in the problem

    Returns
    -------
    float
        Overlap with ground state subspace (sum of probabilities over GS bitstrings)
    """
    gs_bitstrings = load_ground_state_bitstrings(num_qubits)
    overlap = sum(probabilities.get(bs, 0.0) for bs in gs_bitstrings)
    return overlap


def compute_labs_objective(
    cost_op,
    qaoa_angles: list,
    evaluator,
    sign: int,
    mixer=None,
    initial_state=None,
    ansatz_circuit=None,
    objective: str = "energy",
) -> tuple:
    """Compute the objective for LABS QAOA optimization.

    Parameters
    ----------
    cost_op : SparsePauliOp
        The LABS cost operator
    qaoa_angles : list
        The QAOA angles (beta/gamma values)
    evaluator : BaseEvaluator
        The evaluator to use for energy computation
    sign : int
        Sign to control minimization/maximization (1 for minimize, -1 for maximize)
    mixer : QuantumCircuit, optional
        The mixer circuit (default: standard QAOA mixer)
    initial_state : QuantumCircuit, optional
        The initial state circuit (default: equal superposition)
    ansatz_circuit : QuantumCircuit, optional
        Custom ansatz circuit
    objective : str
        The objective type: "energy" or "overlap" (default: "energy")

    Returns
    -------
    tuple
        A tuple (objective_value, tracked_value) where:
        - objective_value: The value to minimize
        - tracked_value: The value to track in history (energy or overlap)
    """
    if objective == "overlap":
        # Optimize overlap with ground state subspace (LABS only)
        # Construct QAOA circuit
        reps = len(qaoa_angles) // 2
        circ = qaoa_ansatz(
            cost_operator=cost_op, reps=reps, mixer_operator=mixer, initial_state=initial_state
        )
        circ_with_params = circ.assign_parameters(qaoa_angles)

        # Check if evaluator supports GPU and use it if available
        use_gpu = False
        device = None

        # Check if evaluator is StatevectorEvaluator and has GPU configured
        if hasattr(evaluator, "_init_args"):
            device = evaluator._init_args.get("device")
            if device == "GPU":
                use_gpu = True

        if use_gpu:
            # Create GPU-enabled simulator
            backend = AerSimulator(device="GPU", method="statevector")

            # Add save_statevector instruction to circuit
            circ_with_save = circ_with_params.copy()
            circ_with_save.save_statevector()

            # Run circuit on GPU and get statevector
            job = backend.run(circ_with_save)
            result = job.result()
            sv = result.get_statevector()

            # Convert to probabilities dict
            probs = sv.probabilities_dict()

        if not use_gpu:
            # CPU-based statevector computation (fallback)
            sv = Statevector(circ_with_params)
            probs = sv.probabilities_dict()

        # Compute overlap with ground state subspace
        overlap = compute_overlap_with_gs_subspace(probs, cost_op.num_qubits)

        # Return 1 - overlap (to minimize, since we want to maximize overlap)
        return 1.0 - overlap, overlap
    else:
        # Standard energy optimization for LABS
        # This path uses the evaluator (which can use GPU)
        energy = sign * evaluator.evaluate(
            cost_op=cost_op,
            params=qaoa_angles,
            mixer=mixer,
            initial_state=initial_state,
            ansatz_circuit=ansatz_circuit,
        )
        objective_value = float(energy)
        return objective_value, sign * objective_value


def create_labs_energy_function(
    cost_op,
    qaoa_angles_function,
    evaluator,
    sign: int,
    energy_evaluation_time,
    energy_history,
    parameter_history,
    mixer=None,
    initial_state=None,
    ansatz_circuit=None,
    objective: str = "energy",
):
    """Create a LABS-specific energy function for optimization.

    This function returns a callable that can be used as the objective function
    for SciPy's minimize, with LABS-specific handling built in.

    Parameters
    ----------
    cost_op : SparsePauliOp
        The LABS cost operator
    qaoa_angles_function : callable
        Function to convert optimization parameters to QAOA angles
    evaluator : BaseEvaluator
        The evaluator to use for energy computation
    sign : int
        Sign to control minimization/maximization (1 for minimize, -1 for maximize)
    mixer : QuantumCircuit, optional
        The mixer circuit (default: standard QAOA mixer)
    initial_state : QuantumCircuit, optional
        The initial state circuit (default: equal superposition)
    ansatz_circuit : QuantumCircuit, optional
        Custom ansatz circuit
    energy_evaluation_time : list
        List to append evaluation times to
    energy_history : list
        List to append energy/overlap values to
    parameter_history : list
        List to append parameter values to
    objective : str
        The objective type: "energy" or "overlap" (default: "energy")

    Returns
    -------
    callable
        The energy function to use with SciPy's minimize
    """

    def _energy(x):
        """Maximize the energy by minimizing the negative energy."""
        estart = time()

        qaoa_angles = qaoa_angles_function(x)

        objective_value, tracked_value = compute_labs_objective(
            cost_op=cost_op,
            qaoa_angles=qaoa_angles,
            evaluator=evaluator,
            sign=sign,
            mixer=mixer,
            initial_state=initial_state,
            ansatz_circuit=ansatz_circuit,
            objective=objective,
        )

        energy_evaluation_time.append(time() - estart)
        energy_history.append(tracked_value)
        parameter_history.append(list(float(val) for val in x))

        return objective_value

    return _energy


def process_labs_post_optimization(
    cost_op,
    param_result: dict,
    mixer=None,
    initial_state=None,
) -> dict:
    """Process all LABS-specific post-optimization tasks.

    This function handles:
    1. Energy reporting with offset calculation
    2. Wavefunction analysis and TTS calculation

    Parameters
    ----------
    cost_op : SparsePauliOp
        The cost operator
    param_result : dict
        The parameter result dictionary to update
    mixer : QuantumCircuit, optional
        The mixer circuit
    initial_state : QuantumCircuit, optional
        The initial state circuit

    Returns
    -------
    dict
        Updated param_result with all LABS-specific fields
    """
    # Process LABS-specific results (energy reporting, offset calculation)
    param_result, _ = process_labs_results(cost_op, param_result)

    # Analyze LABS wavefunction and calculate TTS
    labs_analysis = analyze_labs_wavefunction(
        cost_op=cost_op,
        optimized_qaoa_angles=param_result["optimized_qaoa_angles"],
        mixer=mixer,
        initial_state=initial_state,
    )
    # Update param_result with TTS analysis results
    param_result.update(labs_analysis)

    return param_result


def process_labs_results(cost_op, param_result: dict) -> tuple:
    """Process and report LABS-specific results.

    This function checks if the cost operator is a LABS problem, calculates
    the LABS energy (with offset), and updates the param_result dictionary.

    Parameters
    ----------
    cost_op : SparsePauliOp
        The cost operator
    param_result : dict
        The parameter result dictionary to update

    Returns
    -------
    tuple
        A tuple (updated_param_result, is_labs) where:
        - updated_param_result: Updated param_result with LABS-specific fields if applicable
        - is_labs: Boolean indicating if this is a LABS problem
    """
    is_labs = is_labs_problem(cost_op)
    labs_offset = get_terms_offset(cost_op.num_qubits)[1] if is_labs else None

    # Report energy
    hamiltonian_energy = param_result["energy"]
    if labs_offset is not None:
        labs_energy = hamiltonian_energy + labs_offset
        optimal_str = (
            f", optimal: {true_optimal_energy[cost_op.num_qubits]}"
            if cost_op.num_qubits in true_optimal_energy
            else ""
        )
        print(f"LABS Energy: {labs_energy}{optimal_str}")
        param_result["labs_energy"] = labs_energy
    else:
        print(f"Energy: {hamiltonian_energy}")

    return param_result, is_labs


def analyze_labs_wavefunction(
    cost_op,
    optimized_qaoa_angles: list,
    mixer=None,
    initial_state=None,
    top_n: int = 0,
    max_qubits: int = 17,
) -> dict:
    """Analyze a LABS QAOA wavefunction and calculate TTS.

    This function:
    1. Constructs the QAOA circuit and computes the statevector
    2. Displays top bitstrings with LABS energy verification
    3. Calculates TTS (Time To Solution) using precomputed ground state bitstrings

    Parameters
    ----------
    cost_op : SparsePauliOp
        The LABS cost operator
    optimized_qaoa_angles : list
        The optimized QAOA angles (beta/gamma values)
    mixer : QuantumCircuit, optional
        The mixer circuit (default: standard QAOA mixer)
    initial_state : QuantumCircuit, optional
        The initial state circuit (default: equal superposition)
    top_n : int
        Number of top bitstrings to display (default: 10)
    max_qubits : int
        Maximum number of qubits for which to perform full analysis (default: 11)

    Returns
    -------
    dict
        Dictionary containing:
        - 'p_opt': Probability in ground state
        - 'tts': Time To Solution (1/p_opt)
        - 'num_ground_states': Number of ground states
        - 'analysis_time': Time taken for analysis
    """

    if cost_op.num_qubits > max_qubits:
        return {}

    start_time = time()

    # Construct QAOA circuit and compute statevector
    reps = len(optimized_qaoa_angles) // 2
    circ = qaoa_ansatz(
        cost_operator=cost_op, reps=reps, mixer_operator=mixer, initial_state=initial_state
    )
    sv = Statevector(circ.assign_parameters(optimized_qaoa_angles))
    probs = sv.probabilities_dict()

    # Display top bitstrings and their corresponding LABS energy
    optimal_energy = true_optimal_energy.get(cost_op.num_qubits, None)
    for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        h_energy = sv.from_label(bitstring).expectation_value(cost_op).real
        labs_energy = energy_vals_from_bitstring(
            np.array([int(b) for b in bitstring]), N=cost_op.num_qubits
        )
        is_optimal = "✓" if optimal_energy and labs_energy == optimal_energy else " "
        print(
            f"  {is_optimal} {bitstring}: prob={prob:.6f}, LABS={labs_energy:+.1f}, H={h_energy:+.1f}"
        )

    # Calculate TTS using precomputed ground state bitstrings
    results = {}
    # Load precomputed ground state bitstrings
    precomputed_path = Path(__file__).parent / "precomputed_bitstrings"
    gs_file = precomputed_path / f"precomputed_bitstrings_{cost_op.num_qubits}.npy"

    if gs_file.exists():
        gs_bitstrings_array = np.load(gs_file)
        # Convert numpy array bitstrings to string format
        gs_bitstrings = {"".join(map(str, bs)) for bs in gs_bitstrings_array}

        # Calculate p_opt: probability of finding the wavefunction in any ground state
        p_opt = sum(probs.get(bs, 0.0) for bs in gs_bitstrings)

        # Calculate TTS = 1/p_opt
        tts = 1.0 / p_opt if p_opt > 0 else float("inf")

        print(f"Ground State Analysis:")
        print(f"  Number of ground states: {len(gs_bitstrings)}")
        print(f"  Probability in ground state (p_opt): {p_opt:.6e}")
        print(f"  Time To Solution (TTS = 1/p_opt): {tts:.2e}")
        print(f"=" * 75)

        results["p_opt"] = p_opt
        results["tts"] = tts
        results["num_ground_states"] = len(gs_bitstrings)
    else:
        print(f"\nWarning: Ground state bitstrings file not found: {gs_file}")

    results["analysis_time"] = time() - start_time

    return results
