# Using many functions from https://github.com/jpmorganchase/QOKit
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from itertools import combinations
from numba import njit
from pathlib import Path
from time import time

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import qaoa_ansatz


# approximate optimal energy for small Ns
# from Table 1 of https://arxiv.org/abs/1512.02475
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

        # Check quartic terms specifically - they must have weight ±4 in LABS
        # (negative if Hamiltonian was negated for maximization convention)
        if z_count == 4:
            has_quartic_terms = True
            abs_coeff = abs(coeff)
            # Quartic terms in LABS have weight ±4
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
    r"""Compute LABS energy values from a string of spins

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
    r"""Convenience function
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


# -----------------------------------------------------------------------------
# LABS post-processing functions
# -----------------------------------------------------------------------------


def process_labs_post_optimization(
    cost_op,
    param_result: dict,
    mixer=None,
    initial_state=None,
    hamiltonian_negated: bool = False,
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
    hamiltonian_negated : bool
        If True, the Hamiltonian was negated (for maximization convention),
        so energy values need to be negated back to get true LABS energy.

    Returns
    -------
    dict
        Updated param_result with all LABS-specific fields
    """
    # Process LABS-specific results (energy reporting, offset calculation)
    param_result, _ = process_labs_results(
        cost_op, param_result, hamiltonian_negated=hamiltonian_negated
    )

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


def process_labs_results(cost_op, param_result: dict, hamiltonian_negated: bool = False) -> tuple:
    """Process and report LABS-specific results.

    This function checks if the cost operator is a LABS problem, calculates
    the LABS energy (with offset), and updates the param_result dictionary.

    Parameters
    ----------
    cost_op : SparsePauliOp
        The cost operator
    param_result : dict
        The parameter result dictionary to update
    hamiltonian_negated : bool
        If True, the Hamiltonian was negated (for maximization convention),
        so energy values need to be negated back to get true LABS energy.

    Returns
    -------
    tuple
        A tuple (updated_param_result, is_labs) where:
        - updated_param_result: Updated param_result with LABS-specific fields if applicable
        - is_labs: Boolean indicating if this is a LABS problem
    """
    is_labs = is_labs_problem(cost_op)
    labs_offset = get_terms_offset(cost_op.num_qubits)[1] if is_labs else None

    # Calculate LABS energy (with offset)
    # If Hamiltonian was negated, negate back to get true LABS energy
    hamiltonian_energy = param_result["energy"]
    if hamiltonian_negated:
        hamiltonian_energy = -hamiltonian_energy

    if labs_offset is not None:
        labs_energy = hamiltonian_energy + labs_offset
        param_result["labs_energy"] = labs_energy

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
        Number of top bitstrings to display (default: 0)
    max_qubits : int
        Maximum number of qubits for which to perform full analysis (default: 17)

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

        results["p_opt"] = p_opt
        results["tts"] = tts
        results["num_ground_states"] = len(gs_bitstrings)

    results["analysis_time"] = time() - start_time

    return results
