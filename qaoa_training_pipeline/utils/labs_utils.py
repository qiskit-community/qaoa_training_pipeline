"""LABS (Low Autocorrelation Binary Sequences) utility functions.

This module provides functions for working with the LABS problem in QAOA,
including Hamiltonian construction and energy calculations.
"""

from __future__ import annotations
import numpy as np


def is_labs_problem(cost_op, tolerance: float = 1e-10) -> bool:
    """Check if a SparsePauliOp represents a LABS problem.

    This function verifies if the cost operator matches the structure of a LABS problem
    by checking:
    1. All terms are either quadratic (2 Z operators) or quartic (4 Z operators)
    2. Quartic terms have weight 4.0 (LABS-specific)
    3. The problem has at least some quartic terms (distinguishes from pure quadratic problems)

    Note: This is a heuristic check. Other non-quadratic problems might also pass this test
    if they have the same structure. For a definitive check, you would need to verify
    the exact term structure matches what build_labs_hamiltonian_terms() would produce.

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
        z_count = pauli.count("Z")

        if z_count != 2 and z_count != 4:
            if z_count > 0:
                has_invalid_terms = True
                break

        if z_count == 4:
            has_quartic_terms = True
            abs_coeff = abs(coeff)
            if abs(abs_coeff - 4.0) > tolerance:
                has_invalid_terms = True
                break

    # LABS problems must:
    # 1. Have quartic terms (distinguishes from pure quadratic problems like MaxCut)
    # 2. Not have invalid terms (odd order or > 4 Z operators)
    return has_quartic_terms and not has_invalid_terms


def build_labs_hamiltonian_terms(num_spins: int) -> tuple[list[tuple], int]:
    """Build the LABS Hamiltonian terms and compute the constant offset.

    The LABS energy is defined as E = Σₖ Cₖ² where Cₖ = Σᵢ sᵢ·sᵢ₊ₖ (autocorrelation).
    Expanding the square gives ZZ (quadratic) and ZZZZ (quartic) Pauli terms.

    Parameters
    ----------
    num_spins : int
        Number of spins (problem size N)

    Returns
    -------
    terms : list of tuples
        Each tuple is (weight, indices) where indices are the qubit positions
        for Z operators. E.g., (4, (0, 1, 2, 3)) means 4·Z₀Z₁Z₂Z₃
    offset : int
        Constant energy offset from identity terms (excluded from Hamiltonian)
    """
    # Collect all terms as (weight, indices) tuples, use set for deduplication
    all_terms = set()
    offset = 0

    # E = Σₖ (Σᵢ sᵢ·sᵢ₊ₖ)² expands to cross-terms between pairs (i, i+k) and (j, j+k)
    for lag in range(1, num_spins):
        # Identity contribution: each (sᵢ·sᵢ₊ₖ)² = 1, giving (num_spins - lag) terms
        offset += num_spins - lag

        # Cross terms: 2·sᵢ·sᵢ₊ₖ·sⱼ·sⱼ₊ₖ for all pairs i < j
        for i in range(num_spins - lag):
            for j in range(i + 1, num_spins - lag):
                # Check if indices collapse: when j == i + lag, Z_{i+lag}² = I
                # leaving only Z_i · Z_{j+lag} = Z_i · Z_{i+2·lag}
                if j == i + lag:
                    # Collapsed to ZZ term with weight 2
                    indices = (i, j + lag)
                    all_terms.add((2, indices))
                else:
                    # Full ZZZZ term with weight 4
                    indices = tuple(sorted((i, i + lag, j, j + lag)))
                    all_terms.add((4, indices))

    return list(all_terms), offset


def compute_labs_energy(spins: np.ndarray) -> float:
    """Compute the LABS energy for a spin configuration.

    The LABS energy is E = Σₖ Cₖ² where Cₖ is the autocorrelation at lag k:
    Cₖ = Σᵢ sᵢ·sᵢ₊ₖ

    Parameters
    ----------
    spins : np.ndarray
        Array of spins with values in {-1, +1}

    Returns
    -------
    float
        The LABS energy value
    """
    num_spins = len(spins)
    total_energy = 0.0

    for lag in range(1, num_spins):
        autocorr = np.dot(spins[:-lag], spins[lag:])
        total_energy += autocorr * autocorr

    return total_energy


def compute_labs_energy_from_bits(bits: np.ndarray) -> float:
    """Compute LABS energy from a bitstring (0/1 encoding).

    Converts {0, 1} bits to {+1, -1} spins and computes energy.

    Parameters
    ----------
    bits : np.ndarray
        Array of bits with values in {0, 1}

    Returns
    -------
    float
        The LABS energy value
    """
    spins = 1 - 2 * bits
    return compute_labs_energy(spins)

