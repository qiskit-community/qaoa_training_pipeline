# Conventions

This repository follows the following conventions.
All code, tests, notebooks and docs must comply with it.
By default we minimize the cost function $f(x)$ and by consequence we minimize the energy of the cost Hamiltonian $H_C$ to find its ground state.
Bits are mapped to spins via $x_i = (1 − z_i)/2$.
Therefore, $x_i = 1$ ⇔ $z_i=-1$ which corresponds to the $|1⟩$ state with $Z$ eigenvalue of $-1$.
Qubit registers are ordered following little Endian.
Classical bit `i` holds $x_i$ and candidate solutions $x$ to $f(x)$ are zero-indexed, i.e., variable $x_i$ corresponds to `x[i]` where `x` is a list.
To follow the adiabatic theorem we therefore chose an initial
state which is the *ground state* of the mixer $H_M$. 


## Standard QAOA
The default circuit is constructed as
$|ψ(θ)⟩ = U_M(β_p) U_C(γ_p) ⋯ U_M(β_1) U_C(γ_1) |ψ_0⟩$, i.e., cost layer first, then mixer.
Warm-starts for QAOA are discussed [below](#warm-starts).
The cost layer, mixer and initial state follow the conventions

1. **Cost layer**: $U_C(γ) = \exp(−iγ H_\textrm{PS})$ where $H_\textrm{PS}$ denotes the phase separator. 
   By default the phase-separator is the cost operator, i.e., $H_\textrm{PS}=H_C$.
   No factor ½. 
   Any normalization of the phase separator $H_\textrm{PS}$ is the responsibility of the user. 
   This implies that terms $J_{ij}Z_iZ_j$ and $h_iZ_i$ in $H_\textrm{PS}$ result in the gates $R_{ZZ}(2γJ_{ij})$ and $R_{Z}(2γh_{i})$, respectively.

2. **Initial state**: The initial state is the equal superposition state $|ψ_0⟩ = ⊗_i H|0⟩ = ⊗_i |+⟩$.

3. **Mixer**: $U_M(β) = \exp(−iβ H_M)$. 
   No factor ½. 
   The default mixer is $H_M=-\sum_iX_i$ such that its ground state $|+⟩$ is the initial state.
   This default mixer will result in $R_X(-2β)$ gates in the circuit.


## Parameters
- Parameter vectors are given as $θ = [β_1, …, β_p, γ_1, …, γ_p]$, i.e., the $β$ block comes first.
- Every API that takes or returns $β$ and $γ$ separately lists $β$ first.
- Canonical form of angles $γ_1 ≥ 0$ and $β_k \in [0, π)$ for all $k$ is suggested but not enforced.
  Minimizing from the ground state of the mixer typically gives $γ > 0$ and $β>0$.
  The canonical form can be achieved by potentially replacing $(β, γ)$ with $(-β, -γ)$ and then replacing $β$ by $β + kπ$ with suitably chosen $k \in \mathbb{N}$.

## Warm-starts

The concept of a warm-start is outlined in [Egger et al. (2021)](https://quantum-journal.org/papers/q-2021-06-17-479/).
We now spell out the conventions used in the present repository.
We assume that $c$ is the continuous solution to the relaxed problem.

1. **cost layer**: Unchanged from the default outlined above.

2. **initial state**: The initial state is $|ψ_0⟩ = ⊗_i R_Y(ϑ_i)|0⟩ = ⊗_i (\sqrt{1−c_i}|0⟩ + \sqrt{c_i}|1⟩)$ with $ϑ_i = 2 \arcsin \sqrt{c_i}$.
Therefore, under $c_i=1/2~\forall~i$ the initial state is the equal superposition state $|+⟩$.
This allows us to recover [Standard QAOA](#standard-qaoa) with the same conventions.

3. **mixer**: The initial state is the ground state of the mixer $H_M = -\sum_in^T(ϑ_i)σ_i$ with $n_i = (\sin ϑ_i, 0, \cos ϑ_i)$ and $σ^T_i=(X_i,Y_i,Z_i)$. 
The mixer operator is thus $\exp(-iβH_M) = Π_i R_Y(ϑ_i) R_Z(-2β) R_Y(-ϑ_i)$. I.e., in a circuit we apply $R_Y(-ϑ_i)$ first. 
For the default QAOA, i.e., $c_i= 1/2$ and $ϑ = π/2$, we have $n_i = (1, 0, 0)$, so $U_M(β) = \exp(−iβ (-1)Σ_iX_i) = Π_i R_X(-2β)$ which is consistent with the conventions for the default QAOA.
In edge cases, a warm-start may cause issues since qubits with $c_i\in\{0, 1\}$ are frozen because the mixer is a Pauli-$Z$ rotation and commutes with the phase separator of QAOA.
