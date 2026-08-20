# QAOA training pipeline
This repository is a set of tools to generate good parameters for QAOA Ansatz circuits with classical means.

## Motivation

QAOA seeks to find good solutions to combinatorial optimization problems.
This is done by sampling from a quantum circuit that has trained parameters $\gamma$ and $\beta$.
Since training on quantum hardware is costly and time-intensive, it should be a last resort.
This repository has multiple methods to train the parameters on classical hardware.
In more detail, this repository contains the following.

* Classically efficient methods to evaluate the energy of a QAOA circuit such as matrix product states and light-cones. Note that some of the methods exactly evaluate the energy (e.g., with light-cones) while others (e.g., matrix product states) approximate it.
  See the `evaluation` module for details.
* Methods to train the parameters in the QAOA circuit. These trainers can be implementations of
  known methods published in the literature. The trainers typically rely on an energy evaluator.
  See the `training` module for more details.
* Utility functions to help with data manipulation.
  See the `utils` module for more details.

## Conventions

Technically, QAOA can be defined with various prefactors in the exponentials of the involved Hamiltonians.
Here, we carefully outline the conventions used by this API.
This repository assumes that standard QAOA applies the variational circuit

$$\prod_{k=1}^p \exp{(-i\beta_k H_M)}\exp{(-i\gamma_k H_C)}\left\vert+\right\rangle^{\otimes n}$$

This definition follows the [original QAOA paper](https://arxiv.org/pdf/1411.4028) by Fahri et al. in which the energy of the cost function is **maximized**.
This definition also aligns with the `QAOAAnsatz` class in Qiskit up to the mixer operator.
Here, the important convention is in the exponential functions, i.e., $-i\beta_k H_M$ and $-i\gamma_k H_C$.
By default, i.e., when not otherwise explicitly specified, we assume that the mixer Hamiltonian is defined by

$$H_M = \sum_{i=0}^{n-1} X_i$$

This mixer has $\left\vert+\right\rangle^{\otimes n}$ as the highest excited state.
We thus assume that we are **maximizing** the energy of the cost operator $H_C$.
The users of this repository can interact with the trainers by calling their `train` method.
```
result = trainer.train(cost_op, ...)
```
Here, `trainer` is an instance of a trainer class in the `training` module.
The `cost_op` is a cost operator and is an instance of a `SparsePauliOp` as found in Qiskit.
By convention, the parameters returned by `train` are in the order `[beta_1, ..., beta_p, gamma_1, ..., gamma_p]`.
It is also possible to minimize the energy of `cost_op` by appropriatly initializing the trainer class.

### Input problems

This framework supports optmization problems of the form
$$\max_{x\in\{0,1\}^n}f(x).$$
We exemplify the conversion of such an optimization problem to an Ising Hamiltonian with a QUBO defined as
$$\max_{x\in\{0,1\}^n}\sum_{i,j=0}^{n-1}Q_{ij}x_ix_j.$$
Here, the $n\times n$ matrix $Q$ specifies the QUBO.
With the change of variable $2x_i=1-z_i$ we can rewrite the objective as
$$\frac{1}{4}\max_{x\in\{0,1\}^n}\sum_{i,j=0}^{n-1}Q_{ij}(z_iz_j-z_i-z_j+1).$$
This optimization problem is then related to an Ising Hamiltonian by replacing each $z_i$ with a Pauli-Z operator $Z_i$.
To specify an input problem, the user provides a list of hyper-edges and coefficients.
For example, the input
```
{
    "edge list": [
        {"nodes": [0, 1, 2], "weight": -1.0},
        {"nodes": [0, 2], "weight": 2.0},
    ],
    "Description": "Example of a problem specification."
}
```
will be mapped to the Hamiltonian $H_C=2\cdot ZIZ - 1\cdot ZZZ$.
We now examplify how the Maximum Cut (MaxCut) objective function relates to the Hamiltonian $H_C$.
For a graph $G=(V,E)$, the objective of MaxCut is to maximize the value of the cut
$$C_\text{max}=\max_{x\in\{0,1\}^n}\sum_{i,j=0}^{n-1}w_{ij}x_i(1-x_j)=\sum_{i,j=0}^{n-1}\frac{w_{ij}}{4}+\max_{z\in\{-1, 1\}^n}\sum_{i,j=0}^{n-1}-\frac{w_{ij}}{4}z_iz_j.$$
Here, $w_{ij}$ is the weight of edge $(i,j)\in E$.
Therefore, for MaxCut, we maximize the energy of the Hamiltonian
$$H_C=\sum_{(i,j)\in E}-\frac{w_{ij}}{2}Z_iZ_j.$$
Here, the sum runs over the edges $E$ in the graph and not over $i,j=0$ to $n-1$.
Therefore, we have a factor of $1/2$ instead of $1/4$.
With this convention, the energy $\langle H_C\rangle$ is related to the value of the cut $C$ by $C=\langle H_C\rangle+\sum_{(i,j)\in E}w_{ij}/2$.

## Example usage

The classes in this repository can be used as is in scripts and notebooks.
Examples are provided in the `how_tos`.
In addition, the `train.py` file provides an entry point for command line usage.
When using `train` the user must specify the graph and the method with which to train.
The method specifies the trainers that will be called one after another to find good QAOA parameters.
For convenience, the method is typically contained in JSon file with examples found in the `methods folder`.
The trainers may rely on evaluators to compute the energy of the QAOA circuit (or an approximation thereof).
Performing command line training will require commands of the form
```bash
python -m train --input path_to_graph.json --config path_to_method.json --save --save_dir dir_path --save_file results.json --pre_factor -0.5
```
Additional arguments can be given as described in `train.py`.
Finally, the output of the training is a dictionnary with the results.
The "energy" is the best found energy and "optmized_params" are the parameters found from the optimization.
When running from the command line these results are saved in a JSon file.
The example above corresponds to a MaxCut optimization problem where the input is given as a graph.
Therefore, we specify `--pre_factor -0.5` to convert the graph to the MaxCut Hamiltonian.
I.e., upon data loading the pre-factor of `-0.5` is multipled to each edge to obtain the MaxCut cost function discussed above.
The pre-factor does not need to be specified (it defaults to 1.0) if the problem in the input JSon has the weights of the intended Ising Hamiltonian.

Files with pre-configured methods to train are found under the folder `data/methods/`.
Below we show an example of a training method that uses SciPy to optimize the parameters and evaluate the energy with matrix product states.
We can chain multiple trainers one after another.
For each entry we specify the trainer (i.e., the name of the class), and the initialization arguments `trainer_init` for the trainer.
In addition we can specify run-time arguments under `train_kwargs`.
```
{
    "trainer_chain": [
        {
            "trainer": "ScipyTrainer",
            "trainer_init": {
                "evaluator": "MPSEvaluator",
                "evaluator_init": {
                    "bond_dim_circuit": 24
                },
                "minimize_args": {
                    "options": {
                        "maxiter": 20,
                        "rhobeg": 0.2
                    }
                }
            },
            "train_kwargs": {
                "params0": [0, 0]
            },
            "save_file": "example_result.json"
        }
    ],
    "description": "Use the MPS evaluation with the scipy trainer."
}
```
The result of the optimization is saved in the Json file `example_result.json`.
This file will contain information on the training history and the method employed.

## Installation

Install from a clone of the repository with:

```bash
pip install .
```

For a development install (editable, with all optional dependencies a contributor would need):

```bash
pip install -e ".[dev]"
```

A `uv.lock` file is provided for reproducible installs. If you have [`uv`](https://docs.astral.sh/uv/) available, `uv sync --extra dev` will create a virtual environment and install all development dependencies pinned to the exact versions in the lock file.

### Optional dependency groups

The package declares several optional groups that can be combined as needed:

| Extra | Contents | When you need it |
|-------|----------|-----------------|
| `tns` | `quimb`, `juliacall` | Tensor-network evaluators (`MPSEvaluator`) and Pauli propagation (`PPEvaluator`) |
| `sat` | `python-sat` | SAT-based qubit-mapping pre-processor (`SATMapper`) |
| `cplex` | `qiskit-optimization[cplex]` | Exact MaxCut solver via CPLEX (`solve_max_cut`) |
| `notebooks` | `qiskit-aer`, `qiskit-ibm-runtime` | Running the `how_tos` notebooks |
| `test` | `ddt` | Running the test suite |
| `ci` | `stestr`, `coverage` | CI-only test runner and coverage tooling |
| `lint` | `black`, `ruff` | Code formatting and linting |
| `dev` | all of the above except `ci` | Local development |

Install one or more extras with e.g.:

```bash
pip install -e ".[tns,notebooks]"
```

### Pauli propagation (Julia)

`PPEvaluator` requires the `PauliPropagation` Julia library, which integrates with Python through `juliacall` (included in the `tns` extra).
The first time `PPEvaluator` is used, `pauli_propagation.py` will automatically install `PauliPropagation.jl` into the Julia environment that `juliacall` maintains at `<venv>/julia_env/`.
Note that `PauliPropagation.jl` is under active development and future updates may introduce breaking changes.

To update Julia packages within that environment:

```python
from juliacall import Main as jl
jl.seval("using Pkg; Pkg.update()")
```

## Warning

This repository is still in development: new functionality is being added and the API is subject to change.

## Version tracking

| Version | Added functionality                                          | Pull request |
|---------|--------------------------------------------------------------|--------------|
|       1 | Track system information                                     |           #8 |
|       2 | Add history mix-in                                           |          #11 |
|       3 | Add fidelity bounds for MPS                                  |           #6 |
|       4 | Add QAOA angles functions                                    |          #14 |
|       5 | Switch to qaoa_ansatz                                        |          #16 |
|       6 | Add Pauli Propagation                                        |          #15 |
|       7 | Add problem class in train                                   |          #17 |
|       8 | Improve train tests                                          |          #22 |
|       9 | Add SAT map pre-processing                                   |          #19 |
|      10 | Add recursive transition states trainer                      |          #20 |
|      11 | Bug fix in train pre-processing data                         |          #24 |
|      12 | More data in result saving in train.py                       |          #26 |
|      13 | Create PPEvaluator from configs                              |          #25 |
|      14 | Custom ansatz operator to state vector                       |          #29 |
|      15 | Remove python 3.9 support                                    |          #31 |
|      16 | Fix TQATrainer qaoa_angles_function and returned ParamResult |          #35 |
|      17 | Add QAOA-PCA and data based trainers                         |          #32 |
|      18 | Allow nbr of Fourier coefficients to scale with QAOA depth   |          #27 |
|      19 | Add an interface to the Qiskit Aer and its MPS evaluator     |          #36 |
|      20 | Add linear ramp parameter optimization to the TQA trainer    |          #37 |
|      21 | Fixed degree computation in the fixed angles trainer         |          #43 |
|      22 | SV simulation moved to Qiskit Aer. GPU Support for SV        |          #44 |
|      23 | Improve transparency of transfer trainer                     |          #45 |
|      24 | Adding linear ramp parameter support in train.py             |          #46 |
|      25 | Enable GPU on SV simulation via init_kwargs "GPU"            |          #47 |
|      26 | Clean up pylance warning (type hinting, not None, etc.)      |          #48 |
|      27 | Add LABS with GPU support                                    |          #50 |
|      28 | Add a trainer that computes optimal beta per gamma           |          #49 |
|      29 | Bug fix - Add vertices to the graph in operator_to_graph     |          #52 |
|      30 | Convert a graph with partial assignment to a hamiltonian     |          #53 |
|      31 | Refactor linear angle interpolation (increase QAOA depth)    |          #55 |
|      32 | Standardize the inheritance of evaluators and trainers       |          #57 |
|      33 | Allow Pauli propagation to accept a custom circuit ansatz    |          #60 |
|      34 | ScipyTrainer uses a custom initial_state.                    |          #62 |
|      35 | Bug fix - Initial state used in computing correlates.        |          #64 |
|      36 | Bug fix - Fix Julia issues and installation of Julia in CI.  |          #67 |
|      37 | Refactoring: Add parameter providers and pipeline components |          #66 |
|      38 | Introduce the framework module                               |          #72 |
|      39 | Add framework imports in the main folder                     |          #73 |
|      40 | Add FromConfigParamsProvider to set initial params.          |          #74 |
|      41 | Implement new refactoring design in trainers.                |          #71 |
|      42 | Refactor how_tos to new class design.                        |          #80 |
|      43 | Update provide_params in reweighting                         |          #81 |
|      44 | Streamline MPSAerEvaluator by using native Aer backend       |          #78 |
|      45 | Update dependencies to pyproject.toml with optional groups   |          #88 |
|      46 | Add QPU evaluator to draw samples from HW backend.           |          #76 |
|      47 | Add MPS sampling based energy estimation.                    |          #75 |
|      48 | Update printing method for ParamResult.                      |          #82 |

## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support.
IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.
