# Training methods

This folder contains a set of configurations for training.
Each file is referred to as a method of training.
All training methods in this folder should be able to run as-is, i.e., without
additional command line inputs.
However, it is expected that additional command line inputs will help refine the
training.

# Brief description of the configuration files

 1. `train_method_0.json`: sample configuration file for simulating a QAOA circuit with
    the `EfficientDepthOneEvaluator` method (which is specifically optimized for depth-1
    circuits), and optimizing it with the `DepthOneScanTrainer` method, which constructs
    a two-dimensional grid representation of the cost function, and finds the lowest-energy
    point. The grid parameters can be passed through the `train_kwargs` entry.

 2. `train_method_1.json`: sample configuration file for simulating a QAOA circuit with
    the `EfficientDepthOneEvaluator` method and optimizing it with the `ScipyTrainer` method,
    which invokes a Scipy optimizer (the specific optimized can be selected through the
    `minimize_args` entry -- by default the COBYLA optimizer is used). The Scipy optimizers
    are iterative and, for this reason, we pass through the `train_kwargs` entry the initial
    guess as `params0`.
 
 4. `train_method_3.json`: sample configuration file for simulating a QAOA circuit with
    the `EfficientDepthOne` trainer and by concatenating the optimization methods listed
    above.

 5. `train_method_4.json`: sample configuration file for simulating a QAOA circuit with
    the Matrix Product State (MPS) method. The accuracy of this simulator is tuned by
    the MPS bond dimension (selected through the `bond_dim_circuit` entry) and by the
    singular value decomposition (SVD) truncation threshold (`threshold_circuit`)

 5. `train_method_5.json`: sample configuration file for simulating a QAOA circuit with
    the `ReweightingTrainer` trainer. This optimizer first determines the optimal parameter
    for the QAOA circuit by removing the weights in the cost function, and then uses the
    resulting parameter as initial guess for the "true" optimization.

Different trainers can be concatenated, as shown in the `example_config.json` and
`example_config_concatenated_mps.json` configuration files. Specifically:

 1. The `example_config.json` configuration files optimizes the depth-one circuit as done
    in `train_method_0.json`, and then optimizes the corresponding depth-2 circuit using
    the `TransitionStatesTrainer`, which implements the method described in
    PRA, 107, 062404 (2023).

 2. The `example_config_concatenated_mps.json` configuration file corresponds to two
    concatenated MPS-based optimizations -- the first one with a rather low SVD truncation
    parameter (0.01), and the second one with a tighter truncation (0.001).

 3. The `example_config_mps_ts.json` file implements a three-step optimization strategy for
    a depth-three circuit. The first layer is optimized with the efficient depth-1 method.
    Then, the subsequent two layers are optimized with the MPS simulator, using the transition
    state method to transfer the results obtained for a given layer to the subsequent layer.

Note that, when concatenating different optimizers, one may want to pass the results obtained
from a given optimization step to the following one. This is done through the `result` entry
of the `train_kwargs`. Consider the case (Example 1 here above) where one wants to use the
optimized parameters obtained from a depth-1 optimization as input for the transition state
optimizer. The optimal parameters are associated, in the result file, with the `optimized_params`
keyword. To propagate it to the transition state trainer, one must:

 1. Add an entry named `result` in the `train_kwargs` section of the json file.
 2. Within this section, add `"optimized_params": "previous_optimal_point"`.
    This indicates that the `previous_optimal_point` parameter for the transition-state optimizer
    should be set to the value stored under the `optimized_params` keyword in the results
    of the previous optimization.
