#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to test the fixed-angles trainer."""
from importlib import import_module
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.training.fixed_angle_conjecture import FixedAngleConjecture
from qaoa_training_pipeline.evaluation.statevector_evaluator import StatevectorEvaluator
from qaoa_training_pipeline.utils.graph_utils import solve_max_cut
from ..training_pipeline_test_case import TrainingPipelineTestCase


class TestFixedAngleConjecture(TrainingPipelineTestCase):
    """Class to test the FixedAngleConjecture trainer."""

    def setUp(self):  # pylint: disable=invalid-name
        """Setup the class."""
        self.cost_op = SparsePauliOp.from_list(
            [
                ("IIZZ", -0.5),
                ("IZIZ", -0.5),
                ("ZIIZ", -0.5),
                ("IZZI", -0.5),
                ("ZIZI", -0.5),
                ("ZZII", -0.5),
            ]
        )

    def test_train(self):
        """Test the we can get angles."""

        trainer = FixedAngleConjecture(reps=2)

        result = trainer.provide_params(self.cost_op)

        self.assertListEqual(
            result["optimized_params"],
            [0.5550603400685824, 0.29250781484335187, 0.4877097327098487, 0.8979876956225422],
        )

    def test_degree_one_local(self):
        """One local-terms should not interfere with the degree computation."""
        one_local_op = SparsePauliOp.from_list(
            [
                ("IIIZ", 1),
                ("IIZI", 1),
                ("IZII", 1),
                ("ZIII", 1),
                ("IIZZ", 1),
                ("IZIZ", 1),
                ("ZIIZ", 1),
                ("IZZI", 1),
                ("ZIZI", 1),
                ("ZZII", 1),
            ]
        )

        result = FixedAngleConjecture(reps=2).provide_params(
            one_local_op,
        )
        self.assertEqual(result["data_key"], (2, 3))

    def test_energy(self):
        """Test the we can get the energy."""
        trainer = FixedAngleConjecture(reps=2)
        evaluator = StatevectorEvaluator()

        result = trainer.provide_params(self.cost_op)
        energy = evaluator.evaluate(cost_op=self.cost_op, params=result["optimized_qaoa_angles"])

        # CPLEX installation can be unreliable in CI.
        try:
            import_module("cplex")
            has_cplex = True
        except ImportError:
            has_cplex = False

        if has_cplex:
            _, _, aprrox_ratio = solve_max_cut(self.cost_op, energy)
        else:
            aprrox_ratio = 0.86081  # This is the value that the line above yields.

        self.assertGreater(aprrox_ratio, result["metadata"]["approximation_ratio"])

    def test_parse_train_args(self):
        """Test the parsing of the training arguments."""
        train_args = FixedAngleConjecture.parse_runtime_kwargs("reps:2")

        self.assertDictEqual(train_args, {"reps": 2})

    def test_from_config(self):
        """Test that we can create fixed angle trainers from configs."""
        config = {"reps": 2}

        trainer = FixedAngleConjecture.from_config(config)
        self.assertIsInstance(trainer, FixedAngleConjecture)
        self.assertEqual(trainer._qaoa_depth, 2)
