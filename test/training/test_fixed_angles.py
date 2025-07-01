#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to test the fixed-angles trainer."""

from importlib import import_module
from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_evaluator import MPSEvaluator
from qaoa_training_pipeline.evaluation.statevector_evaluator import StatevectorEvaluator
from qaoa_training_pipeline.training.fixed_angle_conjecture import FixedAngleConjecture
from qaoa_training_pipeline.utils.graph_utils import solve_max_cut


class TestFixedAngleConjecture(TrainingPipelineTestCase):
    """Class to test the FixedAngleConjecture trainer."""

    def setUp(self):
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

        trainer = FixedAngleConjecture()

        result = trainer.train(self.cost_op, reps=2)

        self.assertListEqual(
            result["optimized_params"],
            [0.5550603400685824, 0.29250781484335187, 0.4877097327098487, 0.8979876956225422],
        )

    def test_energy(self):
        """Test the we can get the energy."""
        trainer = FixedAngleConjecture(StatevectorEvaluator())

        result = trainer.train(self.cost_op, reps=2)

        # CPLEX installation can be unreliable in CI.
        try:
            import_module("cplex")
            has_cplex = True
        except ImportError:
            has_cplex = False

        if has_cplex:
            _, _, aprrox_ratio = solve_max_cut(self.cost_op, result["energy"])
        else:
            aprrox_ratio = 0.86081  # This is the value that the line above yields.

        self.assertGreater(aprrox_ratio, result["approximation ratio"])

    def test_from_config(self):
        """Test that we can create fixed angle trainers from configs."""
        config = {}

        trainer = FixedAngleConjecture.from_config(config)
        self.assertIsNone(trainer.evaluator)

        config = {"evaluator": "MPSEvaluator", "evaluator_init": {"bond_dim_circuit": 2}}

        trainer = FixedAngleConjecture.from_config(config)
        self.assertTrue(isinstance(trainer.evaluator, MPSEvaluator))

        self.assertEqual(trainer.evaluator.to_config()["bond_dim_circuit"], 2)

    def test_parse_train_args(self):
        """Test the parsing of the training arguments."""
        trainer = FixedAngleConjecture.from_config({})
        train_args = trainer.parse_train_kwargs("2_none")

        self.assertDictEqual(train_args, {"reps": 2, "degree": None})
