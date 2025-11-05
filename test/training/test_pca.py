#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PCA-based training."""

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation import StatevectorEvaluator
from qaoa_training_pipeline.training import QAOAPCA
from qaoa_training_pipeline.training.data_loading import TrivialDataLoader

from test import TrainingPipelineTestCase


class TestPCA(TrainingPipelineTestCase):
    """Tests of the PCA trainer."""

    def setUp(self):
        # Generated for six-node random-three regular graphs with TQATrainer and
        # ScipyTrainer for depth-four QAOA.
        self._data_base = [
            [1.168, 0.4472, 0.3285, 0.1618, -0.0779, 0.4183, 0.8593, 1.0559],
            [1.2815, 0.4914, 0.5664, 0.2921, -0.1632, 0.4775, 0.6536, 0.821],
            [1.0228, 0.4784, 0.3359, 0.1657, -0.0451, 0.3898, 0.8605, 1.0482],
            [1.2016, 0.4381, 0.3263, 0.1613, -0.0904, 0.4299, 0.856, 1.0511],
            [1.0388, 0.4759, 0.335, 0.1654, -0.0487, 0.3912, 0.861, 1.0492],
        ]

        self._qaoa_depth = len(self._data[0]) // 2

        self._cost_op = SparsePauliOp.from_list(
            [
                ("IIIIZZ", (-0.5 + 0j)),
                ("IZIIIZ", (-0.5 + 0j)),
                ("IIZIIZ", (-0.5 + 0j)),
                ("IIIZZI", (-0.5 + 0j)),
                ("ZIIIZI", (-0.5 + 0j)),
                ("IIZZII", (-0.5 + 0j)),
                ("ZIIZII", (-0.5 + 0j)),
                ("IZZIII", (-0.5 + 0j)),
                ("ZZIIII", (-0.5 + 0j)),
            ]
        )

    def test_simple(self):
        """Test that the workflow runs."""

        data_loader = TrivialDataLoader(self._data)
        n_pcs = 2

        pca = QAOAPCA(data_loader, num_components=n_pcs, evaluator=StatevectorEvaluator())

        result = pca.train(self._cost_op, params0=[0] * n_pcs)

        self.assertEqual(len(result["optimized_params"]), n_pcs)
        self.assertEqual(len(result["optimized_qaoa_angles"]), 2 * self._qaoa_depth)

    def test_from_config(self):
        """Test that we can setup from a config."""
        config = {
            "trainer": "QAOAPCA",
            "trainer_init": {
                "evaluator": "MPSEvaluator",
                "evaluator_init": {
                    "bond_dim_circuit": 24,
                    "use_vidal_form": True,
                    "threshold_circuit": 0.001,
                },
                "data_loader": "TrivialDataLoader",
                "data_loader_init": {"data": [[1, 1, 1, 1], [2, 2, 2, 2]]},
                "minimize_args": {"options": {"maxiter": 20, "rhobeg": 0.2}},
                "num_components": 1,
            },
        }

        trainer = QAOAPCA.from_config(config)
        self.assertTrue(isinstance(trainer, QAOAPCA))
