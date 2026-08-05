#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test base case for the training pipeline repository."""

import unittest
import warnings

from qiskit.exceptions import QiskitWarning


class TrainingPipelineTestCase(unittest.TestCase):
    """Training pipeline test case."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._warning_context = warnings.catch_warnings()
        cls._warning_context.__enter__()
        warnings.filterwarnings("error", category=DeprecationWarning)
        warnings.filterwarnings("error", category=QiskitWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

        # Remove filters after Qiskit 2.0 pin is removed
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*The property.*qiskit.*duration.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*The property.*qiskit.*unit.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*The property.*qiskit.*condition.*",
        )

    @classmethod
    def tearDownClass(cls):
        cls._warning_context.__exit__(None, None, None)
        super().tearDownClass()
