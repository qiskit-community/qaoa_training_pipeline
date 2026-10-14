#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the scale-invariance round trip: scale, predict, unscale.

The two scaling steps need neither a bundle nor onnxruntime, so they are tested
here on their own. The property they exist to provide -- betas fixed, gammas
divided by the operator's scale -- is measured against a stubbed network in
``test_onnx_inference.py`` and through the provider in ``test_ai_inference.py``.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.inference.scaling import (
    denormalize_qaoa_params_np,
    normalize_cost_op,
    rescaling_factor,
    undo_gamma_rescale,
    undo_gamma_rescale_np,
)

from ..training_pipeline_test_case import TrainingPipelineTestCase


def make_extractor():
    """An extractor whose rescaling_factor method must track the function."""
    from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor

    return AIFeatureExtractor(in_features=[], norm_stats={})


class TestRescalingFactor(TrainingPipelineTestCase):
    """The factor is a function of the operator alone, and homogeneous in it."""

    OPERATOR = SparsePauliOp.from_list([("ZZI", -0.5), ("IZZ", -1.5), ("ZIZ", -0.25)])

    def test_the_function_and_the_method_agree(self):
        """The method is kept as a delegate, so it must not drift from it."""
        self.assertEqual(
            rescaling_factor(self.OPERATOR),
            make_extractor().rescaling_factor(self.OPERATOR),
        )

    def test_homogeneous_of_degree_one(self):
        """What makes the normalized features scale-independent."""
        base = rescaling_factor(self.OPERATOR)
        for scale in (0.1, 2.0, 10.0):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(
                    rescaling_factor(self.OPERATOR * scale), base * scale, places=6
                )

    def test_normalizing_twice_is_a_no_op(self):
        """A consequence of homogeneity, and why the chain is not applied twice."""
        factor = rescaling_factor(self.OPERATOR)
        self.assertAlmostEqual(rescaling_factor(self.OPERATOR / factor), 1.0, places=6)

    def test_a_degenerate_operator_falls_back_to_one(self):
        """An all-zero operator must not divide by zero."""
        self.assertEqual(rescaling_factor(SparsePauliOp.from_list([("ZZ", 0.0)])), 1.0)


class TestNormalizeCostOp(TrainingPipelineTestCase):
    """Step one: divide the operator by its factor, and report the factor."""

    OPERATOR = SparsePauliOp.from_list([("ZZI", -0.5), ("IZZ", -1.5), ("ZIZ", -0.25)])

    def test_returns_the_operator_divided_by_the_factor(self):
        """The two returns must be consistent with each other."""
        normalized, rescale_a = normalize_cost_op(self.OPERATOR)
        self.assertEqual(rescale_a, rescaling_factor(self.OPERATOR))
        np.testing.assert_allclose(
            np.real(normalized.coeffs), np.real(self.OPERATOR.coeffs) / rescale_a, rtol=1e-6
        )

    def test_the_normalized_operator_is_scale_free(self):
        """Any positive multiple of an operator normalizes to the same one."""
        base, _ = normalize_cost_op(self.OPERATOR)
        for scale in (0.1, 2.0, 10.0):
            with self.subTest(scale=scale):
                scaled, _ = normalize_cost_op(self.OPERATOR * scale)
                np.testing.assert_allclose(np.real(scaled.coeffs), np.real(base.coeffs), rtol=1e-6)


class TestUndoGammaRescale(TrainingPipelineTestCase):
    """Step three: the gammas are divided by the factor, the betas are not."""

    def test_only_the_second_half_moves(self):
        """The split is what output_dim decides: p betas, then p gammas."""
        angles = undo_gamma_rescale([0.1, 0.2, 0.3, 0.4], p=2, rescale_a=2.0)
        self.assertEqual(angles, [0.1, 0.2, 0.15, 0.2])

    def test_returns_plain_floats(self):
        """The provider puts these in a ParamResult, so no numpy scalars."""
        angles = undo_gamma_rescale(np.asarray([0.1, 0.4], dtype=np.float32), p=1, rescale_a=4.0)
        self.assertTrue(all(isinstance(value, float) for value in angles))

    def test_agrees_with_the_numpy_spelling(self):
        """The list form is a convenience, not a second implementation."""
        raw = [0.1, 0.2, 0.3, 0.4]
        np.testing.assert_allclose(
            undo_gamma_rescale(raw, p=2, rescale_a=3.0),
            undo_gamma_rescale_np(np.asarray(raw), p=2, rescale_a=3.0),
            rtol=1e-6,
        )

    def test_a_factor_of_one_changes_nothing(self):
        """What a degenerate operator falls back to, so it must be a no-op."""
        self.assertEqual(undo_gamma_rescale([0.1, 0.9], p=1, rescale_a=1.0), [0.1, 0.9])


class TestOutputDenormalization(TrainingPipelineTestCase):
    """The bundle's own output scaling, which is part of the predict step."""

    def test_scales_every_angle(self):
        """Unlike the gamma un-rescale, this one applies to betas too."""
        np.testing.assert_allclose(
            denormalize_qaoa_params_np(np.asarray([1.0, 2.0]), scale=0.5), [0.5, 1.0], rtol=1e-6
        )
