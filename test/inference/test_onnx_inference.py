#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""End-to-end tests for the torch-free ONNX inference path.

Covered entrypoint:
  * OnnxQAOAPredictor (qaoa_training_pipeline/inference/onnx_predictor.py)

Two kinds of test:
  * Runtime-only tests need just the exported ``model.onnx`` (no torch, no
    checkpoint). They auto-skip when the artifact is absent.
  * The baseline-regression test compares against frozen golden values and
    likewise needs neither torch nor a checkpoint.

``onnxruntime`` is an optional dependency (the ``inference`` extra), so the whole
module skips if it is not importable.
"""

import importlib.util
import json
import math
import unittest
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp

from ..training_pipeline_test_case import TrainingPipelineTestCase

HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None

MODEL_CONFIGS_DIR = (
    Path(__file__).resolve().parents[2] / "qaoa_training_pipeline" / "inference" / "model_configs"
)
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

MODEL_NAMES = [
    "agg_transformer",
    "diffusion_transformer",
    "edge_transformer",
    "gcn",
    "graph_isomorphism_network",
    "graph_neural_network",
    "graph_transformer",
]

# Graph-consuming model exercised by the behavioral tests alongside the
# scalar-only agg_transformer.
BEHAVIORAL_MODELS = ["agg_transformer", "graph_neural_network"]

# The graph_transformer's Laplacian positional encoding uses a different
# eigensolver (numpy vs torch); its eigenvectors are sign-ambiguous, so the
# model was trained with random sign-flip augmentation and tolerates the
# difference — hence a looser tolerance for it.
PARITY_ATOL = {"graph_transformer": 2e-3}
DEFAULT_PARITY_ATOL = 1e-4


# --- deterministic cost operators (mirrors tools/inference/bench_ops.py) ----


def _zz(num_qubits, i, j, weight=1.0):
    label = ["I"] * num_qubits
    label[i] = "Z"
    label[j] = "Z"
    return "".join(label), weight


def _ring(n, weight=1.0):
    return SparsePauliOp.from_list([_zz(n, k, (k + 1) % n, weight) for k in range(n)])


def _line(n, weight=1.0):
    return SparsePauliOp.from_list([_zz(n, k, k + 1, weight) for k in range(n - 1)])


def _complete(n, weight=1.0):
    return SparsePauliOp.from_list(
        [_zz(n, i, j, weight) for i in range(n) for j in range(i + 1, n)]
    )


def _weighted_ring(n):
    return SparsePauliOp.from_list([_zz(n, k, (k + 1) % n, 0.5 + 0.25 * k) for k in range(n)])


BENCH_OPS = {
    "triangle_3": _complete(3),
    "line_4": _line(4),
    "ring_4": _ring(4),
    "complete_4": _complete(4),
    "ring_6": _ring(6),
    "line_8": _line(8),
    "weighted_ring_6": _weighted_ring(6),
    "complete_5": _complete(5),
}


def config_path(model_name):
    """Path to a model's config bundle."""
    return MODEL_CONFIGS_DIR / model_name / "model_config.json"


def onnx_exists(model_name):
    """True if the exported ONNX artifact for a model is present on disk."""
    return (MODEL_CONFIGS_DIR / model_name / "model.onnx").is_file()


@unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime not installed (install the 'inference' extra)")
class TestOnnxInference(TrainingPipelineTestCase):
    """Torch-free ONNX predictor tests. Skip per-model when artifacts absent."""

    def _predictor(self, model_name):
        from qaoa_training_pipeline.inference.onnx_predictor import OnnxQAOAPredictor

        return OnnxQAOAPredictor(config_path=config_path(model_name), device="cpu")

    def test_onnx_predictor_loads_and_predicts(self):
        """Every exported model loads via onnxruntime and produces p=1 angles."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                if not onnx_exists(model_name):
                    self.skipTest(f"model.onnx for {model_name!r} not present")
                angles = self._predictor(model_name).predict(op)
                self.assertIsInstance(angles, list)
                self.assertEqual(len(angles), 2)  # p=1 -> [beta, gamma]
                self.assertTrue(all(isinstance(a, float) and math.isfinite(a) for a in angles))

    def test_onnx_predictor_is_deterministic(self):
        """Inference is deterministic: same input -> identical output."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_name in BEHAVIORAL_MODELS:
            with self.subTest(model=model_name):
                if not onnx_exists(model_name):
                    self.skipTest(f"model.onnx for {model_name!r} not present")
                predictor = self._predictor(model_name)
                self.assertEqual(predictor.predict(op), predictor.predict(op))

    def test_onnx_predictor_reacts_to_input(self):
        """Different problem graphs yield different predicted angles."""
        op_triangle = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        op_line4 = SparsePauliOp.from_list([("ZZII", 1.0), ("IZZI", 1.0), ("IIZZ", 1.0)])
        for model_name in BEHAVIORAL_MODELS:
            with self.subTest(model=model_name):
                if not onnx_exists(model_name):
                    self.skipTest(f"model.onnx for {model_name!r} not present")
                predictor = self._predictor(model_name)
                self.assertNotEqual(predictor.predict(op_triangle), predictor.predict(op_line4))

    def test_onnx_predictor_raw_vs_denormalized_differ(self):
        """denormalize=False returns the raw (unscaled) model output."""
        op = SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0)])
        for model_name in BEHAVIORAL_MODELS:
            with self.subTest(model=model_name):
                if not onnx_exists(model_name):
                    self.skipTest(f"model.onnx for {model_name!r} not present")
                predictor = self._predictor(model_name)
                raw = predictor.predict(op, denormalize=False)
                scaled = predictor.predict(op, denormalize=True)
                # config output_scale = pi/2, so scaled == raw * pi/2 for the betas
                # (first half); the gammas additionally undergo the rescale_a
                # division, so compare only the betas here (p=1 -> index 0).
                p = predictor.output_dim // 2
                for raw_beta, scaled_beta in zip(raw[:p], scaled[:p]):
                    self.assertAlmostEqual(scaled_beta, raw_beta * (math.pi / 2), places=5)

    def test_onnx_matches_baseline(self):
        """The ONNX predictor reproduces the frozen torch baseline for every op.

        Baselines are generated by tools/inference/gen_baselines.py from the torch
        predictor; regenerate them when the checkpoints or model code change.
        """
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                if not onnx_exists(model_name):
                    self.skipTest(f"model.onnx for {model_name!r} not present")
                baseline_file = BASELINE_DIR / f"{model_name}.json"
                if not baseline_file.is_file():
                    self.skipTest(f"no baseline for {model_name!r}")

                baseline = json.loads(baseline_file.read_text())
                predictor = self._predictor(model_name)
                atol = PARITY_ATOL.get(model_name, DEFAULT_PARITY_ATOL)

                for case_name, expected in baseline["cases"].items():
                    got = predictor.predict(BENCH_OPS[case_name])
                    for got_angle, expected_angle in zip(got, expected):
                        self.assertAlmostEqual(
                            got_angle,
                            expected_angle,
                            delta=atol,
                            msg=f"{model_name}/{case_name}: ONNX drifted from baseline",
                        )
