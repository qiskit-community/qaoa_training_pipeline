#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Torch-free QAOA angle inference.

Exposes :class:`AIInference`, a
:class:`~qaoa_training_pipeline.framework.ProblemParamsProvider` that predicts
QAOA angles from a cost operator using a pre-trained model. It runs an exported
ONNX graph with ``onnxruntime`` + numpy and needs neither torch nor the
original checkpoint.
"""

from qaoa_training_pipeline.inference.ai_inference import AIInference

__all__ = ["AIInference"]
