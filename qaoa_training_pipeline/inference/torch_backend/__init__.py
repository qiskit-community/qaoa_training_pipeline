#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optional PyTorch inference/export backend.

Everything here depends on torch / torch_geometric (the ``inference-torch``
extra). Nothing in this subpackage is imported at runtime by the default ONNX
path; modules are imported lazily by callers that opt into the torch backend.
"""
