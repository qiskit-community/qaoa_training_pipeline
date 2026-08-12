"""Feature/target helpers shared by the AI inference path.

``rescaling_factor`` is pure numpy and is used by the torch-free ONNX runtime.
``angles_to_target`` / ``undo_gamma_rescale`` belong to the torch path and import
torch lazily so importing this module stays torch-free.
"""

# Import juliacall BEFORE torch to avoid a segfault warning on the torch path.
# See: https://github.com/pytorch/pytorch/issues/78829
try:
    import juliacall  # noqa: F401  pylint: disable=unused-import
except ImportError:
    pass  # juliacall is optional

import re
from collections import defaultdict
import numpy as np

# torch is only needed by angles_to_target / undo_gamma_rescale (the training/
# torch path). It is imported lazily inside those functions so the ONNX runtime,
# which only uses rescaling_factor (pure numpy), stays torch-free.


def rescaling_factor(cost_op):
    """Return the QAOA cost-operator rescaling factor (RMS of per-order weights)."""
    terms = defaultdict(list)

    for p in cost_op:
        order = sum(p.paulis[0].z)
        terms[order].append(np.real(p.coeffs[0]) ** 2)

    factor = 0
    for squared_weights in terms.values():
        factor += sum(squared_weights) / len(squared_weights)

    return np.sqrt(factor)


def parse_instance_name(
    instance_name: str, graph_type: str
) -> tuple[str, int, int | tuple[int, int]]:
    """
    Extract properties from instance name:
     random_regular: graph_id, node_count and regularity
     heavy_hex: graph_id, node_count and heavy hex units (as tuple)
     erdos_renyi: graph_id, node_count and percentage of edge probability

    Note: graph_type can include suffix (e.g., "heavy_hex_uniform"),
    which will be stripped to get the base type.
    """
    # For heavy_hex with any suffix, extract base type
    if graph_type.startswith("heavy_hex"):
        pattern = r"(\d{3})_(\d+)_(\d+)_heavyhex_(\d+)nodes"
        m = re.search(pattern, instance_name)
        if not m:
            raise ValueError(f"Could not parse heavy_hex instance_name={instance_name!r}")
        graph_id = m.group(1)
        node_count = int(m.group(4))
        hh_units = (int(m.group(2)), int(m.group(3)))
        return graph_id, node_count, hh_units

    elif graph_type == "random_regular":
        pattern = r"(\d{3})_(\d+)nodes_random(\d+)regular"

    elif graph_type == "erdos_renyi":
        pattern = r"(\d{3})_(\d+)nodes_erdosrenyi(\d+)percent"
    elif graph_type == "barabasi_albert":
        pattern = r"(\d{3})_(\d+)nodes_(\d{1})edges_barabasialbert"
    else:
        raise ValueError(f"Unsupported graph_type={graph_type!r}")

    m = re.search(pattern, instance_name)
    if not m:
        raise ValueError(f"Could not parse result_file_name={instance_name!r}")

    return m.group(1), int(m.group(2)), int(m.group(3))


def angles_to_target(angles: list[float], p: int, rescale_a: float) -> "torch.Tensor":
    """
    Check again if p and len(angles) matches and return torch tensor.
    Applies rescaling to the second half of angles (gammas).
    """
    import torch

    p = int(p)
    if len(angles) != 2 * p:
        raise ValueError(f"qaoa_angles must have length 2*p. Got {len(angles)} for p={p}.")

    angles = list(angles)  # ensure mutable

    # rescale second half by gamma rescale factor
    angles[p:] = [a * rescale_a for a in angles[p:]]

    return torch.tensor(angles, dtype=torch.float32)


def undo_gamma_rescale(
    angles: "torch.Tensor", p: int, rescale_a: "torch.Tensor | float"
) -> "torch.Tensor":
    """Undo the gamma rescaling on the second half of ``angles`` (divide by ``rescale_a``)."""
    import torch

    angles = angles.clone()

    if not torch.is_tensor(rescale_a):
        rescale_a = torch.tensor(rescale_a, dtype=angles.dtype, device=angles.device)
    else:
        rescale_a = rescale_a.to(dtype=angles.dtype, device=angles.device)

    if rescale_a.ndim == 1:
        rescale_a = rescale_a.unsqueeze(-1)

    angles[..., p:] = angles[..., p:] / rescale_a
    return angles
