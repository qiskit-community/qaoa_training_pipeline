"""Feature/target helpers for the torch-free ONNX inference path.

``rescaling_factor`` is pure numpy and is used by the ONNX runtime.
"""

import re
from collections import defaultdict
import numpy as np


def rescaling_factor(cost_op):
    """Return the QAOA cost-operator rescaling factor (RMS of per-order weights).

    The factor is used as a divisor both on input (``cost_op / rescale_a``) and
    on output (gammas ``/ rescale_a``). For a degenerate operator with no
    non-identity terms or all-zero coefficients the RMS is 0; we fall back to
    ``1.0`` so normalization/denormalization is a no-op instead of dividing by
    zero.
    """
    terms = defaultdict(list)

    for p in cost_op:
        order = sum(p.paulis[0].z)
        terms[order].append(np.real(p.coeffs[0]) ** 2)

    factor = 0
    for squared_weights in terms.values():
        factor += sum(squared_weights) / len(squared_weights)

    factor = np.sqrt(factor)
    return factor if factor > 0 else 1.0


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
