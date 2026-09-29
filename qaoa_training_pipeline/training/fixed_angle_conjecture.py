#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class that implements a parameter transfer method based on the fixed angle conjecture"""

from pathlib import Path

from qaoa_training_pipeline.training.transfer_trainer import TransferTrainer
from qaoa_training_pipeline.pre_processing.feature_extraction import GraphFeatureExtractor
from qaoa_training_pipeline.training.data_loading import LoadFromJson


class FixedAngleConjecture(TransferTrainer):
    """Fixed angle conjecture.

    This class is an interface to load the known angles of the fixed angle conjecture.
    These angles are given by the following work:

    * Jonathan Wurtz and Peter Love, Phys. Rev. A 103, 042612 (2021)
    * Jonathan Wurtz and Danylo Lykov, Phys. Rev. A 104, 052419 (2021)

    In particular, the angles loaded by the class are obtained from
    https://github.com/danlkv/fixed-angle-QAOA/blob/master/angles_regular_graphs.json

    These angles are designed for maximum cut problems on random k-regular graphs.
    The angles provided are for various QAOA depths p from 1 up to at most 11, and for k ranging
    from 3 to 11. The angles are obtained by optimizing the QAOA angles for tree graphs of the
    corresponding degree. The provided approximation ratios are lower bounds to the approximation
    ratios that can be obtained in practice.
    """

    def __init__(self, reps: int):
        """Setup the class and load the parameters."""
        project_root = Path(__file__).resolve().parent
        super().__init__(
            LoadFromJson(
                project_root / "data" / "fixed_angle_conjecture_parameter_transfer.json",
            ),
            GraphFeatureExtractor(
                extract_num_nodes=False,
                extract_num_edges=False,
                extract_avg_node_degree=True,
                extract_avg_edge_weights=False,
                extract_standard_devs=False,
                extract_density=False,
                include_one_local=False,
            ),
            reps=reps,
        )

    @classmethod
    def from_config(cls, config: dict) -> "TransferTrainer":
        """Create a class from a config."""
        return cls(
            reps=config["reps"],
        )
