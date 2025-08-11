#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of methods to pre-process data before QAOA parameter training."""

from .sat_mapping import SATMapper

PREPROCESSORS = {
    "sat": SATMapper,
}
