#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Warning helper that attributes a warning to the caller of this package."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

PACKAGE_DIR = str(Path(__file__).resolve().parent)


def warn_at_caller(message: str, category: type[Warning] = UserWarning) -> None:
    """Emit ``message``, attributed to the first stack frame outside this package.

    The inference warnings are raised several frames deep: a sign-convention
    warning fires inside the feature extractor, which the predictor calls,
    which the provider calls. No fixed ``stacklevel`` points at the user's code
    from every one of those entry points, so a warning that is really about the
    operator the user built would be reported against ``feature_extractor.py``.
    Walking out of the package instead gets it right whether the caller came in
    through :class:`~qaoa_training_pipeline.inference.AIInference`,
    :meth:`OnnxQAOAPredictor.predict_normalized` or the extractor directly.

    Args:
        message: The warning message.
        category: The warning class to raise. Defaults to ``UserWarning``.
    """
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - interpreter without frame support
        warnings.warn(message, category, stacklevel=2)
        return

    # stacklevel=1 is this function's own warnings.warn call, so the first
    # frame above it is 2, the one above that 3, and so on.
    stacklevel = 1
    caller = frame.f_back
    while caller is not None:
        stacklevel += 1
        if not caller.f_code.co_filename.startswith(PACKAGE_DIR):
            break
        caller = caller.f_back

    warnings.warn(message, category, stacklevel=stacklevel)
