"""ATHENA-R1: an AI agent for treatment reasoning over a biomedical tool universe."""

import logging as _logging
import os as _os

from ._version import __version__
from .agent import AnswerResult, AthenaR1, Backend, RoundEvent

__all__ = [
    "AthenaR1",
    "AnswerResult",
    "Backend",
    "RoundEvent",
    "__version__",
]


# Default logging setup: send agent internals to the standard logging system.
# Users can override by configuring `logging` themselves before constructing
# an `AthenaR1`. Set ATHENA_R1_LOG_LEVEL=DEBUG to see the multi-step trace.
_level_name = _os.environ.get("ATHENA_R1_LOG_LEVEL", "WARNING").upper()
_level = getattr(_logging, _level_name, _logging.WARNING)
_logger = _logging.getLogger("athena_r1")
if not _logger.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(_level)
# Stop bubbling up to the root logger — host processes that call
# logging.basicConfig() would otherwise see every athena_r1 record twice
# (once via our StreamHandler, once via root's). Library loggers that bring
# their own handler should always disable propagation.
_logger.propagate = False
