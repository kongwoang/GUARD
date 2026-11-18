"""GUARD: risk-controlled correction for frozen models under missing modalities."""

from .policy import Condition, build_candidate, random_split, run

__version__ = "1.0.0"
__all__ = ["Condition", "build_candidate", "random_split", "run"]
