"""embed-ab: A/B-test two embedding models on your own documents."""

from embed_ab.compare import CompareResult, compare
from embed_ab.evalset import EvalSet, EvalSetError, load_evalset
from embed_ab.providers import ProviderError

__version__ = "0.2.0"

__all__ = [
    "CompareResult",
    "EvalSet",
    "EvalSetError",
    "ProviderError",
    "__version__",
    "compare",
    "load_evalset",
]
