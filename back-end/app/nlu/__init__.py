"""Natural-language understanding sub-package.

Public surface:
    Intent            — intent taxonomy (StrEnum)
    NLUResult         — structured NLU output (Pydantic)
    IntentClassifier  — stateless classifier wrapping an LLM chain
    HANDLER_FOR       — dispatch table: Intent → handler function
    normalize         — text normalisation (lowercase, slang, typos)
"""

from app.nlu.intent_classifier import (
    HANDLER_FOR,
    Intent,
    IntentClassifier,
    NLUResult,
    handle_unknown,
)
from app.nlu.preprocessing import normalize

__all__ = [
    "Intent",
    "NLUResult",
    "IntentClassifier",
    "HANDLER_FOR",
    "handle_unknown",
    "normalize",
]
